import numpy as np
import jax
import jax.numpy as jnp
import custom_jax.nb_knn as nb_knn
from .tree import summarize_leaves, lvl_to_ext, get_node_box
from .common import conditional_callback

jax.ffi.register_ffi_target("IlistKNNSearch", nb_knn.IlistKNNSearch(), platform="CUDA")
jax.ffi.register_ffi_target("ConstructIlist", nb_knn.ConstructIlist(), platform="CUDA")
jax.ffi.register_ffi_target("SegmentSort", nb_knn.SegmentSort(), platform="CUDA")

def ilist_knn_search(xT, isplitT, xleaf, lvl_leaf, ilist, ir2list, ilist_splitsB, xQ=None,  isplitQ=None, k=32, boxsize=0.):
    """Finds the k nearest neighbors of xfind in the z-sorted positions xzsort
    """
    if xQ is None: xQ = xT
    if isplitQ is None: isplitQ = isplitT

    assert ir2list.shape == ilist.shape, "rilist must have the same shape as ilist"

    assert xT.dtype == xQ.dtype == jnp.float32
    assert xT.shape[-1] == xQ.shape[-1] == xleaf.shape[-1] == 3
    assert isplitT.dtype == isplitQ.dtype == jnp.int32
    assert lvl_leaf.dtype == ilist.dtype == ilist_splitsB.dtype == jnp.int32
    assert k in (4,8,12,16,32,64), "Only k=4,8,12,16,32,64 supported"

    x4a = jnp.concatenate((xT, jnp.zeros(xT.shape[:-1])[...,None]), axis=-1)
    x4b = jnp.concatenate((xQ, jnp.zeros(xQ.shape[:-1])[...,None]), axis=-1)

    x4leaf = jnp.concatenate((xleaf, lvl_leaf.view(jnp.float32)[...,None]), axis=-1)

    out_type = jax.ShapeDtypeStruct((xQ.shape[0], k, 2), jnp.int32)
    knn = jax.ffi.ffi_call("IlistKNNSearch", (out_type, ))(
        x4a, x4b, isplitT, isplitQ, x4leaf, ilist, ir2list, ilist_splitsB,
        boxsize=np.float32(boxsize)
    )[0]
    rknn, iknn = knn[...,0].view(jnp.float32), knn[...,1].view(jnp.int32)
 
    return rknn, iknn
ilist_knn_search.jit = jax.jit(ilist_knn_search, static_argnames=("k", "boxsize"))

def build_ilist_knn(xleaf, lvl_leaf, npart_leaf, isplit, node_ilist, node_ir2list, node_ilist_splits, k=32, boxsize=0., 
                    alloc_fac=128, rfac_maxbin=16.):
    assert node_ilist_splits.shape[0] == isplit.shape[0], "Should both correspond to no. of nodes+1"

    assert node_ilist.shape == node_ir2list.shape, "node_ilist and node_ir2list must have the same shape"

    x4leaf = jnp.concatenate((xleaf, lvl_leaf.view(jnp.float32)[...,None]), axis=-1)

    rbuf = jax.ShapeDtypeStruct((len(xleaf),), jnp.float32)
    leaf_ilist = jax.ShapeDtypeStruct((int(alloc_fac * len(xleaf)),), jnp.int32)
    leaf_ilist_splits = jax.ShapeDtypeStruct((len(xleaf)+1,), jnp.int32)
    leaf_ilist_rad = jax.ShapeDtypeStruct(leaf_ilist.shape, jnp.float32)

    radii, il, ir2l, ispl = jax.ffi.ffi_call("ConstructIlist", (rbuf, leaf_ilist, leaf_ilist_rad, leaf_ilist_splits))(
        x4leaf, npart_leaf, isplit, node_ilist, node_ir2list, node_ilist_splits,
        k=np.int32(k), blocksize_fill=np.uint64(32), blocksize_sort=np.uint64(64),
        rfac_maxbin=np.float32(rfac_maxbin), boxsize=np.float32(boxsize)
    )

    def myerror(n1, n2):
        raise MemoryError(f"The interaction list allocation is too small. (need: {n1} have: {n2})" +
                          f"increase alloc_fac at least by a factor of {n1/n2:.1f}")
    ispl = ispl + conditional_callback(ispl[-1] > il.size, myerror, ispl[-1], il.size)

    return il, ir2l, ispl
build_ilist_knn.jit = jax.jit(build_ilist_knn, static_argnames=["k", "boxsize", "alloc_fac"])

def offset_sum(num):
    cs = jnp.cumsum(num, axis=0)
    return cs - num, cs[-1]

def cumsum_starting_with_zero(num):
    cs = jnp.cumsum(num, axis=0)
    return jnp.concatenate([jnp.array([0], dtype=num.dtype), cs], axis=0)

def masked_prefix(mask):
    off, _ = offset_sum(mask)
    off_masked = jnp.where(mask, off, len(mask))
    return off_masked
    
def dense_ilist(num, mask=None):
    ilist = jnp.array((jnp.arange(num, dtype=jnp.int32),)*num)
    isplits = jnp.arange(num+1, dtype=jnp.int32)*num

    if mask is not None: 
        # Mask interactions with nodes > nmax
        # We write this in this way, to make sure nmax does not need to be known at compile time
        valid = mask[None,:] & mask[:,None]
        nvalid = jnp.sum(valid, axis=1)

        prefix = masked_prefix(valid.flatten())
        ilist = ilist.flatten().at[prefix].set(ilist.flatten())
        isplits = jnp.concatenate([jnp.array([0]), jnp.cumsum(nvalid)])

    return ilist.flatten(), isplits

def build_ilist_recursive(xleaf, lvleaf, nleaf, max_size=48, num_part=None, 
        refine_fac=8, k=16, stop_coarsen=2048, boxsize=0., alloc_fac=128.):
    """Recursively builds an interaction list for kNN search. This is done by recursively:
    (1) Coarsen the leaves
    (2) Get the interaction list for the coarsened leaves (recursively)
    (3) Use the interaction list of the coarsened leaves to build the finer interaction list
    """
    max_ref = (2 * num_part / max_size) / stop_coarsen

    if max_ref <= 2.: 
        # We are very close to the target level, don't coarsen any more
        # We bundle 4 leaves each
        nnodes = (len(xleaf) + 31) // 32
        inodes = jnp.arange(nnodes, dtype=jnp.int32)
        il_c, ispl_c = dense_ilist(nnodes, nleaf[32*inodes] > 0)

        ir2l_c = jnp.zeros(il_c.shape, dtype=jnp.float32)
        spl2 = 32*jnp.arange(0, nnodes + 1, dtype=jnp.int32)

        il, ir2l, ispl = build_ilist_knn(
            xleaf, lvleaf, nleaf, spl2, il_c, ir2l_c, ispl_c, alloc_fac=alloc_fac, 
            k=k, boxsize=boxsize)

        return il, ir2l, ispl

    refine_fac = min(refine_fac, max_ref)

    spl2, nleaf2, lvleaf2, xleaf2, numleaves2 = summarize_leaves(
        xleaf, max_size=max_size*refine_fac, nleaf=nleaf, num_part=num_part, ref_fac=refine_fac)
    il_c, ir2l_c, ispl_c = build_ilist_recursive(
        xleaf2, lvleaf2, nleaf2, max_size=max_size*refine_fac, num_part=num_part,
        alloc_fac=alloc_fac*np.sqrt(refine_fac), refine_fac=refine_fac, k=k, 
        stop_coarsen=stop_coarsen, boxsize=boxsize)
    
    il, ir2l, ispl = build_ilist_knn(
        xleaf, lvleaf, nleaf, spl2, il_c, ir2l_c, ispl_c, alloc_fac=alloc_fac, 
        k=k, boxsize=boxsize)
    
    return il, ir2l, ispl
build_ilist_recursive.jit = jax.jit(build_ilist_recursive, static_argnames=[
    'max_size', 'num_part', 'refine_fac', 'k', 'stop_coarsen', 'boxsize', 'alloc_fac'])

def knn(posz, k=16, boxsize=0., alloc_fac=256., max_leaf_size=48):
    spl, nleaf, llvl, xleaf, numleaves = summarize_leaves(posz, max_size=max_leaf_size)

    il, ir2l, ispl = build_ilist_recursive(xleaf, llvl, nleaf, max_size=max_leaf_size, refine_fac=15,
                                          num_part=len(posz), k=k, boxsize=boxsize, alloc_fac=alloc_fac)

    rknn, iknn = ilist_knn_search(posz, spl, xleaf, llvl, il, ir2l, ispl, k=k, boxsize=boxsize)

    return rknn, iknn
knn.jit = jax.jit(knn, static_argnames=["k", "boxsize", "alloc_fac", "max_leaf_size"])

def segment_sort(key, val, isplit, smem_size=512):
    """Sorts key/val pairs within segments defined by isplit"""
    assert key.dtype == jnp.float32
    assert val.dtype == jnp.int32
    assert isplit.dtype == jnp.int32
    assert key.shape == val.shape
    assert isplit.ndim == 1
    assert smem_size >= 64

    out_type = (jax.ShapeDtypeStruct(key.shape, key.dtype), jax.ShapeDtypeStruct(val.shape, val.dtype))
    key_sorted, val_sorted = jax.ffi.ffi_call("SegmentSort", out_type)(
        key, val, isplit, smem_size=np.int32(smem_size)
    )
    return key_sorted, val_sorted
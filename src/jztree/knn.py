import numpy as np
import jax
import jax.numpy as jnp
from jztree_cuda import ffi_knn
from .tree import pos_zorder_sort, search_sorted_z
from .common import conditional_callback, masked_prefix_sum, cumsum_starting_with_zero, inverse_indices
from .data import KNNData, KNNConfig

import fmdj

jax.ffi.register_ffi_target("IlistKNN", ffi_knn.IlistKNN(), platform="CUDA")
jax.ffi.register_ffi_target("ConstructIlist", ffi_knn.ConstructIlist(), platform="CUDA")
jax.ffi.register_ffi_target("SegmentSort", ffi_knn.SegmentSort(), platform="CUDA")

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

def ilist_knn_search(xT, isplitT, ilist, ir2list, ilist_splitsB, xQ=None,  isplitQ=None, k=32, boxsize=0.):
    """Finds the k nearest neighbors of xfind in the z-sorted positions xzsort
    """
    if xQ is None: xQ = xT
    if isplitQ is None: isplitQ = isplitT

    assert ir2list.shape == ilist.shape, "rilist must have the same shape as ilist"

    assert xT.dtype == xQ.dtype == jnp.float32
    assert xT.shape[-1] == xQ.shape[-1] == 3
    assert isplitT.dtype == isplitQ.dtype == jnp.int32
    assert ilist.dtype == ilist_splitsB.dtype == jnp.int32
    assert k in (4,8,12,16,32,64), "Only k=4,8,12,16,32,64 supported"

    x4a = jnp.concatenate((xT, jnp.zeros(xT.shape[:-1])[...,None]), axis=-1)
    x4b = jnp.concatenate((xQ, jnp.zeros(xQ.shape[:-1])[...,None]), axis=-1)

    out_type = jax.ShapeDtypeStruct((xQ.shape[0], k, 2), jnp.int32)
    knn = jax.ffi.ffi_call("IlistKNN", (out_type, ))(
        x4a, x4b, isplitT, isplitQ, ilist, ir2list, ilist_splitsB,
        boxsize=np.float32(boxsize), k=np.int32(k)
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
        key, val, isplit, smem_size=np.uint64(smem_size)
    )
    return key_sorted, val_sorted

# ------------------------------------------------------------------------------------------------ #
#                                         Helper Functions                                         #
# ------------------------------------------------------------------------------------------------ #
    
def dense_ilist(nleaves, leaf_mask=None, ngroup=32, size=None, size_ilist=None):
    num = (nleaves + ngroup - 1) // ngroup
    
    ilist = jnp.array((jnp.arange(num, dtype=jnp.int32),)*num)
    isplits = jnp.arange(num+1, dtype=jnp.int32)*num
    spl = jnp.minimum(ngroup*jnp.arange(0, num + 1, dtype=jnp.int32), nleaves)

    if leaf_mask is not None: 
        # translate to node mask (any leaf must be valid)
        lcum = cumsum_starting_with_zero(leaf_mask)
        node_mask = lcum[spl[1:]-1] > lcum[spl[:-1]]

        # Now create a reduced ilist that only contains valid interactions
        valid = node_mask[None,:] & node_mask[:,None]
        nvalid = jnp.sum(valid, axis=1)

        isplits = jnp.concatenate([jnp.array([0]), jnp.cumsum(nvalid)])

        prefix = masked_prefix_sum(valid.flatten())
        if size_ilist is None:
            buf = jnp.zeros_like(ilist.flatten())
        else:
            def err(n1, n2):
                raise MemoryError(f"Dense interaction list buffer is too small. (need: {n1} have: {n2})" +
                                  f"increase alloc_fac_ilist at least by a factor of {n1/n2:.1f}")
            isplits = isplits + conditional_callback(isplits[-1] > size_ilist, err, isplits[-1], size_ilist)

            buf = jnp.zeros(size_ilist, dtype=jnp.int32)
        ilist = buf.at[prefix].set(ilist.flatten())
        
        spl = jnp.minimum(spl, lcum[-1])

        # extend our buffers if wished:
        if size is not None:
            assert size >= nleaves
            spl = jax.lax.dynamic_update_slice(jnp.full(size+1, spl[-1], dtype=spl.dtype), spl, (0,))
            isplits = jax.lax.dynamic_update_slice(jnp.full(size+1, isplits[-1], dtype=isplits.dtype), isplits, (0,))

    ir2list = jnp.zeros(ilist.shape, dtype=jnp.float32)

    return spl, ilist.flatten(), ir2list.flatten(), isplits

# ------------------------------------------------------------------------------------------------ #
#                                      User Exposed Functions                                      #
# ------------------------------------------------------------------------------------------------ #

def evaluate_knn_z(d : KNNData, posz_query=None):
    """Evaluates the kNN for a given set of positions and precomputed interaction list
    """
    if posz_query is not None:
        # To use custom query points we need to group them by our original leaves
        # we can represent each leaf by the first of the particles inside
        xleaf = jnp.where((d.spl[1:] > d.spl[:-1])[:,None], d.posz[d.spl[:-1]], jnp.inf)
        ileaf = search_sorted_z(xleaf, posz_query, leaf_search=True)
        spl_query = jnp.searchsorted(ileaf, jnp.arange(len(xleaf)+1), side="left")
    else:
        posz_query, spl_query = None, None
    
    rnnz, innz = ilist_knn_search(
        d.posz, d.spl, d.ilist, d.ir2list, d.ilist_spl, k=d.k, boxsize=d.boxsize,
        xQ=posz_query, isplitQ=spl_query)
    
    if d.idz is not None: 
        # map back to original indices
        innz = d.idz[innz]
    
    return rnnz, innz
evaluate_knn_z.jit = jax.jit(evaluate_knn_z)

def evaluate_knn(d : KNNData, pos_query=None):
    """Evaluates the kNN for a given set of positions and precomputed interaction list
    """
    if pos_query is not None:
        posz_query, idz_query = pos_zorder_sort(pos_query)
    else:
        posz_query, idz_query = None, d.idz
    
    rnn, inn = evaluate_knn_z(d, posz_query=posz_query)

    if idz_query is not None:
        idz_inv = inverse_indices(idz_query)
        return rnn[idz_inv], inn[idz_inv]
    else:
        return rnn, inn
evaluate_knn.jit = jax.jit(evaluate_knn)

def prepare_knn_z_new(posz, k, boxsize=None, cfg : KNNConfig = KNNConfig(), idz=None) -> KNNData:
    """Prepares an instance of KNNData for a given set of positions posz
    posz is assumed to be sorted in z-order (use prepare_knn if it is not)

    if idz is given it is assumed that posz = pos0[idz] for some original pos0
    and output indices will be mapped back to original indices
    """
    boxsize = 0. if boxsize is None else boxsize

    cfg_fmdj = fmdj.Config(fmm = fmdj.config.FMMConfig(
        alloc_fac_nodes=cfg.alloc_fac_nodes,
        max_leaf_size=cfg.max_leaf_size,
        coarse_fac=cfg.rfac,
        stop_coarsen=cfg.stop_coarsen,
        multipoles_around_com=False
    ))
    posmassz = fmdj.data.PosMass(posz, jnp.ones((len(posz),), dtype=jnp.float32))
    th = fmdj.ztree.build_tree_hierarchy(posmassz, cfg_fmdj)
    
    nplanes = th.num_planes()
    valid = jnp.arange(th.plane_sizes[-1], dtype=jnp.int32) < th.lvl.num(nplanes-1)

    size = th.plane_sizes[0]
    size_ilist = int(size * cfg.alloc_fac_ilist)

    spl, il, ir2l, ispl = dense_ilist(
        th.plane_sizes[-1], valid, ngroup=32 #, size=size, size_ilist=size_ilist
    )
    
    def handle_level(i, carry):
        level = nplanes - i - 1 # have to do manual reversed loop with jax
        spl, il, ir2l, ispl = carry
        node_x = th.geom_cent.get(level, size)
        node_lvl = th.lvl.get(level, size)
        node_npart = th.npart(level, size)
        il, ir2l, ispl = build_ilist_knn(
            node_x, node_lvl, node_npart,
            spl, il, ir2l, ispl, k=k, boxsize=boxsize, alloc_fac=cfg.alloc_fac_ilist
        )
        spl = th.ispl_n2n.get(level, size+1)
        return spl, il, ir2l, ispl
    
    # This works, but it seemed to be slightly slower in tests...
    # Probably, because it requires us to use a larger list at the lowest level
    # spl, il, ir2l, ispl = jax.lax.fori_loop(
    #     0, nplanes, handle_level, (spl, il, ir2l, ispl)
    # )

    for i in range(nplanes):
        spl, il, ir2l, ispl = handle_level(i, (spl, il, ir2l, ispl))

    data = KNNData(
        k=k,
        boxsize=boxsize,
        posz=posz,
        idz=idz,
        spl=spl,
        ilist=il,
        ir2list=ir2l,
        ilist_spl=ispl
    )
    
    return data
prepare_knn_z_new.jit = jax.jit(prepare_knn_z_new, static_argnames=["k", "boxsize", "cfg"])

def prepare_knn(pos0, k, boxsize=None, cfg : KNNConfig = KNNConfig()) -> KNNData:
    """Prepares an instance of KNNData for a given set of positions pos0
    pos0 is NOT assumed to be sorted in z-order (use prepare_knnz to skip sorting)
    """
    boxsize = 0. if boxsize is None else boxsize

    posz, idz = pos_zorder_sort(pos0)

    data = prepare_knn_z_new(posz, k, boxsize=boxsize, cfg=cfg, idz=idz)

    return data
prepare_knn.jit = jax.jit(prepare_knn, static_argnames=["k", "boxsize", "cfg"])

def knn_z(posz, k=16, boxsize=0., cfg : KNNConfig = KNNConfig(), posz_query=None):
    """Finds the k nearest neighbors of posz using a kNN search with interaction list
    posz is assumed to be sorted in z-order (use "knn" if it is not)
    """
    data = prepare_knn_z_new(posz, k, boxsize=boxsize, cfg=cfg)
    rknn, iknn = evaluate_knn_z(data, posz_query=posz_query)
    return rknn, iknn
knn_z.jit = jax.jit(knn_z, static_argnames=["k", "boxsize", "cfg"])

def knn(pos0, k=16, boxsize=0., cfg : KNNConfig = KNNConfig(), pos_query=None):
    """Finds the k nearest neighbors of pos0 using a kNN search with interaction list"""
    data = prepare_knn(pos0, k, boxsize=boxsize, cfg=cfg)
    rknn, iknn = evaluate_knn(data, pos_query=pos_query)
    return rknn, iknn
knn.jit = jax.jit(knn, static_argnames=["k", "boxsize", "cfg"])
import numpy as np
import jax
import jax.numpy as jnp
import custom_jax.nb_knn as nb_knn
from .tree import summarize_leaves, lvl_to_ext, get_node_box, pos_zorder_sort, search_sorted_z
from .common import conditional_callback

jax.ffi.register_ffi_target("IlistKNNSearch", nb_knn.IlistKNNSearch(), platform="CUDA")
jax.ffi.register_ffi_target("ConstructIlist", nb_knn.ConstructIlist(), platform="CUDA")
jax.ffi.register_ffi_target("SegmentSort", nb_knn.SegmentSort(), platform="CUDA")

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
    knn = jax.ffi.ffi_call("IlistKNNSearch", (out_type, ))(
        x4a, x4b, isplitT, isplitQ, ilist, ir2list, ilist_splitsB,
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

def masked_prefix_sum(mask):
    off, _ = offset_sum(mask)
    off_masked = jnp.where(mask, off, len(mask))
    return off_masked
    
def dense_ilist(nleaves, leaf_mask=None, ngroup=32):
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

        prefix = masked_prefix_sum(valid.flatten())
        ilist = ilist.flatten().at[prefix].set(ilist.flatten())
        isplits = jnp.concatenate([jnp.array([0]), jnp.cumsum(nvalid)])
        
        spl = jnp.minimum(spl, lcum[-1])

    ir2list = jnp.zeros(ilist.shape, dtype=jnp.float32)

    return spl, ilist.flatten(), ir2list.flatten(), isplits

def build_ilist_recursive(xleaf, lvleaf, nleaf, k=16, boxsize=0., max_size=48, num_part=None, 
                          refine_fac=8, stop_coarsen=2048, alloc_fac=128., alloc_fac_nodes=1.):
    """Recursively builds an interaction list for kNN search. This is done by recursively:
    (1) Coarsen the leaves
    (2) Get the interaction list for the coarsened leaves (recursively)
    (3) Use the interaction list of the coarsened leaves to build the finer interaction list
    """
    max_ref = (2 * num_part / max_size) / stop_coarsen

    if max_ref <= 2.: 
        # We are very close to the target level, don't coarsen any more and instead use
        # a dens interaction list
        spl2, il_c, ir2l_c, ispl_c = dense_ilist(len(xleaf), nleaf > 0, ngroup=32)
    else:
        refine_fac = min(refine_fac, max_ref)

        spl2, nleaf2, lvleaf2, xleaf2, numleaves2 = summarize_leaves(
            xleaf, max_size=max_size*refine_fac, nleaf=nleaf, num_part=num_part, 
            ref_fac=refine_fac, alloc_fac_nodes=alloc_fac_nodes)
        il_c, ir2l_c, ispl_c = build_ilist_recursive(
            xleaf2, lvleaf2, nleaf2, k=k, boxsize=boxsize, max_size=max_size*refine_fac, 
            num_part=num_part, refine_fac=refine_fac, stop_coarsen=stop_coarsen, 
            alloc_fac=alloc_fac*np.sqrt(refine_fac), alloc_fac_nodes=alloc_fac_nodes)
    
    il, ir2l, ispl = build_ilist_knn(
        xleaf, lvleaf, nleaf, spl2, il_c, ir2l_c, ispl_c, k=k, boxsize=boxsize, alloc_fac=alloc_fac)
    
    return il, ir2l, ispl
build_ilist_recursive.jit = jax.jit(build_ilist_recursive, static_argnames=[
    'max_size', 'num_part', 'refine_fac', 'k', 'stop_coarsen', 'boxsize', 'alloc_fac'])

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


# =================================== User exposed functions ===================================== #

def knn_old(posz, k=16, boxsize=0., alloc_fac=256., max_leaf_size=48):
    spl, nleaf, llvl, xleaf, numleaves = summarize_leaves(posz, max_size=max_leaf_size)

    il, ir2l, ispl = build_ilist_recursive(xleaf, llvl, nleaf, max_size=max_leaf_size, refine_fac=15,
                                          num_part=len(posz), k=k, boxsize=boxsize, alloc_fac=alloc_fac)

    rknn, iknn = ilist_knn_search(posz, spl, il, ir2l, ispl, k=k, boxsize=boxsize)

    return rknn, iknn
knn_old.jit = jax.jit(knn_old, static_argnames=["k", "boxsize", "alloc_fac", "max_leaf_size"])

def inverse_indices(iargsort):
    """Given the indices that would sort an array, return the indices that would unsort it"""
    iunsort = jnp.zeros_like(iargsort)
    iunsort = iunsort.at[iargsort].set(jnp.arange(len(iargsort), dtype=iargsort.dtype))
    return iunsort

from dataclasses import dataclass
from functools import partial

@dataclass(frozen=True)
class KNNConfig:
    max_leaf_size: int = 48
    rfac : float = 8.
    alloc_fac_ilist: float = 256.
    alloc_fac_nodes: float = 1.
    stop_coarsen: int = 2048

@partial(jax.tree_util.register_dataclass, 
         meta_fields=["k", "boxsize"],
         data_fields=["posz", "idz", "spl", "ilist", "ir2list", "ilist_spl"])
@dataclass
class KNNData:
    k : int
    boxsize : float

    posz: jnp.ndarray       # z-sorted positions
    idz: jnp.ndarray        # ids so that posz = pos0[idz]
    spl: jnp.ndarray        # leaf splits so that posz[spl[i]:spl[i+1]] are in leaf i
    ilist: jnp.ndarray      # interaction list (leaf indices)
    ir2list: jnp.ndarray    # interaction r2 list (lower bound leaf-leaf distances squared)
    ilist_spl: jnp.ndarray  # leaf i interacts with leaves ilist[ilist_spl[i]:ilist_spl[i+1]]

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

def prepare_knn_z(posz, k, boxsize=None, cfg : KNNConfig = KNNConfig(), idz=None) -> KNNData:
    """Prepares an instance of KNNData for a given set of positions posz
    posz is assumed to be sorted in z-order (use prepare_knn if it is not)

    if idz is given it is assumed that posz = pos0[idz] for some original pos0
    and output indices will be mapped back to original indices
    """
    boxsize = 0. if boxsize is None else boxsize

    spl, nleaf, llvl, xleaf, numleaves = summarize_leaves(
        posz, max_size=cfg.max_leaf_size, alloc_fac_nodes=cfg.alloc_fac_nodes)
    
    il, ir2l, ispl = build_ilist_recursive(
        xleaf, llvl, nleaf, num_part=len(posz), k=k, boxsize=boxsize,
        max_size=cfg.max_leaf_size, refine_fac=cfg.rfac, alloc_fac=cfg.alloc_fac_ilist, 
        stop_coarsen=cfg.stop_coarsen, alloc_fac_nodes=cfg.alloc_fac_nodes)
    
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
prepare_knn_z.jit = jax.jit(prepare_knn_z, static_argnames=["k", "boxsize", "cfg"])

def prepare_knn(pos0, k, boxsize=None, cfg : KNNConfig = KNNConfig()) -> KNNData:
    """Prepares an instance of KNNData for a given set of positions pos0
    pos0 is NOT assumed to be sorted in z-order (use prepare_knnz to skip sorting)
    """
    boxsize = 0. if boxsize is None else boxsize

    posz, idz = pos_zorder_sort(pos0)

    data = prepare_knn_z(posz, k, boxsize=boxsize, cfg=cfg, idz=idz)

    return data
prepare_knn.jit = jax.jit(prepare_knn, static_argnames=["k", "boxsize", "cfg"])

def knn_z(posz, k=16, boxsize=0., cfg : KNNConfig = KNNConfig(), posz_query=None):
    """Finds the k nearest neighbors of posz using a kNN search with interaction list
    posz is assumed to be sorted in z-order (use "knn" if it is not)
    """
    data = prepare_knn_z(posz, k, boxsize=boxsize, cfg=cfg)
    rknn, iknn = evaluate_knn_z(data, posz_query=posz_query)
    return rknn, iknn
knn_z.jit = jax.jit(knn_z, static_argnames=["k", "boxsize", "cfg"])

def knn(pos0, k=16, boxsize=0., cfg : KNNConfig = KNNConfig(), pos_query=None):
    """Finds the k nearest neighbors of pos0 using a kNN search with interaction list"""
    data = prepare_knn(pos0, k, boxsize=boxsize, cfg=cfg)
    rknn, iknn = evaluate_knn(data, pos_query=pos_query)
    return rknn, iknn
knn.jit = jax.jit(knn, static_argnames=["k", "boxsize", "cfg"])
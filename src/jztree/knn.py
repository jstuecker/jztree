import numpy as np
import jax
import jax.numpy as jnp

from .config import KNNConfig
from .data import KNNData, PosLvl, PosLvlNum, InteractionList, PosId
from .tree import pos_zorder_sort, search_sorted_z, grouped_dense_interaction_list, build_tree_hierarchy
from .tools import inverse_indices
from .jax_ext import raise_if

from jztree_cuda import ffi_knn
jax.ffi.register_ffi_target("IlistKNN", ffi_knn.IlistKNN(), platform="CUDA")
jax.ffi.register_ffi_target("ConstructIlist", ffi_knn.ConstructIlist(), platform="CUDA")
jax.ffi.register_ffi_target("SegmentSort", ffi_knn.SegmentSort(), platform="CUDA")

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

def ilist_knn_search(ilist: InteractionList, isplitT, xT, xQ=None,  isplitQ=None, k=32, boxsize=0.):
    """Finds the k nearest neighbors of xfind in the z-sorted positions xzsort
    """
    if xQ is None: xQ = xT
    if isplitQ is None: isplitQ = isplitT

    assert ilist.rad2.shape == ilist.iother.shape, "rilist must have the same shape as ilist"

    assert xT.dtype == xQ.dtype == jnp.float32
    assert xT.shape[-1] == xQ.shape[-1] == 3
    assert isplitT.dtype == isplitQ.dtype == jnp.int32
    assert ilist.iother.dtype == ilist.ispl.dtype == jnp.int32
    assert k in (4,8,12,16,32,64), "Only k=4,8,12,16,32,64 supported"

    x4a = jnp.concatenate((xT, jnp.zeros(xT.shape[:-1])[...,None]), axis=-1)
    x4b = jnp.concatenate((xQ, jnp.zeros(xQ.shape[:-1])[...,None]), axis=-1)

    out_type = jax.ShapeDtypeStruct((xQ.shape[0], k, 2), jnp.int32)
    knn = jax.ffi.ffi_call("IlistKNN", (out_type, ))(
        x4a, x4b, isplitT, isplitQ, ilist.iother, ilist.rad2, ilist.ispl,
        boxsize=np.float32(boxsize), k=np.int32(k)
    )[0]
    rknn, iknn = knn[...,0].view(jnp.float32), knn[...,1].view(jnp.int32)
 
    return rknn, iknn
ilist_knn_search.jit = jax.jit(ilist_knn_search, static_argnames=("k", "boxsize"))

def knn_node2node_ilist(ilist: InteractionList, spl_parent: jax.Array, node_data: PosLvlNum,
                  k: int = 32, boxsize: float = 0., rfac_maxbin: float = 16.) -> InteractionList:
    boxsize = 0. if boxsize is None else boxsize

    assert ilist.ispl.shape[0] == spl_parent.shape[0], "Should both correspond to no. of nodes+1"
    assert ilist.iother.shape == ilist.rad2.shape, "node_ilist and node_ir2list must have the same shape"
    assert ilist.size() < 2**31, f"Ilist allocation is in overflow danger {ilist.size()/2**31}"

    size = len(node_data.poslvl.pos)
    x4leaf = node_data.poslvl.pos_lvl()

    outputs = (
        jax.ShapeDtypeStruct((size,), jnp.float32), # radii buffer (temporary)
        jax.ShapeDtypeStruct((ilist.size(),), jnp.int32), # ilist
        jax.ShapeDtypeStruct((ilist.size(),), jnp.float32), # ilist radii
        jax.ShapeDtypeStruct((size+1,), jnp.int32) # ilist splits
    )

    radii, il, ilr2, ispl = jax.ffi.ffi_call("ConstructIlist", outputs)(
        x4leaf, node_data.npart, spl_parent, ilist.iother, ilist.rad2, ilist.ispl,
        k=np.int32(k), blocksize_fill=np.uint64(32), blocksize_sort=np.uint64(64),
        rfac_maxbin=np.float32(rfac_maxbin), boxsize=np.float32(boxsize)
    )

    ispl = ispl + raise_if(ispl[-1] > il.size,
        "The interaction list allocation is too small. (need: {n1}, have: {n2})\n"
        "Hint: increase alloc_fac_ilist at least by a factor of {ratio:.1f}",
        n1=ispl[-1], n2=il.size, ratio=ispl[-1]/il.size
    )

    return InteractionList(ispl, il, rad2=ilr2)
knn_node2node_ilist.jit = jax.jit(knn_node2node_ilist, static_argnames=["k", "boxsize", "rfac_maxbin"])

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
#                                      User Exposed Functions                                      #
# ------------------------------------------------------------------------------------------------ #

def evaluate_knn_z(d : KNNData, posz_query=None):
    """Evaluates the kNN for a given set of positions and precomputed interaction list
    """
    if posz_query is not None:
        # To use custom query points we need to group them by our original leaves
        # we can represent each leaf by the first of the particles inside
        xleaf = jnp.where((d.spl[1:] > d.spl[:-1])[:,None], d.partz.pos[d.spl[:-1]], jnp.inf)
        ileaf = search_sorted_z(xleaf, posz_query, leaf_search=True)
        spl_query = jnp.searchsorted(ileaf, jnp.arange(len(xleaf)+1), side="left")
    else:
        posz_query, spl_query = None, None
    
    rnnz, innz = ilist_knn_search(
        d.ilist, d.spl, d.partz.pos, k=d.k, boxsize=d.boxsize,
        xQ=posz_query, isplitQ=spl_query)
    
    if d.partz.id is not None: 
        # map back to original indices
        innz = d.partz.id[innz]
    
    return rnnz, innz
evaluate_knn_z.jit = jax.jit(evaluate_knn_z)

def evaluate_knn(d : KNNData, pos_query=None):
    """Evaluates the kNN for a given set of positions and precomputed interaction list
    """
    if pos_query is not None:
        posz_query, idz_query = pos_zorder_sort(pos_query)
    else:
        posz_query, idz_query = None, d.partz.id
    
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
    th = build_tree_hierarchy(posz, cfg.tree)
    
    nlevels = th.num_planes()
    size = th.base_size()

    # initialize top-level interaction list
    spl, ilist, nsup = grouped_dense_interaction_list(
        th.lvl.num(nlevels-1), int(size*cfg.alloc_fac_ilist), ngroup=32, size_super=size
    )
    # Define super-node data
    spl_n2n = th.ispl_n2n.append(spl, nsup+1, fill_value=spl[-1], resize=True)
    ilist.rad2 = jnp.zeros(ilist.iother.shape, dtype=jnp.float32)

    def handle_level(i, ilist: InteractionList):
        level = nlevels - i - 1
        
        spl_parent = spl_n2n.get(level+1, size+1)

        node_pos_lvl = PosLvl(pos=th.geom_cent.get(level, size), lvl=th.lvl.get(level, size))
        node_data = PosLvlNum(node_pos_lvl, npart=th.npart(level, size))

        return knn_node2node_ilist(ilist, spl_parent, node_data, k=k, boxsize=boxsize)
    
    ilist = jax.lax.fori_loop(0, nlevels, handle_level, ilist)

    data = KNNData(
        k=k,
        boxsize=boxsize,
        partz=PosId(pos=posz, id=idz),
        spl=spl_n2n.get(0, size+1),
        ilist=ilist
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
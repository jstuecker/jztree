from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .config import KNNConfig
from .data import KNNData, PosLvl, PosLvlNum, InteractionList, PosId, TreeHierarchy, PackedArray
from .tree import pos_zorder_sort, search_sorted_z, grouped_dense_interaction_list, build_tree_hierarchy
from .tree import distr_grouped_dense_interaction_list, simplify_interaction_list
from .tools import inverse_indices
from .jax_ext import raise_if, pcast_vma, pcast_like, shard_map_constructor
from .comm import get_rank_info, all_to_all_request_children

from jztree_cuda import ffi_knn
jax.ffi.register_ffi_target("KnnLeaf2Leaf", ffi_knn.KnnLeaf2Leaf(), platform="CUDA")
jax.ffi.register_ffi_target("KnnNode2Node", ffi_knn.KnnNode2Node(), platform="CUDA")
jax.ffi.register_ffi_target("SegmentSort", ffi_knn.SegmentSort(), platform="CUDA")

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

def _knn_leaf2leaf(ilist: InteractionList, splT, xT, splQ=None, xQ=None, k=32, boxsize=0.):
    """Finds the k nearest neighbors of xfind in the z-sorted positions xzsort
    """
    if xQ is None: xQ = xT
    if splQ is None: splQ = splT

    assert ilist.rad2.shape == ilist.iother.shape, "rilist must have the same shape as ilist"

    assert xT.dtype == xQ.dtype == jnp.float32
    assert xT.shape[-1] == xQ.shape[-1] == 3
    assert splT.dtype == splQ.dtype == jnp.int32
    assert ilist.iother.dtype == ilist.ispl.dtype == jnp.int32
    assert k in (4,8,12,16,32,64), "Only k=4,8,12,16,32,64 supported"

    x4a = jnp.concatenate((xT, jnp.zeros(xT.shape[:-1])[...,None]), axis=-1)
    x4b = jnp.concatenate((xQ, jnp.zeros(xQ.shape[:-1])[...,None]), axis=-1)

    out_type = jax.ShapeDtypeStruct((xQ.shape[0], k, 2), jnp.int32)
    knn = jax.ffi.ffi_call("KnnLeaf2Leaf", (out_type, ))(
        ilist.ispl, ilist.iother, ilist.rad2, splT, x4a, splQ, x4b,
        boxsize=np.float32(boxsize), k=np.int32(k)
    )[0]
    rknn, iknn = knn[...,0].view(jnp.float32), knn[...,1].view(jnp.int32)
 
    return rknn, iknn
_knn_leaf2leaf.jit = jax.jit(_knn_leaf2leaf, static_argnames=("k", "boxsize"))

def _knn_node2node_ilist(ilist: InteractionList, spl_parent: jax.Array, node_data: PosLvlNum,
                  k: int = 32, boxsize: float = 0., rfac_maxbin: float = 16.) -> InteractionList:
    boxsize = 0. if boxsize is None else boxsize

    assert ilist.ispl.shape[0] == spl_parent.shape[0], "Should both correspond to no. of nodes+1"
    assert ilist.iother.shape == ilist.rad2.shape, "node_ilist and node_ir2list must have the same shape"
    assert ilist.size() < 2**31, f"Ilist allocation is in overflow danger {ilist.size()/2**31}"

    size = len(node_data.pos)
    poslvl = node_data.pos_lvl()

    outputs = (
        jax.ShapeDtypeStruct((size,), jnp.float32), # radii buffer (temporary)
        jax.ShapeDtypeStruct((size+1,), jnp.int32), # ilist splits
        jax.ShapeDtypeStruct((ilist.size(),), jnp.int32), # ilist
        jax.ShapeDtypeStruct((ilist.size(),), jnp.float32) # ilist radii
    )

    radii, ispl, il, ilr2 = jax.ffi.ffi_call("KnnNode2Node", outputs)(
        ilist.ispl, ilist.iother, ilist.rad2, spl_parent, poslvl, node_data.npart,
        k=np.int32(k), blocksize_fill=np.uint64(32), blocksize_sort=np.uint64(64),
        rfac_maxbin=np.float32(rfac_maxbin), boxsize=np.float32(boxsize)
    )

    # Mark as varying per process if input is varying
    radii, ispl, il, ilr2 = jax.tree.map(
        lambda x: pcast_like(x, spl_parent), (radii, ispl, il, ilr2)
    )

    ispl = ispl + raise_if(ispl[-1] > il.size,
        "The interaction list allocation is too small. (need: {n1}, have: {n2})\n"
        "Hint: increase alloc_fac_ilist at least by a factor of {ratio:.1f}",
        n1=ispl[-1], n2=il.size, ratio=ispl[-1]/il.size
    )

    return InteractionList(ispl, il, rad2=ilr2)
_knn_node2node_ilist.jit = jax.jit(_knn_node2node_ilist, static_argnames=["k", "boxsize", "rfac_maxbin"])

def _segment_sort(spl, key, val, smem_size=512):
    """Sorts key/val pairs within segments defined by isplit"""
    assert key.dtype == jnp.float32
    assert val.dtype == jnp.int32
    assert spl.dtype == jnp.int32
    assert key.shape == val.shape
    assert spl.ndim == 1
    assert smem_size >= 64

    out_type = (jax.ShapeDtypeStruct(key.shape, key.dtype), jax.ShapeDtypeStruct(val.shape, val.dtype))
    key_sorted, val_sorted = jax.ffi.ffi_call("SegmentSort", out_type)(
        spl, key, val, smem_size=np.uint64(smem_size)
    )
    return key_sorted, val_sorted

# ------------------------------------------------------------------------------------------------ #
#                                          Dual Tree Walk                                          #
# ------------------------------------------------------------------------------------------------ #

def _knn_dual_walk(th: TreeHierarchy, k: int, boxsize: float | None = None, 
                   alloc_fac_ilist: float = 32.) -> InteractionList:
    rank, ndev, axis_name = get_rank_info()

    nlevels = th.num_planes()
    size = th.size()

    # initialize top-level interaction list
    if ndev > 1:
        spl, ilist, nsup = distr_grouped_dense_interaction_list(
            th.num(nlevels-1), size, size_ilist=int(th.size_leaves*alloc_fac_ilist)
        )
        ilist.rad2 = pcast_like(jnp.zeros(ilist.iother.shape, dtype=jnp.float32), ilist.iother)
    else:
        spl, ilist, nsup = grouped_dense_interaction_list(
            th.lvl.num(nlevels-1), int(th.size_leaves*alloc_fac_ilist), ngroup=32, size_super=size
        )
        ilist.rad2 = jnp.zeros(ilist.iother.shape, dtype=jnp.float32)
    
    # include splitting points for top-level nodes
    spl_n2n = th.ispl_n2n.append(spl, nsup+1, fill_value=spl[-1], resize=True)

    def handle_level(i, ilist: InteractionList):
        level = nlevels - i - 1
        
        parent_spl = spl_n2n.get(level+1, size+1)

        node_data = PosLvlNum(
            pos=th.geom_cent.get(level, size),
            lvl=th.lvl.get(level, size),
            npart=th.npart(level, size)
        )

        if ndev > 1:
            # Request the remote node data that we need to interact with
            (node_data, ids), parent_spl, dev_spl = all_to_all_request_children(
                ilist.dev_spl, ilist.ids, parent_spl, (node_data, jnp.arange(size)),
                axis_name=axis_name, err_hint="\nHint: increase alloc_fac_nodes"
            )

        ilist = _knn_node2node_ilist(ilist, parent_spl, node_data, k=k, boxsize=boxsize)

        if ndev > 1:
            # simplify interaction list to remove unused remotes
            ilist = replace(ilist, ids=ids, dev_spl=dev_spl)
            ilist = simplify_interaction_list(ilist)
        
        return ilist

    ilist = jax.lax.fori_loop(0, nlevels, handle_level, ilist)
    
    return ilist
_knn_dual_walk.jit = jax.jit(_knn_dual_walk, static_argnames=["k", "boxsize", "alloc_fac_ilist"])
_knn_dual_walk.smap = shard_map_constructor(_knn_dual_walk,
    in_specs=(P(-1), None, None, None),
    static_argnames=("k", "boxsize", "alloc_fac_ilist")
)

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
    
    rnnz, innz = _knn_leaf2leaf(
        d.ilist, d.spl, d.partz.pos, k=d.k, boxsize=d.boxsize, splQ=spl_query, xQ=posz_query
    )
    
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

def prepare_knn_z(posz, k, boxsize=None, cfg : KNNConfig = KNNConfig(), idz=None) -> KNNData:
    """Prepares an instance of KNNData for a given set of positions posz
    posz is assumed to be sorted in z-order (use prepare_knn if it is not)

    if idz is given it is assumed that posz = pos0[idz] for some original pos0
    and output indices will be mapped back to original indices
    """
    th: TreeHierarchy = build_tree_hierarchy(posz, cfg.tree)
    
    ilist = _knn_dual_walk(th, k, boxsize=boxsize, alloc_fac_ilist=cfg.alloc_fac_ilist)

    return KNNData(
        k=k,
        boxsize=boxsize,
        partz=PosId(pos=posz, id=idz),
        spl=th.ispl_n2n.get(0, th.size()+1),
        ilist=ilist
    )
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

# ------------------------------------------------------------------------------------------------ #
#                                          Distributed KNN                                         #
# ------------------------------------------------------------------------------------------------ #

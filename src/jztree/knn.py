from dataclasses import replace
import numpy as np
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .config import KNNConfig
from .data import KNNData, PosLvl, PosLvlNum, InteractionList, PosId, TreeHierarchy, DistrKNNData, RankIdx
from .data import Pos, RankIdx, get_pos
from .tree import pos_zorder_sort, search_sorted_z, grouped_dense_interaction_list, build_tree_hierarchy
from .tree import distr_grouped_dense_interaction_list, simplify_interaction_list, distr_zsort_and_tree
from .tools import inverse_indices, inverse_of_splits, masked_to_dense, masked_scatter, masked_inverse
from .jax_ext import raise_if, pcast_vma, pcast_like, shard_map_constructor, tree_map_by_len
from .comm import get_rank_info, all_to_all_request_children, all_to_all_with_irank

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
                  k: int = 32, boxsize: float = 0., rfac_maxbin: float = 16., mode: int = 0) -> InteractionList:
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
        rfac_maxbin=np.float32(rfac_maxbin), boxsize=np.float32(boxsize),
        mode=np.int32(mode)
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

        ilist_in = ilist
        ilist = _knn_node2node_ilist(ilist_in, parent_spl, node_data, k=k, boxsize=boxsize, mode=0)

        # ilist = _knn_node2node_ilist(ilist_in, parent_spl, node_data, k=k, boxsize=boxsize, mode=1)
        # jax.debug.log("{}", ilist.ispl)
        # jax.debug.log("{} {}", ilist.ispl, ilist2.ispl)

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
evaluate_knn_z.smap = shard_map_constructor(evaluate_knn_z)

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

def distr_prepare_knn_z(posz, th: TreeHierarchy, k, boxsize=None, cfg : KNNConfig = KNNConfig(), idz=None) -> KNNData:
    rank, ndev, axis_name = get_rank_info()

    ilist = _knn_dual_walk(th, k, boxsize=boxsize, alloc_fac_ilist=cfg.alloc_fac_ilist)

    pdata = PosId(pos=posz, id=idz)

    spl = th.ispl_n2n.get(0, th.size()+1)
    (pdata, ids), spl, dev_spl = all_to_all_request_children(
        ilist.dev_spl, ilist.ids, spl, (pdata, jnp.arange(len(posz))),
        axis_name=axis_name, err_hint="\nHint: increase padding."
    )

    return DistrKNNData(
        k=k,
        boxsize=boxsize,
        partz=pdata,
        spl=spl,
        ilist=ilist,
        origin = RankIdx(rank=inverse_of_splits(dev_spl, len(posz)), idx=ids)
    )
distr_prepare_knn_z.smap = shard_map_constructor(distr_prepare_knn_z,
    in_specs=(P(-1), P(-1), None, None, None, P(-1)),
    static_argnames=("k", "boxsize", "cfg")
)

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

def distr_knn(
        part: jax.Array | Pos,
        k: int,
        boxsize: float | None = None,
        th: TreeHierarchy | None = None,
        result: str | jax.Array | Any = "rad_origin",
        reduce_func: Callable | None = None,
        output_order: str = "input",
        cfg: KNNConfig = KNNConfig()
    ):
    assert output_order in ("z", "input", "remote")

    rank, ndev, axis_name = get_rank_info()

    size = len(get_pos(part))

    origin = RankIdx(jnp.full(size, rank, dtype=jnp.int32), jnp.arange(size, dtype=jnp.int32))

    if th is None:
        if output_order == "input": # keep track of pre-sort origin
            partz, origin, th = distr_zsort_and_tree(part, cfg.tree, data=origin)
            origin_cts = jax.lax.all_gather(part.num, axis_name=axis_name)
        else:
            partz, th = distr_zsort_and_tree(part, cfg.tree)
            origin_cts = jax.lax.all_gather(partz.num, axis_name=axis_name)
    else:
        # particles must already be sorted
        # put a sorted check here later !!!
        partz = part
        origin_cts = jax.lax.all_gather(part.num, axis_name=axis_name)

    # Build Leaf-leaf interaction list through dual tree walk
    ilist = _knn_dual_walk(th, k, boxsize=boxsize, alloc_fac_ilist=cfg.alloc_fac_ilist)

    # Localize particle data
    spl = th.ispl_n2n.get(0, th.size()+1)
    (premote, origin_rem), spl, dev_spl = all_to_all_request_children(
        ilist.dev_spl, ilist.ids, spl, (partz, origin),
        axis_name=axis_name, err_hint="\nHint: increase padding."
    )

    # Evaluate knn
    rnnz, innz = _knn_leaf2leaf(ilist, spl, get_pos(premote), k=k, boxsize=boxsize)
    
    res = []
    if type(result) == str:
        res_keys = result.split("_")
        for key in res_keys:
            if key == "rad":
                res.append(rnnz)
            elif key == "drad": # Same as rad, but result will be differentible
                x = get_pos(premote)
                res.append(jnp.linalg.norm(x[:,None] - x[innz], axis=-1))
            elif key == "rankidx":
                nn_origin = jax.tree.map(lambda x: x[innz], origin_rem)
                res.append(nn_origin)
            elif key == "globalidx":
                # dtype = jnp.int64 if ndev > 1 else jnp.int32  !!! address this later
                dtype = jnp.int32
                # with jax.enable_x64():
                dev_offsets = jnp.cumsum(origin_cts, dtype=dtype) - origin_cts.astype(dtype)
                gidx = dev_offsets[origin_rem.rank].astype(dtype) + origin_rem.idx.astype(dtype)
                res.append(gidx[innz])
            elif key == "part":
                res.append(tree_map_by_len(lambda x: x[innz], premote, size))
            elif key == "reduce":
                assert reduce_func is not None, "Please provide reduce_func if using 'reduce'"
                x = reduce_func(part=premote, rnn=rnnz, inn=innz, origin=origin)
                assert len(x) == len(rnnz)
                res.append(x)
            elif hasattr(premote, key):
                arr = getattr(premote, key)
                res.append(tree_map_by_len(lambda x: x[innz], arr, size))
            else:
                raise ValueError(f"Invalid result key {key}")
    else:
        raise NotImplementedError()
    
    if len(res) == 1:
        res = res[0]
    else:
        res = tuple(res)
    
    # Now put results in desired output order
    if output_order == "remote":
        return res

    # with jax.enable_x64():
    idx = jnp.arange(size, dtype=jnp.int32)
    mask = (idx >= dev_spl[rank]) & (idx < dev_spl[rank+1])
    res, num = masked_to_dense(res, mask)

    if output_order == "z":
        return res
    elif output_order == "input":
        (res, idx), dev_spl = all_to_all_with_irank(
            origin.rank, (res, origin.idx), num=num, axis_name=axis_name,
            err_hint="\nThis should never fail..."
        )
        inverse = masked_inverse(idx, mask=jnp.arange(len(idx)) < dev_spl[-1])

        return tree_map_by_len(lambda x: x[inverse], res, size)
    else:
        raise ValueError(f"Unknown output order {output_order}")
distr_knn.smap = shard_map_constructor(distr_knn,
    in_specs=(P(-1), None, None, P(-1), None, None, None, None),
    static_argnames=("k", "boxsize", "result", "reduce_func", "output_order", "cfg")
)
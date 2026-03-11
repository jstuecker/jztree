from dataclasses import replace
import numpy as np
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .config import KNNConfig
from .data import KNNData, PosLvl, PosLvlNum, InteractionList, PosId, TreeHierarchy, DistrKNNData, RankIdx
from .data import Pos, RankIdx, get_pos, get_num, get_num_total
from .tree import pos_zorder_sort, grouped_dense_interaction_list, zsort_and_tree_multi_type
from .tree import distr_grouped_dense_interaction_list, simplify_interaction_list, zsort_and_tree
from .tools import inverse_indices, inverse_of_splits, masked_to_dense, masked_scatter, masked_inverse
from .jax_ext import raise_if, pcast_vma, pcast_like, shard_map_constructor, tree_map_by_len
from .comm import get_rank_info, all_to_all_request_children, all_to_all_with_irank
from .stats import stats_callback, AllocStats, InteractionStats

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

    assert xT.dtype == xQ.dtype
    assert xT.shape[-1] == xQ.shape[-1]
    assert splT.dtype == splQ.dtype == jnp.int32
    assert ilist.iother.dtype == ilist.ispl.dtype == jnp.int32
    assert len(ilist.ispl) == len(splQ)

    dtype = xT.dtype

    if k > 32:
        tmp_buf = jax.ShapeDtypeStruct((xQ.shape[0], dtype.itemsize + 4), jnp.uint8)
    else:
        tmp_buf = jax.ShapeDtypeStruct((), jnp.uint4)

    outputs = (
        jax.ShapeDtypeStruct((xQ.shape[0], k), xT.dtype),
        jax.ShapeDtypeStruct((xQ.shape[0], k), jnp.int32),
        tmp_buf
    )

    rnn, inn = jax.ffi.ffi_call("KnnLeaf2Leaf", outputs)(
        ilist.ispl, ilist.iother, ilist.rad2, splT, xT, splQ, xQ,
        boxsize=np.float32(boxsize), k=np.int32(k)
    )[0:2]
    return rnn, inn
_knn_leaf2leaf.jit = jax.jit(_knn_leaf2leaf, static_argnames=("k", "boxsize"))

def _knn_node2node_ilist(
        ilist: InteractionList,
        spl_parent: jax.Array,
        node_data: PosLvlNum,
        k: int = 32,
        boxsize: float = 0.) -> InteractionList:
    boxsize = 0. if boxsize is None else boxsize

    dtype = node_data.pos.dtype
    assert dtype in (jnp.float32, jnp.float64)

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
        boxsize=np.float32(boxsize), dim=np.int32(node_data.pos.shape[-1])
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

    stats_callback("allocation", AllocStats.record_filled_interactions, ispl[-1], il.size)
    stats_callback(
        "interaction", InteractionStats.record_largest_interaction, jnp.max(ispl[1:] - ispl[:-1])
    )

    return InteractionList(ispl, il, rad2=ilr2)
_knn_node2node_ilist.jit = jax.jit(_knn_node2node_ilist, static_argnames=["k", "boxsize"])

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
            npart=th.npart(level, size=size, ptype=0)
        )

        if ndev > 1:
            # Request the remote node data that we need to interact with
            (node_data, ids), parent_spl, dev_spl = all_to_all_request_children(
                ilist.dev_spl, ilist.ids, parent_spl, (node_data, jnp.arange(size)),
                axis_name=axis_name, err_hint_child="\nHint: increase alloc_fac_nodes",
                err_hint_parent="\nHint: increase alloc_fac_nodes"
            )
            stats_callback("allocation", AllocStats.record_filled_nodes_interaction, dev_spl[-1], size)

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
#                                       User Exposed Function                                      #
# ------------------------------------------------------------------------------------------------ #

def knn(
        part: jax.Array | Pos,
        k: int,
        boxsize: float | None = None,
        th: TreeHierarchy | None = None,
        part_query: jax.Array | Pos | None = None,
        result: str | jax.Array | Any = "rad_globalidx",
        reduce_func: Callable | None = None,
        output_order: str = "input",
        cfg: KNNConfig = KNNConfig()
    ):
    assert output_order in ("z", "input")

    rank, ndev, axis_name = get_rank_info()

    def init_origin(part):
        size = len(get_pos(part))
        ridx = RankIdx(
            rank = jnp.full(size, rank, dtype=jnp.int32) if ndev > 1 else None,
            idx = jnp.arange(size, dtype=jnp.int32)
        )
        return ridx, get_num(part, default_to_length=(ndev==1))

    origin, num_origin = init_origin(part)

    # Reorder particles, build tree and keep track of the original task and index of each particle
    if th is not None:
        assert part_query is None, "Querying with pre-sorted particles is not supported yet"
        partz = partz_q = part
        origin_q, num_origin_q = origin, num_origin
    elif part_query is None:
        partz, origin, th = zsort_and_tree(part, cfg.tree, data=origin)
        partz_q, origin_q, num_origin_q = partz, origin, num_origin
    else:
        origin_q, num_origin_q = init_origin(part_query)
        (partz, partz_q), (origin, origin_q), th = zsort_and_tree_multi_type(
            (part, part_query), cfg_tree=cfg.tree, data=(origin, origin_q)
        )

    if output_order == "z": # overwrite origin with the z-order location
        origin, num_origin = init_origin(partz)
        origin_q, num_origin_q = init_origin(partz_q)

    origin_cts = jax.lax.all_gather(num_origin, axis_name=axis_name)

    # Build Leaf-leaf interaction list through dual tree walk
    ilist = _knn_dual_walk(th, k, boxsize=boxsize, alloc_fac_ilist=cfg.alloc_fac_ilist)

    # Request particle data for interactions
    spl = th.splits_leaf_to_part(ptype=0)
    if ndev > 1:
        (partz_req, origin_req), spl, dev_spl = all_to_all_request_children(
            ilist.dev_spl, ilist.ids, spl, (partz, origin), axis_name=axis_name,
            err_hint_parent="\nHint: increase alloc_fac_nodes.",
            err_hint_child="\nHint: increase padding."
        )

        stats_callback("allocation", AllocStats.record_filled_part_interactions,
            dev_spl[-1], len(origin_req.idx)
        )

        ilist = ilist.without_remote_query_points(rank)
    else:
        partz_req, origin_req = partz, origin

    spl_q = th.splits_leaf_to_part(ptype = 0 if part_query is None else 1)

    # Evaluate knn
    rnnz, innz = _knn_leaf2leaf(
        ilist, spl, get_pos(partz_req), k=k, boxsize=boxsize, splQ=spl_q, xQ=get_pos(partz_q)
    )
    
    # Infer requested result data
    if type(result) != str:
        raise NotImplementedError("result must be a string")
    
    res = []
    res_keys = result.split("_")
    for key in res_keys:
        if key == "rad":
            res.append(rnnz)
        elif key == "drad": # Same as rad, but result will be differentible
            x, xq = get_pos(partz_req), get_pos(partz_q)
            res.append(jnp.linalg.norm(xq[:,None] - x[innz], axis=-1))
        elif key == "rankidx":
            res.append(jax.tree.map(lambda x: x[innz], origin_req))
        elif key == "globalidx":
            if get_num_total(part, default_to_length=(ndev==1)) >= 2**31:
                raise ValueError("I have > 2**31 particles, globalidx will overflow. Use rankidx instead!")
            if ndev > 1:
                dev_offsets = jnp.cumsum(origin_cts) - origin_cts
                gidx = dev_offsets[origin_req.rank] + origin_req.idx
                res.append(gidx[innz])
            else:
                res.append(origin_req.idx[innz])
        elif key == "part":
            res.append(tree_map_by_len(lambda x: x[innz], partz_req, len(origin_req.idx)))
        elif key == "reduce":
            assert reduce_func is not None, "Please provide reduce_func if using 'reduce'"
            x = reduce_func(part=partz_req, rnn=rnnz, inn=innz, origin=origin)
            assert (len(x) == len(rnnz)) or (output_order=="z")
            res.append(x)
        elif hasattr(partz_req, key):
            arr = getattr(partz_req, key)
            res.append(tree_map_by_len(lambda x: x[innz], arr, len(origin_req.idx)))
        else:
            raise ValueError(f"Invalid result key {key}")
    
    res = tuple(res) if len(res) > 1 else res[0]

    if output_order == "input":
        if ndev > 1:
            (res, idx), dev_spl = all_to_all_with_irank(
                origin_q.rank, (res, origin_q.idx), num=partz_q.num, axis_name=axis_name,
                err_hint="\nThis should never fail..."
            )
        else:
            idx = origin_q.idx
        inverse = masked_inverse(idx, mask=jnp.arange(len(idx)) < num_origin_q)

        res = tree_map_by_len(lambda x: x[inverse], res, len(origin_q.idx))
    
    return res
knn.jit = jax.jit(knn,
    static_argnames=("k", "boxsize", "result", "reduce_func", "output_order", "cfg")
)
knn.smap = shard_map_constructor(knn,
    in_specs=(P(-1), None, None, P(-1), P(-1), None, None, None, None),
    static_argnames=("k", "boxsize", "result", "reduce_func", "output_order", "cfg")
)
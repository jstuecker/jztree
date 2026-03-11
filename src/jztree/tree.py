import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from typing import Tuple, Any
from dataclasses import replace

from .data import Pos, PosMass, PackedArray, TreeHierarchy, InteractionList, LevelInfo
from .data import get_num_total, get_pos, get_num, verify_ilist, same_width_int
from .config import TreeConfig, RegularizationConfig
from .tools import cumsum_starting_with_zero, div_ceil, inverse_of_splits
from .comm import send_to_left, send_to_right, shift_particles_left
from .comm import all_to_all_with_splits, global_splits, all_to_all_with_irank
from .jax_ext import pcast_like, get_rank_info, tree_map_by_len, raise_if, shard_map_constructor
from .jax_ext import concatenate_pytrees, separate_pytrees
from .stats import statistics, stats_callback, AllocStats

from jztree_cuda import ffi_tree, ffi_sort

jax.ffi.register_ffi_target("PosZorderSort", ffi_sort.PosZorderSort(), platform="CUDA")
jax.ffi.register_ffi_target("SearchSortedZ", ffi_sort.SearchSortedZ(), platform="CUDA")
jax.ffi.register_ffi_target("FlagLeafBoundaries", ffi_tree.FlagLeafBoundaries(), platform="CUDA")
jax.ffi.register_ffi_target("FindNodeBoundaries", ffi_tree.FindNodeBoundaries(), platform="CUDA")
jax.ffi.register_ffi_target("GetNodeGeometry", ffi_tree.GetNodeGeometry(), platform="CUDA")
jax.ffi.register_ffi_target("GetBoundaryExtendPerLevel", ffi_tree.GetBoundaryExtendPerLevel(), platform="CUDA")
jax.ffi.register_ffi_target("CenterOfMass", ffi_tree.CenterOfMass(), platform="CUDA")

# ------------------------------------------------------------------------------------------------ #
#                                         Helper Functions                                         #
# ------------------------------------------------------------------------------------------------ #

def lvl_to_lvlvec(level: jax.Array, dim: int, stacked: bool = True):
    olvl = level // dim
    omod = level - olvl*dim

    lvec = []
    for i in range(0, dim):
        lvec.append(olvl + (omod >= dim-i).astype(jnp.int32))
    if stacked:
        return jnp.stack(lvec, axis=-1)
    else:
        return lvec

def lvl_to_ext(level: jax.Array, linfo: LevelInfo):
    lvec = lvl_to_lvlvec(level, linfo.dim)
    return jnp.astype(2., linfo.dtype) ** lvec

def get_node_box(x, level_binary):
    raise NotImplemented("implementation broken")
    node_size = lvl_to_ext(level_binary)
    node_cent = x + jnp.sign(x)*(0.5 * node_size - jnp.mod(jnp.abs(x), node_size))
    return node_cent, node_size

def level_histogram(lvl: jax.Array, linfo: LevelInfo, weights: jax.Array | None = None,
                    dtype: jnp.dtype = jnp.float32):
    rank, ndev, axis_name = get_rank_info()

    nlvl = linfo.max_lvl() - linfo.min_lvl() + 1
    counts = jnp.bincount(lvl - linfo.min_lvl(), length=nlvl, weights=weights)
    counts = jnp.astype(counts, dtype) # After summing floating point is a reliable representation

    if ndev > 1:
        counts = jax.lax.psum(counts, axis_name=axis_name)
    
    levels = jnp.arange(nlvl, dtype=jnp.int32) + linfo.min_lvl()
    
    return levels, counts

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #


def _pos_zorder_sort_impl(x: jax.Array, block_size=64):
    # assert x.dtype == jnp.float32
    assert x.ndim == 2
    dim = x.shape[-1]
    assert dim in (2,3)

    idtype = same_width_int(x.dtype)

    outputs = (
        jax.ShapeDtypeStruct((x.shape[0],dim), x.dtype),
        jax.ShapeDtypeStruct((x.shape[0],), idtype),
        jax.ShapeDtypeStruct((x.shape[0] + np.maximum(1024, x.shape[0]//16), dim+1), idtype)
    )
    pos, ids = jax.ffi.ffi_call("PosZorderSort", outputs, vmap_method="sequential",
                                input_output_aliases={0:0})(
        x, block_size=np.uint64(block_size)
    )[0:2]

    return pos, ids

def pos_zorder_sort(x: jax.Array | Pos):
    """Brings 3d-positions into z-order

    If x is a pytree, it needs to have a "pos" attribute which will be used as the sorting key. 
    All remaining leaves of the pytree will be sorted accordingly along the leading axis (which 
    should have consistent length)
    """
    assert get_pos(x).ndim == 2, "positions must have shape (N,3)"

    @jax.custom_vjp
    def eval(x):
        if isinstance(x, jax.Array):
            return _pos_zorder_sort_impl(x)
        else: # assuming x is a pytree with x.pos attribute
            posz, idz = _pos_zorder_sort_impl(x.pos)
            def apply_sort(val):
                return val[idz]
            out = tree_map_by_len(apply_sort, x, len(posz))
            out.pos = posz # overwiting here allows jit to discard the unnecessary position gather

            return out, idz
    
    def eval_fwd(x):
        pos, idz = eval(x)
        return (pos, idz), idz
    
    def eval_bwd(idz, g):
        gxout, gids = g
        # Scatter the gradients back to the original ordering
        idinv = jnp.zeros_like(idz).at[idz].set(jnp.arange(len(idz), dtype=idz.dtype))
        
        return (tree_map_by_len(lambda x: x[idinv], gxout, len(idz)),)
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(x)
pos_zorder_sort.jit = jax.jit(pos_zorder_sort)

def search_sorted_z(xz, xz_query, block_size=64, leaf_search=False):
    """Finds the indices in xz where elements of xz_query would be inserted to keep order.
    This is similar to np.searchsorted, but works for 3D points sorted in Z-order.
    On equality maintains the rule: xz[idx] < v <= xz[idx+1]
    if leaf_search is True, it is assumed that xz contains one point per leaf and we 
    return the index of the leaf that the query point belongs to.
    """
    assert xz.dtype ==  xz_query.dtype
    assert xz.shape[-1] == xz_query.shape[-1] == 3

    out_type = jax.ShapeDtypeStruct((xz_query.shape[0],), jnp.int32)
    inds = jax.ffi.ffi_call("SearchSortedZ", (out_type,))(
        xz, xz_query, block_size=np.uint64(block_size), leaf_search=leaf_search)[0]
    return inds
search_sorted_z.jit = jax.jit(search_sorted_z, static_argnames=("block_size", "leaf_search"))

def weighted_percentile(x, weights, percentile):
    # Determine percentile of the particle-weighted leaf-size distribution:
    isort = jnp.argsort(x)
    ncum = jnp.cumsum(weights[isort])

    iperc = jnp.searchsorted(ncum, ncum[-1]*0.01*percentile, side="left")
    return x[isort[iperc]]

def find_regularization_level(lvl: jax.Array, linfo: LevelInfo, weights: jax.Array | None = None,
                              max_vol_fac: float = 50., min_percentile: float = 90.):
    """Determines the first level where the node volume is > max_vol_frac times the average vol."""
    levels, counts = level_histogram(lvl, linfo, weights=weights)

    # Find a reference level (to avoid volume undeflows/overlows in the relevant domain)
    # We determine the min_percentile for this and we only consider to regularize
    # above this level
    cts_csum = jnp.cumsum(counts)
    consider = cts_csum >= (0.01*min_percentile)*cts_csum[-1]
    ref_level = jnp.min(jnp.where(consider, levels, linfo.max_lvl()))

    vol_rel = jnp.astype(2., linfo.dtype)**(levels - ref_level)

    # find the level where the node volume exceeds the average volume of smaller nodes

    vol_sum = jnp.sum(jnp.where(levels <= ref_level, vol_rel*counts, 0))
    cts_sum = jnp.sum(jnp.where(levels <= ref_level, counts, 0))
    avg_vol = vol_sum / cts_sum

    too_large = consider & (vol_rel >= max_vol_fac * avg_vol)
    
    # find the first too large level
    first_too_large = jnp.where(jnp.any(too_large), jnp.argmax(too_large), len(levels-1))

    # jax.debug.log("avol: {} csum: {}", avg_vol[-10 - linfo.min_lvl()] * 2.**ref_level, cts_csum[-1])

    return levels[first_too_large]

def detect_leaf_boundaries(
        posz: jax.Array,
        ptype: jax.Array | None = None,
        num_types: int | None = None,
        leaf_size: int = 32,
        npart: int | None = None,
        lvl_bound = None,
        block_size: int = 64,
        alloc_size: int | None = None,
        cfg_reg: RegularizationConfig | None = None
    ) -> jax.Array:
    """Leaves are defined so that no single type has more than leaf_size particles in each leaf"""
    
    if alloc_size is None:
        alloc_size = int(div_ceil(len(posz), np.maximum(leaf_size//2, 1))) + 1
    if npart is None:
        npart = len(posz)
    if lvl_bound is None:
        lvl_max = LevelInfo(posz.shape[-1], posz.dtype).max_lvl()
        lvl_bound = (lvl_max, lvl_max)

    outputs = (
        jax.ShapeDtypeStruct((posz.shape[0]+1,), jnp.int8),
        jax.ShapeDtypeStruct((posz.shape[0]+1,), jnp.int32)
    )

    if ptype is None:
        ptype = jnp.zeros(0, dtype=jnp.uint8)
        num_types = 1
    else:
        assert num_types is not None, "Please provide num_types if providing ptype"
        assert len(ptype) == len(posz)
        ptype = jnp.astype(ptype, jnp.uint8)

    flag_split, lvl = jax.ffi.ffi_call("FlagLeafBoundaries", outputs, vmap_method="sequential")(
        posz, ptype, jnp.asarray(lvl_bound), npart, max_size=np.int32(leaf_size),
        block_size=np.uint64(block_size), scan_size=np.int32((leaf_size+1)*num_types),
        num_types=np.int32(num_types)
    )

    if cfg_reg is not None: # Ensure leaves to not exceed a dynamically determined maximum level
        num_pre = jnp.sum(flag_split)
        splits = jnp.where(flag_split, size=alloc_size, fill_value=npart)[0]
        leaf_npart = splits[1:] - splits[:-1]

        # Use upper extent level for this calculation
        leaf_lvl = jnp.minimum(lvl[splits[1:]], lvl[splits[:-1]]) - 1

        linfo = LevelInfo(dim=posz.shape[-1], dtype=posz.dtype)
        lvl_max = find_regularization_level(
            leaf_lvl, linfo, weights=leaf_npart,
            max_vol_fac=cfg_reg.max_volume_fac, min_percentile=cfg_reg.regularize_percentile
        )

        flag_split = flag_split | ((lvl >= lvl_max) & (jnp.arange(len(posz)+1) < npart+1))

        stats_callback(
            "allocation", AllocStats.record_leaf_regularization, lvl_max, num_pre, jnp.sum(flag_split)
        )

    nfilled = jnp.sum(flag_split)

    flag_split = flag_split + raise_if(nfilled > alloc_size,
        "Tree-Leaf allocation too small: filled={filled} size={size}.\n"
        "Hint: Increase alloc_fac_nodes or max_leaf_size",
        filled=nfilled, size=alloc_size, max_trace_depth=5
    )

    splits = jnp.where(flag_split, size=alloc_size, fill_value=npart)[0].astype(jnp.int32)

    return splits
detect_leaf_boundaries.jit = jax.jit(detect_leaf_boundaries,
    static_argnames=("leaf_size", "block_size", "cfg_reg")
)

def determine_znode_boundaries(posz: jax.Array, block_size: int = 64, nleaves: jnp.array = None) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Builds a Z-order tree from positions"""
    if nleaves is None:
        nleaves = jnp.array(len(posz), dtype=jnp.int32)
    nleaves = jnp.minimum(len(posz), nleaves)

    info = LevelInfo(posz.shape[-1], posz.dtype)
    
    # Set domain boundaries to behave like infinities
    # Note that this is fine for multi-GPU, because we ensure in advance that no
    # top-node intersects the boundary.
    pos_bound = jnp.full((2,posz.shape[-1]), jnp.nan, dtype=posz.dtype)

    out_types = (jax.ShapeDtypeStruct((posz.shape[0]+1,), jnp.int32),)*3
    lvl, lbound, rbound = jax.ffi.ffi_call("FindNodeBoundaries", out_types)(
        posz, pos_bound, nleaves, block_size=np.uint64(block_size),
        lvl_max = np.int32(info.max_lvl()), lvl_invalid = np.int32(-2000)
    )

    return lvl, lbound, rbound
determine_znode_boundaries.jit = jax.jit(determine_znode_boundaries)

def get_node_geometry(posz: jax.Array, lbound: jax.Array, rbound: jax.Array, 
                      num: jnp.array = None, block_size: int = 64, result: str = "lvl_cent_ext",
                      upper_extent: bool = False
                      ) -> Tuple[jax.Array, jax.Array, jax.Array]:
    assert lbound.shape == rbound.shape
    assert lbound.dtype == rbound.dtype == jnp.int32

    mode_flags = 0

    if "lvl" in result:
        mode_flags += 1
    if "cent" in result:
        mode_flags += 2
    if "ext" in result:
        mode_flags += 4

    if num is None:
        num = jnp.array(len(lbound))

    def rlen(flag):
        return lbound.shape[0] if flag else 1

    out_types = (
        jax.ShapeDtypeStruct((rlen("lvl" in result),), jnp.int32),
        jax.ShapeDtypeStruct((rlen("cent" in result), posz.shape[-1]), posz.dtype),
        jax.ShapeDtypeStruct((rlen("ext" in result), posz.shape[-1]), posz.dtype)
    )

    rdict = {}
    
    rdict["lvl"], rdict["cent"], rdict["ext"] = jax.ffi.ffi_call("GetNodeGeometry", out_types)(
        posz, lbound, rbound, num, block_size=np.uint64(block_size),
        lvl_invalid=np.int32(-2000), mode_flags=np.uint32(mode_flags), upper_extent=upper_extent
    )

    res = []
    for key in result.split("_"):
        res.append(rdict[key])

    if len(res) == 1:
        return res[0]
    else:
        return tuple(res)
get_node_geometry.jit = jax.jit(get_node_geometry)

def distr_boundary_extend(posz, npart=None, block_size: int = 64):
    rank, ndev, axis_name = get_rank_info()

    if npart is None:
        npart = jnp.sum(~jnp.isnan(posz[...,0]))

    info = LevelInfo(dim=posz.shape[-1], dtype=posz.dtype)

    nlevels = info.max_lvl() - info.min_lvl() + 1
    out_types = (jax.ShapeDtypeStruct((nlevels,), jnp.int32),)

    irange = jnp.array([0, npart])

    # Distance from the left boundary where each levels node ends
    xleft = send_to_right(posz[npart-1], axis_name, invalid_float=-jnp.inf)
    ext_lr = jax.ffi.ffi_call("GetBoundaryExtendPerLevel", out_types)(
        xleft, irange, posz, block_size=np.uint64(block_size), left=True,
        lvl_min=np.int32(info.min_lvl()), lvl_max=np.int32(info.max_lvl())
    )[0]

    # Distance from the right boundary where each levels node starts
    xright = send_to_left(posz[0], axis_name, invalid_float=jnp.inf)
    ext_rl = jax.ffi.ffi_call("GetBoundaryExtendPerLevel", out_types)(
        xright, irange, posz, block_size=np.uint64(block_size), left=False,
        lvl_min=np.int32(info.min_lvl()), lvl_max=np.int32(info.max_lvl())
    )[0]

    ext_rr = send_to_left(ext_lr, axis_name, invalid_int=0)
    ext_ll =  send_to_right(ext_rl-npart, axis_name, invalid_int=0)

    return ext_ll, ext_lr, ext_rl, ext_rr

def zsort_and_tree(
        part: Pos,
        cfg_tree: TreeConfig = TreeConfig(),
        data: Any | None = None,
        ptype: jax.Array | None = None,
        num_types: int | None = None,
        shrink: bool = True
    ) -> Tuple[Pos, TreeHierarchy]:
    rank, ndev, axis_name = get_rank_info()

    npart_tot = get_num_total(part, default_to_length=(ndev==1))
    dt = (data, ptype)

    if ndev > 1:
        partz, dtz = distr_zsort(part, data=dt, nsamp=cfg_tree.nsamp)
    else:
        partz, isort = pos_zorder_sort(part)
        dtz = tree_map_by_len(lambda x: x[isort], dt, len(get_pos(partz)))

    if ndev > 1:
        top_node_size = define_tree_level_node_sizes(npart_tot, cfg_tree)[-1]
        partz, dtz, lvl_bound = adjust_domain_for_nodesize(partz, top_node_size, dataz=dtz)
    else:
        lvl_bound = None

    if ptype is None:
        th = build_tree_hierarchy(partz, cfg_tree, lvl_bound=lvl_bound)
    else:
        th = build_tree_hierarchy(
            partz, cfg_tree, lvl_bound=lvl_bound, ptype=dtz[1], num_types=num_types
        )

    out_part = [partz]
    if (data is not None) or not shrink:
        out_part.append(dtz[0])
    if (ptype is not None) or not shrink:
        out_part.append(dtz[1])
    
    return (*out_part, th)
zsort_and_tree.jit = jax.jit(zsort_and_tree, static_argnames="cfg_tree")
zsort_and_tree.smap = shard_map_constructor(
    zsort_and_tree, in_specs=(P(-1), None, P(-1), P(-1), None, None), static_argnames="cfg_tree"
)

def zsort_and_tree_multi_type(
        part: Tuple[Pos, ...],
        cfg_tree: TreeConfig = TreeConfig(),
        data: Any | None = None
    ) -> Tuple[Tuple[Pos, ...], TreeHierarchy]:
    num_types = len(part)
    size_of_type = tuple(len(get_pos(p)) for p in part)
    num_of_type = jnp.array(tuple(get_num(p) for p in part))
    
    if data is not None:
        part_stacked, data_stacked = concatenate_pytrees(tuple(zip(part, data)), num_of_type)
    else:
        part_stacked = concatenate_pytrees(part, num_of_type)
        data_stacked = None
    
    if hasattr(part_stacked, "num"):
        part_stacked.num = jnp.sum(num_of_type)
    if hasattr(part_stacked, "num_total"):
        part_stacked.num_total = sum(get_num_total(p, default_to_length=True) for p in part)

    ptype = inverse_of_splits(cumsum_starting_with_zero(num_of_type), len(get_pos(part_stacked)))

    partz_stacked, dataz_stacked, ptypez, th = zsort_and_tree(
        part_stacked, cfg_tree=cfg_tree, data=data_stacked, ptype=ptype, num_types=num_types,
        shrink=False
    )

    part_data_z, num_of_type = separate_pytrees(
        (partz_stacked, dataz_stacked), ptypez, out_sizes=size_of_type, num=get_num(partz_stacked)
    )
    partz, dataz = tuple(zip(*part_data_z)) # transpose
    if hasattr(partz_stacked, "num"):
        partz = tuple(replace(partz[t], num=num_of_type[t]) for t in range(num_types))

    if data is not None:
        return partz, dataz, th
    else:
        return partz, th
zsort_and_tree_multi_type.jit = jax.jit(zsort_and_tree_multi_type, static_argnames="cfg_tree")
zsort_and_tree_multi_type.smap = shard_map_constructor(
    zsort_and_tree_multi_type, in_specs=(P(-1), None, P(-1)), static_argnames="cfg_tree"
)

def center_of_mass(ispl: jax.Array, part: PosMass, kahan_summation: bool = True, block_size=32
                   ) -> PosMass:
    """Computes the center of mass of the nodes in the tree plane"""
    assert part.pos.dtype == jnp.float32
    assert ispl.dtype == jnp.int32
    assert part.pos.ndim == 2

    out_xcent = jax.ShapeDtypeStruct((ispl.size-1, 4), part.pos.dtype)

    mass = jnp.broadcast_to(part.mass, part.pos.shape[:-1])

    xm = jax.ffi.ffi_call("CenterOfMass", (out_xcent,))(
        ispl, part.pos, mass,
        kahan = kahan_summation,
        block_size=np.uint64(block_size)
    )[0]

    xm = pcast_like(xm, like=part.pos)

    return PosMass(pos=xm[...,0:3], mass=xm[...,4])
center_of_mass.jit = jax.jit(center_of_mass, static_argnames=['kahan_summation', 'block_size'])


# ------------------------------------------------------------------------------------------------ #
#                                       Domain Decomposition                                       #
# ------------------------------------------------------------------------------------------------ #


def determine_npart(x):
    """Determines the number of valid particles (that are not nan)"""
    valid = ~jnp.isnan(get_pos(x))
    return jnp.sum(valid[...,0] & valid[...,1] & valid[...,2])

    
def distr_zsort(part: Pos, data: Any | None = None, nsamp: int = 1024, equalize: bool = True):
    rank, ndev, axis_name = get_rank_info()

    # if ndev == 1:
    #     return pos_zorder_sort(part)[0]

    pos = get_pos(part)
    numtot = get_num_total(part)
    if numtot >= ndev*len(pos):
        raise ValueError(f"Particles ({numtot}/{ndev*len(pos)}) appear to be not padded... Please"
                         " use jztree.data.pad_particles to leave some space for communication!")
    
    # Sample based domain decomposition
    key = jax.random.key(0)
    isamp = jax.random.randint(key, shape=(nsamp,), minval=0, maxval=get_num(part))
    posall = jax.lax.all_gather(pos[isamp], axis_name=axis_name, tiled=True)
    xpivot = pos_zorder_sort(posall)[0][nsamp::nsamp]
    xpivot = jnp.pad(xpivot, ((1,1), (0,0)), constant_values=jnp.inf).at[0].set(-jnp.inf)

    # Now organize and determine which chunks need to be send to each rank
    irank = search_sorted_z(xpivot, part.pos)-1
    (part, data), dev_spl = all_to_all_with_irank(
        irank, (part, data), num=part.num, err_hint="\nHint: Increase padding."
    )
    
    part.num = dev_spl[-1]

    partz, idz = pos_zorder_sort(part)
    dataz = tree_map_by_len(lambda x: x[idz], data, len(get_pos(partz)))

    stats_callback("allocation", AllocStats.record_filled_sort, dev_spl[-1], len(part.pos))

    if equalize:
        # We have posz globally and locally in z-order now
        # Let's do another communication step to improve the balance
        with jax.enable_x64():
            spl_have = global_splits(jnp.astype(partz.num, jnp.int64), axis_name=axis_name)
            spl_target = (jnp.arange(0, ndev+1, dtype=jnp.int64) * (numtot // ndev)).at[-1].set(numtot)
            spl_send = jnp.clip(spl_target - spl_have[rank], 0, partz.num).astype(jnp.int32)

        (partz, dataz), dev_spl = all_to_all_with_splits(
            (partz, dataz), spl_send, err_hint="\nHint: Increase padding of positions",
        )
        partz.num = dev_spl[-1]

    return partz, dataz
distr_zsort.smap = shard_map_constructor(
    distr_zsort, in_specs=(P(-1), P(-1), None, None), static_argnames=("nsamp", "equalize")
)

def adjust_domain_for_nodesize(partz: Pos, max_node_size: int, dataz: Any | None = None):
    """Shifts particles so that nodes with size <= max_node_size always lie on a single GPU"""
    rank, ndev, axis_name = get_rank_info()

    npart, ntot = get_num(partz), get_num_total(partz)
    
    if max_node_size >= ntot*0.5/ndev:
        raise ValueError(f"Topnodes are dangerously large: size={max_node_size} npart={ntot/ndev}\n"
                          "Hint: increase stop_coarsen for more/smaller topnodes.")
    assert ntot/ndev <= 2**30, "Might get integer overflows..."

    ext_ll, ext_lr, ext_rl, ext_rr = distr_boundary_extend(get_pos(partz), npart=npart)
    npart_l = ext_lr - ext_ll
    
    ilvl_max = jnp.max(jnp.where(npart_l <= max_node_size, jnp.arange(len(npart_l)), -1))

    npshift = ext_lr[ilvl_max]

    (partz, dataz), npart = shift_particles_left(
        (partz, dataz), npshift, max_send=max_node_size, npart=npart
    )
    partz.num = npart

    stats_callback("allocation", AllocStats.record_filled_domain, partz.num, len(partz.pos))

    # Find the level of the new boundary
    posz = get_pos(partz)

    xr = send_to_left(posz[0], axis_name)
    xl = send_to_right(posz[npart-1], axis_name)

    lvl_bound = get_node_geometry(
        jnp.array([xl, posz[0], posz[npart-1], xr]), 
        lbound=jnp.array([0,2], dtype=jnp.int32), rbound=jnp.array([2,4], dtype=jnp.int32), num=2,
        result="lvl"
    )

    return partz, dataz, lvl_bound
adjust_domain_for_nodesize.jit = jax.jit(adjust_domain_for_nodesize, static_argnames="max_node_size")

# ------------------------------------------------------------------------------------------------ #
#                                      Tree Building Functions                                     #
# ------------------------------------------------------------------------------------------------ #

def estimate_node_number(npart, max_node_size, alloc_fac=1.):
    """Gives an estimate of the number of nodes
    
    Assumes that particles are grouped in z-order into nodes that have <= node_size particles each.
    """
    # Since most nodes with size <= max_node_size/2 can be grouped, we generally get a reasonably 
    # safe estimate by dividing node_size/2. However, the estimate is not totally guaranteed, since
    # imbalanced distributions may require some nodes with size >= max_node_size/2 may block
    # multiple nodes with size <= max_node_size/2 from being summarized.
    return int(div_ceil(npart*alloc_fac, np.maximum(max_node_size//2, 1))) + 1

def define_tree_level_node_sizes(npart: int, cfg_tree: TreeConfig):
    max_num_leaves = estimate_node_number(npart, cfg_tree.max_leaf_size)

    nlevels = np.log(max_num_leaves / cfg_tree.stop_coarsen) / np.log(cfg_tree.coarse_fac)
    nlevels = np.maximum(int(np.ceil(nlevels)), 1)

    node_sizes = [int(cfg_tree.max_leaf_size * (cfg_tree.coarse_fac ** i)) for i in range(0, nlevels)]

    return node_sizes

def splits_per_type(spl, ptype, num_types, npart):
    spls = []
    for t in range(num_types):
        prefix = cumsum_starting_with_zero((ptype == t) & (jnp.arange(len(ptype)) < npart))
        spls.append(prefix[spl])
    return jnp.stack(spls, axis=0)

def define_split_hierarchy(
        posz: jax.Array,
        node_sizes: Tuple[int],
        alloc_size: int,
        npart: int | None = None,
        ptype: jax.Array | None = None,
        num_types: int | None = None,
        lvl_bound = None,
        cfg_reg: RegularizationConfig | None = None
    ) -> Tuple[jax.Array, PackedArray, PackedArray]:
    """Finds the splitting point of the tree hierarchy

    returns
      ispl: the splitting points of leafs in the particle array
      ispl_n2n: a PackedArray that defines the node to node splitting points per level
      ispl_n2l: a PackedArray that defines the node to leaf splitting points per level
    """
    nlevels = len(node_sizes)

    if lvl_bound is None:
        lvl_max = LevelInfo(posz.shape[-1], posz.dtype).max_lvl()
        lvl_bound = (lvl_max, lvl_max)
    
    ispl = detect_leaf_boundaries(
        posz, ptype=ptype, num_types=num_types,
        leaf_size=node_sizes[0], npart=npart, alloc_size=alloc_size, lvl_bound=lvl_bound,
        cfg_reg=cfg_reg
    )

    nleaves = jnp.argmax(ispl)
    lvl, lbound, rbound = determine_znode_boundaries(posz[ispl[:-1]], nleaves=nleaves)

    if ptype is not None:
        # If we have multiple types, we require that the number per type does not exceed a maximum
        ispl_per_type = splits_per_type(ispl, ptype, num_types, npart)
        npart_if_not_split = jnp.max(ispl_per_type[:,rbound] - ispl_per_type[:,lbound], axis=0)
    else:
        npart_if_not_split = ispl[rbound] - ispl[lbound]
        ispl_per_type = ispl.reshape((1,) + ispl.shape)

    # At each level of the hierarchy, the active nodes are determined by the node sizes
    is_spl_on_level = npart_if_not_split[None,:] > jnp.array(node_sizes, dtype=jnp.int32)[:,None]
    # Additionally we need to activate domain boundary nodes that lie above the boudary level
    is_spl_on_level = is_spl_on_level | ((lbound <= 0) & (lvl > lvl_bound[0]))
    is_spl_on_level = is_spl_on_level | ((rbound >= nleaves) & (lvl > lvl_bound[1]))

    if cfg_reg is not None: # optionally regularize nodes by a maximum extent per level
        # leaves were already regularized, so let's keep them all:
        new_is_spl = [(npart_if_not_split > 0) | is_spl_on_level[0]] 

        lvl_up = jnp.minimum(lvl[lbound], lvl[rbound]) - 1 # use upper level in this calculation

        for i in range(1, nlevels):
            is_node = (~is_spl_on_level[i]) & is_spl_on_level[i,lbound] & is_spl_on_level[i,rbound]
            is_node = is_node & (jnp.arange(len(lvl)) < nleaves)

            linfo = LevelInfo(dim=posz.shape[-1], dtype=posz.dtype)
            lvl_max = find_regularization_level(
                lvl_up, linfo, weights=jnp.where(is_node, npart_if_not_split, 0),
                max_vol_fac=cfg_reg.max_volume_fac, min_percentile=cfg_reg.regularize_percentile
            )

            new = is_spl_on_level[i] | ((lvl > lvl_max) &  (jnp.arange(len(lvl)) <= nleaves))

            # rank = jax.lax.axis_index("gpus")
            # jax.debug.log("r {} lv {}, lvmax {} isnode {}", rank, i, lvl_max, jnp.mean(is_node))

            stats_callback(
                "allocation", AllocStats.record_node_regularization, 
                jnp.sum(is_spl_on_level[i]), jnp.sum(new)
            )

            new_is_spl.append(new)
        
        is_spl_on_level = jnp.array(new_is_spl)

    # Calculate a prefix accross hierarchy levels to densly stack the nodes later
    offsets = jnp.cumsum(is_spl_on_level.flatten()).reshape(is_spl_on_level.shape)
    level_spl = jnp.pad(offsets[:,-1], (1,0), constant_values=np.int32(0)) # save level start/end points
    nnodes_on_level = level_spl[1:] - level_spl[:-1] - 1 # -1 since splits are always 1 larger than nodes

    # Check that the allocation is big enough
    level_spl = level_spl + raise_if(level_spl[-1] > alloc_size,
        "Tree-node allocation too small: filled={filled} size={size}.\n"
        "Hint: Increase alloc_fac_nodes, max_leaf_size or coarse_fac",
        filled=level_spl[-1], size=alloc_size
    )

    stats_callback("allocation", AllocStats.record_filled_nodes, level_spl[-1], alloc_size)

    # correct offsets to exclude the element at hand and invalidate inactive elements:
    offsets = jnp.where(is_spl_on_level, offsets - is_spl_on_level, alloc_size).astype(jnp.int32)
    
    # node-to-leaf relation is given by the leaf that is active at the node location
    ispl_n2l = jnp.zeros(alloc_size, dtype=jnp.int32).at[offsets].set(offsets[0:1,:])
    ispl_n2l = PackedArray.from_data(ispl_n2l, level_spl, fill_values=nnodes_on_level[0])

    # node-to-node relation is given by the last level node that is active at the node location
    # for the leaf-level we insert the leaf to particle relation here
    ilevel = jnp.arange(nlevels)
    value = jnp.where(ilevel[:,None] == 0, ispl, offsets[ilevel-1,:] - level_spl[ilevel-1,None])
    ispl_n2n = jnp.zeros(alloc_size, dtype=jnp.int32).at[offsets].set(value.astype(jnp.int32))
    # out of bounds access shall give nnodes of next smaller level (or npart for leaves):
    fill_val = jnp.pad(nnodes_on_level[:-1], (1,0), constant_values=ispl[-1])
    ispl_n2n = PackedArray.from_data(ispl_n2n, level_spl, fill_values=fill_val)
    
    return ispl, ispl_per_type, ispl_n2l, ispl_n2n

def get_tree_mass_centers(part: PosMass, ispl_n2n: PackedArray) -> Tuple[PackedArray, PackedArray]:
    prop_array_spl = cumsum_starting_with_zero(ispl_n2n.ispl[1:] - ispl_n2n.ispl[:-1] - 1)

    def handle_mcent_level(i, carry):
        node_mcent, node_mass, posm = carry
        posm = center_of_mass(ispl_n2n.get(i, size=len(posm.mass)+1), posm)
        node_mass = node_mass.set(i, posm.mass, num=ispl_n2n.num(i)-1)
        node_mcent = node_mcent.set(i, posm.pos, num=ispl_n2n.num(i)-1)
        return node_mcent, node_mass, posm
    
    posm = center_of_mass(ispl_n2n.get(0), part)
    npos = PackedArray.from_data(posm.pos, ispl=prop_array_spl, fill_values=jnp.nan)
    node_mass = PackedArray.from_data(posm.mass, ispl=prop_array_spl, fill_values=jnp.nan)

    node_mcent, node_mass, _ = jax.lax.fori_loop(
        1, ispl_n2n.nlevels(), handle_mcent_level, (npos, node_mass, posm)
    )

    return node_mcent, node_mass

def build_tree_hierarchy(
        part: PosMass | jax.Array,
        cfg_tree: TreeConfig,
        lvl_bound: jax.Array | None = None,
        ptype: jax.Array | None = None,
        num_types: jax.Array | None = None
    ) -> TreeHierarchy:
    """Builds a tree hierarchy from z-order positions

    The zeroth level of the tree corresponds to leaves, which contain multiple particles.
    Nodes (and leaves) are selected so that they are as big as possible while not containing more
    than a maximum number of particles that starts at cfg_tree.max_leaf_size and increases per level
    by a factor cfg_tree.coarse_fac.
    
    Nodes are parameterized through a set of splits. For example the particles that lie in the leaf
    with index i are given py part[ispl_n2n.get(level=0)[i]: ispl_n2n.get(level=0)[i+1]]
    The ith node of level n contain all level n-1 nodes in the range :
    ispl_n2n.get(n)[i]: ispl_n2n.get(n)[i+1]

    In jax memory size needs to be known at compile time, but the required number of nodes is 
    data dependent on each level. To limit the number of allocations that we need to predict, we
    use the PackedArray class, that helps us to stack multiple different levels into a single
    continguous array, but to access it "almost" as if they were separate arrays.
    """
    rank, ndev, axis_name = get_rank_info()

    posz = get_pos(part)
    npart_tot = get_num_total(part, default_to_length=(ndev==1))
    np_per_dev = npart_tot // ndev # static estimate of number of unpadded-particles
    npart = get_num(part, default_to_nancount=True)

    node_sizes = define_tree_level_node_sizes(npart_tot, cfg_tree)

    alloc_size = estimate_node_number(np_per_dev, cfg_tree.max_leaf_size, cfg_tree.alloc_fac_nodes)

    ispl_l2p, ispl_per_type, ispl_n2l, ispl_n2n = define_split_hierarchy(
        posz, node_sizes, alloc_size, npart=npart,
        ptype=ptype, num_types=num_types,
        lvl_bound=lvl_bound,
        cfg_reg=cfg_tree.regularization
    )

    # We can handle all levels at once for node geometry:
    ispl_n2p = ispl_l2p[ispl_n2l.data] # node to particle relation
    lvl, geom_cent = get_node_geometry(
        posz, ispl_n2p[:-1], ispl_n2p[1:], num=ispl_n2l.nfilled()-1, result="lvl_cent"
    )
    # However, the splits are discontinuous at level boundaries. We have to delete the extra entries
    lvl = jnp.delete(lvl, ispl_n2l.ispl[1:-1]-1, assume_unique_indices=True)
    geom_cent = jnp.delete(geom_cent, ispl_n2l.ispl[1:-1]-1, axis=0, assume_unique_indices=True)

    # node property arrays are on each level one element smaller than the splitting point arrays
    prop_array_spl = cumsum_starting_with_zero(ispl_n2n.ispl[1:] - ispl_n2n.ispl[:-1] - 1)
    lvl = PackedArray.from_data(lvl, prop_array_spl, fill_values=-2000)
    geom_cent = PackedArray.from_data(geom_cent, prop_array_spl, fill_values=jnp.nan)

    if cfg_tree.mass_centered:
        assert hasattr(part, "mass"), "To use mass centering, please provide PosMass input"
        nmass_cent, nmass = get_tree_mass_centers(part, ispl_n2n)
    else:
        nmass_cent, nmass = None, None

    size_leaves = estimate_node_number(np_per_dev, cfg_tree.max_leaf_size)
    th = TreeHierarchy(
        size_leaves, ispl_n2n, ispl_n2l, ispl_per_type,
        lvl, geom_cent,
        mass = nmass, mass_cent = nmass_cent
    )
    
    return th
build_tree_hierarchy.jit = jax.jit(build_tree_hierarchy, static_argnames=['cfg_tree'])

# ------------------------------------------------------------------------------------------------ #
#                                     Interaction List Helpers                                     #
# ------------------------------------------------------------------------------------------------ #

def dense_interaction_list(nnodes: jax.Array, size_nodes: int, size_ilist: int,
                           node_range: jax.Array | None = None) -> InteractionList:
    """A dense interaction list where all nodes interact with all other nodes.

    size_ilist: size of the interaction list. (Required at compile time)
    nnodes: actual number of filled nodes (Can be dynamic, used to invalidating unused nodes)
    """

    dtype = nnodes.dtype

    idx = jnp.arange(size_ilist)
    if node_range is not None:
        # !!! Put some checks here!
        nint = nnodes*(node_range[1] - node_range[0])
        ilist = jnp.where(idx < nint, idx % nnodes, 0)
        node_idx = jnp.arange(size_nodes)
        ispl = cumsum_starting_with_zero((node_idx >= node_range[0]) & (node_idx < node_range[1])) * nnodes
    else:
        nint = nnodes*nnodes
        ilist = jnp.where(idx < nint, idx % nnodes, 0)
        ispl = jnp.minimum(jnp.arange(0, size_nodes+1, dtype=dtype) * nnodes, nint)

    ispl = ispl + raise_if((nnodes > size_nodes) | (nint > size_ilist),
        "Cannot fit {nnodes}/{size_nodes}, {nint}/{size_ilist}\n"
        "Hint: Increase alloc_fac_ilist",
        nnodes=nnodes, size_nodes=size_nodes, nint=nint, size_ilist=size_ilist
    )

    stats_callback("allocation", AllocStats.record_filled_interactions, nint, size_ilist)
    
    return verify_ilist(InteractionList(ispl=ispl, iother=ilist))
dense_interaction_list.jit = jax.jit(dense_interaction_list, static_argnames=['size_ilist', 'size_nodes'])

def grouped_dense_interaction_list(nnodes: jax.Array | int, size_ilist: int,
                                   ngroup: int = 32, size_super: int | None = None,
                                   node_range: jax.Array | None = None,
                                   dtype=jnp.int32
                                   ) -> Tuple[jax.Array, InteractionList, jax.Array]:
    """Defines an all-to-all interaction list over super-nodes and a super-node to node relation

    This is useful for evaluating all-to-all interactions in a grouped manner on GPU

    node_range: if specified, the interaction list will only contain interactions with
                receiving indices in node_range will be evaluated
    """
    nnodes, ngroup = jnp.astype(nnodes, dtype), jnp.astype(ngroup, dtype)

    if size_super is None: # if not provided, guarantee a sufficient allocation
        size_super = np.ceil(np.sqrt(size_ilist)).astype(np.int64) + 2
    
    # define the super node to node relation
    if node_range is None:
        nsuper_nodes = div_ceil(nnodes, ngroup)
        nint = nsuper_nodes*nsuper_nodes
        spl_super = jnp.minimum(jnp.arange(size_super+1,dtype=dtype) * ngroup, nnodes)
        ispl = jnp.minimum(jnp.arange(size_super+1,dtype=dtype) * nsuper_nodes, nint)
    else:
        node_range = jnp.astype(node_range, dtype)
        super_range = node_range // ngroup
        # In this case we only evaluate receiving nodes that lie inside of the node_range
        # further, we have to make sure that spl_super splits at our indices
        nsuper_nodes = div_ceil(nnodes, ngroup) + 2
        spl_super = jnp.minimum(jnp.arange(size_super+1,dtype=dtype) * ngroup, nnodes)
        spl_super = jnp.insert(spl_super, super_range + 1, node_range)

        valid = (spl_super[:-1] >= node_range[0]) & (spl_super[1:] <= node_range[1])
        valid = valid & (spl_super[1:] > spl_super[:-1]) # may have some 0 nodes due to way we inserted
        ispl = cumsum_starting_with_zero(jnp.where(valid, nsuper_nodes, 0).astype(dtype))
        nint = ispl[-1]

    nsuper_nodes = nsuper_nodes + raise_if(nint > size_ilist,
        "Cannot fit {n} interactions into interaction list with size {size}\n"
        "Hint: Increase alloc_fac_ilist",
        n=nint, size=size_ilist
    )
    
    idx = jnp.arange(size_ilist, dtype=dtype)
    ilist = jnp.where(idx < nint, idx % nsuper_nodes, 0)

    ilist = verify_ilist(InteractionList(ispl=ispl, iother=ilist))

    stats_callback("allocation", AllocStats.record_filled_interactions, nint, size_ilist)

    return spl_super, ilist, nsuper_nodes
grouped_dense_interaction_list.jit = jax.jit(
    grouped_dense_interaction_list, static_argnames=["size_ilist", "size_super"]
)

def _linearly_grouped(num, size, ngroup=32):
    num_sup = div_ceil(num, ngroup)
    return jnp.minimum(jnp.arange(size+1) * ngroup, num), num_sup

def distr_grouped_dense_interaction_list(
        num_local: int, size: int, size_ilist: int, only_geq: bool = False
        ) -> Tuple[jax.Array, InteractionList, jax.Array]:
    rank, ndev, axis_name = get_rank_info()

    spl, nsuper = _linearly_grouped(num_local, size, ngroup=32)

    nper_rank = jax.lax.all_gather(nsuper, axis_name)

    if only_geq:
        dev_spl = cumsum_starting_with_zero(nper_rank * (jnp.arange(ndev) >= rank))
    else:
        dev_spl = cumsum_starting_with_zero(nper_rank)
    
    # Define a dense interaction list on top-nodes:
    ilist = dense_interaction_list(dev_spl[-1], size, size_ilist,
        node_range=jnp.array([dev_spl[rank], dev_spl[rank+1]])
    )
    ilist.ids = jnp.arange(size) - dev_spl[inverse_of_splits(dev_spl, size)] # !!! verify size
    ilist.dev_spl = dev_spl

    return spl, ilist, nsuper

def masked_scatter(mask, arr, indices, values):
    indices = jnp.where(mask, indices, len(arr))
    return arr.at[indices].set(values)

def simplify_interaction_list(ilist: InteractionList, always_keep: jax.Array | None = None
                              ) -> InteractionList:
    """Get reduced version of the interaction and node list skipping nodes without interactions
    
    Useful in multi-GPU scenarios where many non-local nodes will not have any local interactions
    """
    size_nodes = ilist.ispl.size - 1
    idx = jnp.arange(ilist.iother.size)

    # flag all nodes that appear at least in one interaction
    ioth = jnp.where(idx < ilist.ispl[-1], ilist.iother, size_nodes)
    flag = ilist.ispl[1:] > ilist.ispl[:-1] # appears as receiver
    flag = flag.at[ioth].set(True) # appears as source

    if always_keep is not None:
        flag = flag | always_keep
    
    # create reduced id list
    prefix = cumsum_starting_with_zero(flag)

    reduced_ids = masked_scatter(flag, jnp.zeros_like(ilist.ids), prefix[:-1], ilist.ids)
    reduced_dev_spl = prefix[ilist.dev_spl]

    # change the label and the offsets of the interaction list
    ispl = jnp.full(ilist.ispl.shape, ilist.ispl[-1], ilist.ispl.dtype).at[prefix].set(ilist.ispl)

    ilist = InteractionList(
        ispl, prefix[ilist.iother], rad2=ilist.rad2, ids=reduced_ids, dev_spl=reduced_dev_spl
    )
    
    return verify_ilist(ilist)

# ------------------------------------------------------------------------------------------------ #
#                                       Tree Walking Helpers                                       #
# ------------------------------------------------------------------------------------------------ #

from dataclasses import dataclass
from typing import Callable

@dataclass
class TreeWalkFunctions:
    top_node_out: Callable
    child_input: Callable
    evaluate_n2n: Callable

from typing import Callable
def dual_tree_walk(
    f: TreeWalkFunctions,
    th: TreeHierarchy,
    size_ilist: int
):
    nplanes = th.num_planes()

    size = th.size()

    spl, ilist, nsup = grouped_dense_interaction_list(
        th.num(nplanes-1), size_ilist=size_ilist, ngroup=32, size_super=size
    )

    node_data = f.top_node_out(num=nsup, size=len(spl)-1)

    # Add an extra-level to splits for super-nodes
    spl_n2n = th.ispl_n2n.resize_levels(nplanes+1).set(nplanes, spl, nsup+1, fill_value=spl[-1])

    def handle_level(i, carry):
        level = nplanes - 1 - i
        node_data, ilist = carry

        spl = spl_n2n.get(level+1, size=size+1)

        child_data = f.child_input(level=level, node_data=node_data, spl=spl)

        child_res, child_ilist = f.evaluate_n2n(
            level=level, ilist=ilist, node_data=node_data, spl=spl, child_data=child_data
        )
        
        return child_res, child_ilist

    for i in range(nplanes):
        node_data, ilist = handle_level(i, (node_data, ilist))
    
    return node_data, ilist
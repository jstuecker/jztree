import numpy as np
import jax
import jax.numpy as jnp
from jztree_cuda import ffi_fof
from .tree import pos_zorder_sort, search_sorted_z
from .common import conditional_callback, masked_prefix_sum, cumsum_starting_with_zero, inverse_indices
from .data import  FofConfig, FofData

import fmdj

jax.ffi.register_ffi_target("NodeFofAndIlist", ffi_fof.NodeFofAndIlist(), platform="CUDA")
jax.ffi.register_ffi_target("ParticleFof", ffi_fof.ParticleFof(), platform="CUDA")

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

def node_fof_and_ilist(
        node_igroup, node_ilist_splits, node_ilist, isplit, xleaf, lvl_leaf,
        # npart_leaf, isplit, node_ilist, node_ir2list, node_ilist_splits, 
        rlink, boxsize=0., alloc_fac=128, block_size=32
    ):
    assert node_ilist_splits.shape[0] == isplit.shape[0], "Should both correspond to no. of nodes+1"
    assert isplit.shape[0] == node_igroup.shape[0]+1, "Should both correspond to no. of nodes"
    
    x4leaf = jnp.concatenate((xleaf, lvl_leaf.view(jnp.float32)[...,None]), axis=-1)

    leaf_igroup = jax.ShapeDtypeStruct((len(xleaf),), jnp.int32)
    leaf_ilist_ispl = jax.ShapeDtypeStruct((len(xleaf)+1,), jnp.int32)
    leaf_ilist = jax.ShapeDtypeStruct((int(alloc_fac * len(xleaf)),), jnp.int32)
    
    outputs = (leaf_igroup, leaf_ilist_ispl, leaf_ilist)

    res = jax.ffi.ffi_call("NodeFofAndIlist", outputs)(
        node_igroup, node_ilist_splits, node_ilist, isplit, x4leaf,
        r2link=np.float32(rlink*rlink), boxsize=np.float32(boxsize), block_size=np.int32(block_size)
    )

    leaf_igroup, leaf_ilist_ispl, leaf_ilist = res

    def err(n1, n2):
        raise MemoryError(f"The interaction list allocation is too small. (need: {n1} have: {n2})" +
                          f"increase alloc_fac at least by a factor of {n1/n2:.1f}")
    n1, n2 = leaf_ilist_ispl[-1], leaf_ilist.shape[0]
    leaf_igroup = leaf_igroup + conditional_callback(n1 > n2, err, n1, n2)

    return leaf_igroup, leaf_ilist_ispl, leaf_ilist
node_fof_and_ilist.jit = jax.jit(
    node_fof_and_ilist, static_argnames=["boxsize", "rlink", "alloc_fac", "block_size"]
)

def contract_links(igroup):
    def body(carry):
        igroup, _ = carry
        igroup_new = igroup[abs(igroup)]
        jax.debug.print("diff: {}", jnp.sum(igroup_new != igroup))
        return igroup_new, jnp.any(abs(igroup_new) != abs(igroup))
    
    igroup_new = jax.lax.while_loop(lambda carry: carry[1], body, (igroup, True))[0]

    return igroup_new

from .knn import dense_ilist

def node_node_fof(th: fmdj.data.TreeHierarchy, rlink: float, boxsize: float=0., alloc_fac_ilist: int = 128):
    nplanes = th.num_planes()
    nnodes = th.lvl.num(nplanes-1)

    # initialize interaction list
    valid = jnp.arange(th.plane_sizes[-1], dtype=jnp.int32) < nnodes
    spl, il, _, ispl = dense_ilist(th.plane_sizes[-1], valid, ngroup=32)

    # Get coarsest plane data
    igroup = jnp.arange(len(spl)-1, dtype=jnp.int32)

    for level in reversed(range(nplanes)):
        size = th.plane_sizes[level]
        xleaf = th.geom_cent.get(level, size)
        lvl_leaf = th.lvl.get(level, size)

        igroup, ispl, il = node_fof_and_ilist(
            igroup, ispl, il, spl, xleaf, lvl_leaf, 
            rlink=rlink, boxsize=boxsize, alloc_fac=alloc_fac_ilist
        )

        spl = th.ispl_n2n.get(level, size+1)
        
    return igroup, ispl, il, spl
node_node_fof.jit = jax.jit(node_node_fof, static_argnames=["rlink", "boxsize", "alloc_fac_ilist"])

def particle_particle_fof(node_igroup, ispl, il, spl, posz, rlink: float, boxsize: float = 0., block_size=32):
    igroup = jax.ffi.ffi_call("ParticleFof", (jax.ShapeDtypeStruct((len(posz),), node_igroup.dtype),))(
        node_igroup, ispl, il, spl, posz,
        r2link=np.float32(rlink*rlink), boxsize=np.float32(boxsize), block_size=np.int32(block_size)
    )[0]

    return igroup
particle_particle_fof.jit = jax.jit(particle_particle_fof, static_argnames=["rlink", "boxsize", "block_size"])

# ------------------------------------------------------------------------------------------------ #
#                                          User Interface                                          #
# ------------------------------------------------------------------------------------------------ #

def prepare_fof_z(posz: jnp.ndarray, rlink: float, boxsize: float | None = None, 
                  cfg: FofConfig = FofConfig()) -> FofData:
    cfg_fmdj = fmdj.Config(fmm = fmdj.config.FMMConfig(
        alloc_fac_nodes=cfg.alloc_fac_nodes,
        max_leaf_size=cfg.max_leaf_size,
        coarse_fac=cfg.coarse_fac,
        stop_coarsen=cfg.stop_coarsen,
        multipoles_around_com=False
    ))

    posmass_z = fmdj.data.PosMass(posz, jnp.ones((len(posz),), dtype=jnp.float32))
    th: fmdj.data.TreeHierarchy = fmdj.ztree.build_tree_hierarchy(posmass_z, cfg=cfg_fmdj)

    igroup, ispl, il, spl = node_node_fof(
        th, rlink=rlink, boxsize=boxsize, alloc_fac_ilist=cfg.alloc_fac_ilist
    )

    return FofData(rlink, boxsize, posz, igroup, ispl, il, spl)
prepare_fof_z.jit = jax.jit(prepare_fof_z, static_argnames=["rlink", "boxsize", "cfg"])

def evaluate_fof_z(d: FofData):
    # Could in principle support switching out the particle data at this point.
    return particle_particle_fof(
        d.igroup, d.ilist_spl, d.ilist, d.spl, d.posz, rlink=d.rlink, boxsize=d.boxsize
    )
evaluate_fof_z.jit = jax.jit(evaluate_fof_z)

def fof_z(posz: jnp.ndarray, rlink: float, boxsize: float = 0., cfg: FofConfig = FofConfig()) -> jnp.ndarray:
    data = prepare_fof_z(posz, rlink, boxsize, cfg)
    return evaluate_fof_z(data)
fof_z.jit = jax.jit(fof_z, static_argnames=["rlink", "boxsize", "cfg"])

def fof(pos: jnp.ndarray, rlink: float, boxsize: float = 0., cfg: FofConfig = FofConfig()) -> jnp.ndarray:
    posz, idz = pos_zorder_sort(pos)
    
    igroupz = fof_z(posz, rlink, boxsize=boxsize, cfg=cfg)

    igroup = igroupz.at[idz].set(igroupz)

    return igroup
fof.jit = jax.jit(fof, static_argnames=["rlink", "boxsize", "cfg"])
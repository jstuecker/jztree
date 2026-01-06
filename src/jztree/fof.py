import numpy as np
import jax
import jax.numpy as jnp
from jztree_cuda import ffi_fof
from .tree import pos_zorder_sort, grouped_dense_interaction_list
from .common import conditional_callback
from .data import  FofConfig, FofData, PosLvl
from fmdj.data import InteractionList
from typing import Tuple

import fmdj

jax.ffi.register_ffi_target("NodeFofAndIlist", ffi_fof.NodeFofAndIlist(), platform="CUDA")
jax.ffi.register_ffi_target("ParticleFof", ffi_fof.ParticleFof(), platform="CUDA")

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

def node_fof_and_ilist(
        node_igroup: jnp.ndarray, node_ilist: InteractionList, isplit: jnp.ndarray, 
        child_data: PosLvl,
        rlink: float, boxsize: float = 0., alloc_fac:float = 128., block_size:int = 32
    ) -> Tuple[jnp.ndarray, InteractionList]:
    assert node_ilist.ispl.shape[0] == isplit.shape[0], "Should both correspond to no. of nodes+1"
    assert isplit.shape[0] == node_igroup.shape[0]+1, "Should both correspond to no. of nodes"
    assert node_ilist.iother.size < 2**31, "So far only int32 supported {ilist_alloc_size/2**31}"
    
    nchild = len(child_data.pos)

    child_igroup = jax.ShapeDtypeStruct((nchild,), jnp.int32)
    child_ilist_ispl = jax.ShapeDtypeStruct((nchild+1,), jnp.int32)
    child_ilist = jax.ShapeDtypeStruct((int(alloc_fac * nchild),), jnp.int32)
    
    outputs = (child_igroup, child_ilist_ispl, child_ilist)

    res = jax.ffi.ffi_call("NodeFofAndIlist", outputs)(
        node_igroup, node_ilist.ispl, node_ilist.iother, isplit, child_data.pos_lvl(),
        r2link=np.float32(rlink*rlink), boxsize=np.float32(boxsize), block_size=np.int32(block_size)
    )

    child_igroup, child_ilist_ispl, child_ilist = res
    child_ilist = InteractionList(child_ilist_ispl, child_ilist, nfilled=child_ilist_ispl[-1])

    def err(n1, n2):
        raise MemoryError(f"The interaction list allocation is too small. (need: {n1} have: {n2})" +
                          f"increase alloc_fac at least by a factor of {n1/n2:.1f}")
    n1, n2 = child_ilist.nfilled, child_ilist.iother.shape[0]
    child_igroup = child_igroup + conditional_callback(n1 > n2, err, n1, n2)

    return child_igroup, child_ilist
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

def node_node_fof(th: fmdj.data.TreeHierarchy, rlink: float, boxsize: float=0., alloc_fac_ilist: int = 128):
    nplanes = th.num_planes()

    # initialize top-level interaction list
    spl, ilist, nsup = grouped_dense_interaction_list(
        th.lvl.num(nplanes-1), size_ilist=int(th.plane_sizes[nplanes-1]*alloc_fac_ilist), ngroup=32
    )

    # Get coarsest plane data
    igroup = jnp.arange(len(spl)-1, dtype=jnp.int32)

    for level in reversed(range(nplanes)):
        size = th.plane_sizes[level]
        child_data = PosLvl(th.geom_cent.get(level, size), th.lvl.get(level, size))

        igroup, ilist = node_fof_and_ilist(
            igroup, ilist, spl, child_data, 
            rlink=rlink, boxsize=boxsize, alloc_fac=alloc_fac_ilist
        )

        spl = th.ispl_n2n.get(level, size+1)
        
    return igroup, ilist, spl
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

    th: fmdj.data.TreeHierarchy = fmdj.ztree.build_tree_hierarchy(
        posz, cfg_tree=cfg.tree
    )

    igroup, ilist, spl = node_node_fof(
        th, rlink=rlink, boxsize=boxsize, alloc_fac_ilist=cfg.alloc_fac_ilist
    )

    return FofData(rlink, boxsize, posz, igroup, ilist, spl)
prepare_fof_z.jit = jax.jit(prepare_fof_z, static_argnames=["rlink", "boxsize", "cfg"])

def evaluate_fof_z(d: FofData):
    # Could in principle support switching out the particle data at this point.
    return particle_particle_fof(
        d.igroup, d.ilist.ispl, d.ilist.iother, d.spl, d.posz, rlink=d.rlink, boxsize=d.boxsize
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
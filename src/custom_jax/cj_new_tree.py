import numpy as np

import jax
import jax.numpy as jnp

from fmdj.config import Config, TreeConfig

import custom_jax_cuda.ffi_multipoles as ffi_multipoles

from typing import Tuple

jax.ffi.register_ffi_target("multipoles_from_particles", ffi_multipoles.multipoles_from_particles(), platform="CUDA")
jax.ffi.register_ffi_target("coarsen_multipoles", ffi_multipoles.coarsen_multipoles(), platform="CUDA")
jax.ffi.register_ffi_target("evaluate_tree_plane", ffi_multipoles.evaluate_tree_plane(), platform="CUDA")

# Note: This import may break things if imported in the wrong order... Have to fix this later!
from fmdj.new_tree import TreePlane, Multipoles, Particles, InteractionList


def multipoles_from_particles(tp: TreePlane, part: Particles, *, cfg: Config) -> Multipoles:
    cfg_tree: TreeConfig = cfg.tree

    assert cfg_tree.multipoles_around_com

    posm = part.posm()

    assert posm.dtype == jnp.float32
    assert tp.ispl.dtype == jnp.int32

    ncomb = np.array([1, 4, 10, 20, 35, 56, 84, 120, 165])
    
    out_mp = jax.ShapeDtypeStruct((tp.size(), ncomb[cfg_tree.p]), posm.dtype)
    out_xcent = jax.ShapeDtypeStruct((tp.size(), 3), posm.dtype)

    mp, xcent = jax.ffi.ffi_call("multipoles_from_particles", (out_mp, out_xcent))(
        tp.ispl, posm, p=np.uint64(cfg_tree.p), block_size=np.uint64(32)
    )
    
    return Multipoles(xcent=xcent, values=mp, p=cfg_tree.p, around_com=True)
multipoles_from_particles.jit = jax.jit(multipoles_from_particles, static_argnames=['cfg'])

def coarsen_multipoles(mp: Multipoles, tp: TreePlane, *, cfg: Config) -> Multipoles:
    """Determines the multipoles at the next coarser tree plane"""
    assert mp.around_com

    dtype = mp.values.dtype

    assert mp.values.dtype == jnp.float32
    assert tp.ispl.dtype == jnp.int32

    ncomb = np.array([1, 4, 10, 20, 35, 56, 84, 120, 165])

    out_mp = jax.ShapeDtypeStruct((tp.size(), ncomb[mp.p]), dtype)
    out_xcent = jax.ShapeDtypeStruct((tp.size(), 3), dtype)

    mpnew, xcent = jax.ffi.ffi_call("coarsen_multipoles", (out_mp, out_xcent))(
        tp.ispl, mp.values, mp.center(), p=np.uint64(mp.p), block_size=np.uint64(32)
    )
    return Multipoles(xcent=xcent, values=mpnew, p=mp.p, around_com=mp.around_com)
coarsen_multipoles.jit = jax.jit(coarsen_multipoles, static_argnames=['cfg'])

def cj_evaluate_tree_plane(
        plane: TreePlane, 
        plane_lr: TreePlane | None = None,
        ilist_lr: InteractionList | None = None,
        loc_lr: jnp.ndarray | None = None,
        cfg: Config = None
    ) -> Tuple[jnp.ndarray, InteractionList]:
    """
    Evaluates M2L for a tree plane and generates child interaction list.
    
    Inputs:
    - plane: TreePlane at current level
    - plane_lr: TreePlane at lower resolution (parent level), optional
    - ilist_lr: InteractionList at lower resolution, optional
    - loc_lr: Local expansion at lower resolution, optional
    - cfg: Configuration object
    
    Outputs:
    - loc: multipole local expansion, shape (Nchild, M)
    - new_ilist: new interaction list for children
    """
    assert plane_lr is not None
    assert ilist_lr is not None
    
    cfg_tree: TreeConfig = cfg.tree
    nint_out = cfg_tree.ilist_alloc_fac * plane.size()
    
    node_range = jnp.array([0, plane_lr.nnodes], dtype=jnp.int32)
    spl_nodes = plane.ispl
    spl_ilist = ilist_lr.ispl
    ilist_nodes = ilist_lr.iother
    
    nchild = plane.nnodes
    xchild = plane.mp.center()
    
    mp_values = plane.mp.values
    
    # Determine output shapes
    out_loc = jax.ShapeDtypeStruct(plane.mp.values.shape, jnp.float32)
    out_spl_child_ilist = jax.ShapeDtypeStruct((nchild + 1,), jnp.int32)
    out_child_ilist = jax.ShapeDtypeStruct((nint_out,), jnp.int32)
    
    # Make FFI call
    loc, spl_child_ilist, child_ilist = jax.ffi.ffi_call(
        "evaluate_tree_plane",
        (out_loc, out_spl_child_ilist, out_child_ilist)
    )(
        node_range, spl_nodes, spl_ilist, ilist_nodes, xchild, mp_values,
        p=np.uint64(cfg_tree.p), 
        block_size=np.uint64(32),
        epsilon=np.float32(cfg.softening)
    )
    
    # Create interaction list from outputs
    new_ilist = InteractionList(ispl=spl_child_ilist, iother=child_ilist, nfilled=spl_child_ilist[-1])
    
    return loc, new_ilist
cj_evaluate_tree_plane.jit = jax.jit(cj_evaluate_tree_plane, static_argnames=['cfg'])
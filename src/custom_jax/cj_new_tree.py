import numpy as np

import jax
import jax.numpy as jnp

from fmdj.config import Config, TreeConfig

import custom_jax_cuda.ffi_multipoles as ffi_multipoles
import custom_jax_cuda.ffi_fmm as ffi_fmm

from typing import Tuple

jax.ffi.register_ffi_target("MultipolesFromParticles", ffi_multipoles.MultipolesFromParticles(), platform="CUDA")
jax.ffi.register_ffi_target("CoarsenMultipoles", ffi_multipoles.CoarsenMultipoles(), platform="CUDA")
jax.ffi.register_ffi_target("CountInteractionsAndM2L", ffi_fmm.CountInteractionsAndM2L(), platform="CUDA")
jax.ffi.register_ffi_target("InsertInteractions", ffi_fmm.InsertInteractions(), platform="CUDA")

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

    mp, xcent = jax.ffi.ffi_call("MultipolesFromParticles", (out_mp, out_xcent))(
        tp.ispl, posm, p=np.int32(cfg_tree.p), block_size=np.uint64(32)
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

    mpnew, xcent = jax.ffi.ffi_call("CoarsenMultipoles", (out_mp, out_xcent))(
        tp.ispl, mp.values, mp.center(), p=np.int32(mp.p), block_size=np.uint64(32)
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
    spl_nodes = plane_lr.ispl
    spl_ilist = ilist_lr.ispl
    ilist_nodes = ilist_lr.iother
    mp_values = plane.mp.values
    
    nchild = plane.size()

    children = jnp.concatenate((plane.mp.center(), plane.lvl.view(jnp.float32)[...,None]), axis=-1)
    
    # Determine output shapes
    out_loc = jax.ShapeDtypeStruct(plane.mp.values.shape, jnp.float32)
    out_interaction_count = jax.ShapeDtypeStruct((nchild + 1,), jnp.int32)
    
    # Count opened interactions and evaluate M2L
    loc, interaction_counts = jax.ffi.ffi_call(
        "CountInteractionsAndM2L",
        (out_loc, out_interaction_count, )
    )(
        node_range, spl_nodes, spl_ilist, ilist_nodes, children, mp_values,
        p=np.int32(cfg_tree.p),
        softening=np.float32(cfg.softening),
        opening_angle=np.float32(cfg.opening.opening_angle)
    )

    # Insert interactions
    ispl_child = jnp.pad(jnp.cumsum(interaction_counts), (1, 0))
    out_child_ilist = jax.ShapeDtypeStruct((nint_out,), jnp.int32)

    child_ilist = jax.ffi.ffi_call(
        "InsertInteractions",
        (out_child_ilist,)
    )(
        node_range, spl_nodes, spl_ilist, ilist_nodes, children, ispl_child,
        opening_angle=np.float32(cfg.opening.opening_angle)
    )[0]

    # Create interaction list from outputs
    new_ilist = InteractionList(ispl=ispl_child, iother=child_ilist, nfilled=ispl_child[-1])
    
    return loc, new_ilist
cj_evaluate_tree_plane.jit = jax.jit(cj_evaluate_tree_plane, static_argnames=['cfg'])

def simple_arange(n: int) -> jnp.ndarray:
    out_type = jax.ShapeDtypeStruct((n,), jnp.int32)
    arr = jax.ffi.ffi_call("SimpleArange", (out_type,))(
        block_size=np.uint64(64)
    )[0]
    return arr
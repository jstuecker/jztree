from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from custom_jax_cuda import ffi_old_forces
from custom_jax_cuda import ffi_forces

jax.ffi.register_ffi_target("potential", ffi_old_forces.potential(), platform="CUDA")
jax.ffi.register_ffi_target("force", ffi_old_forces.force(), platform="CUDA")
jax.ffi.register_ffi_target("ilist_fphi", ffi_old_forces.ilist_fphi(), platform="CUDA")
jax.ffi.register_ffi_target("ilist_fphi_bwd", ffi_old_forces.ilist_fphi_bwd(), platform="CUDA")

jax.ffi.register_ffi_target("ilist_fphi_new", ffi_forces.IlistForceAndPotKernel(), platform="CUDA")


jax.ffi.register_ffi_target("ForceAndPotential", ffi_forces.ForceAndPotential(), platform="CUDA")

def force_and_potential(x, mass=1., block_size=64, softening=1e-2, kahan=False):
    assert x.dtype == jnp.float32
    assert x.shape[-1] == 3
    assert x.ndim >= 2
    # assert np.prod(x.shape[:-1]) % block_size == 0, "The length of x must be divisible by the block size."
    assert softening > 0, "Epsilon must be positive to deal with self-interaction."

    # Reading on GPU will be more efficient if we read pos and mass together as single float4 values
    xm = jnp.concatenate([x, jnp.broadcast_to(mass, x.shape[:-1])[:,None]], axis=-1)

    out_type = jax.ShapeDtypeStruct(xm.shape, xm.dtype)
    fphi = jax.ffi.ffi_call("ForceAndPotential", (out_type,))(xm, block_size=np.uint64(block_size), epsilon=np.float32(softening),
                                                              kahan=kahan)[0]
    return fphi
force_and_potential.jit = jax.jit(force_and_potential, static_argnames=("block_size", "softening", "kahan"))


# ------------------------------------------------------------------------------------------------ #
#                                Old Kernels (will be removed later)                               #
# ------------------------------------------------------------------------------------------------ #


# ======= Interfaces from CUDA code to Python =======

def potential(x, mass=1., block_size=64, softening=1e-2):
    assert x.dtype == jnp.float32
    assert x.shape[-1] == 3
    assert x.ndim >= 2
    assert np.prod(x.shape[:-1]) % block_size == 0, "The length of x must be divisible by the block size."
    assert softening > 0, "Epsilon must be positive to deal with self-interaction."

    # Reading on GPU will be more efficient if we read pos and mass together as single float4 values
    xm = jnp.concatenate([x, jnp.broadcast_to(mass, x.shape[:-1])[:,None]], axis=-1)

    out_type = jax.ShapeDtypeStruct(x.shape[:-1], x.dtype)
    phi = jax.ffi.ffi_call("potential", (out_type,))(xm, block_size=np.uint64(block_size), epsilon=np.float32(softening))
    return phi[0]
potential_jit = jax.jit(potential, static_argnames=("block_size", "softening"))

def force(x, mass=1., block_size=64, softening=1e-2, get_potential=False):
    assert x.dtype == jnp.float32
    assert x.shape[-1] == 3
    assert x.ndim >= 2
    assert np.prod(x.shape[:-1]) % block_size == 0, "The length of x must be divisible by the block size."
    assert softening > 0, "Epsilon must be positive to deal with self-interaction."

    xm = jnp.concatenate([x, jnp.broadcast_to(mass, x.shape[:-1])[:,None]], axis=-1)

    if get_potential:
        out_type = jax.ShapeDtypeStruct(x.shape[:-1] + (4,), x.dtype)
        f = jax.ffi.ffi_call("force", (out_type,))(xm, block_size=np.uint64(block_size), epsilon=np.float32(softening), get_potential=True)[0]
        return f[..., :3], f[..., 3]  # Return force and potential separately
    else:
        out_type = jax.ShapeDtypeStruct(x.shape[:-1] + (3,), x.dtype)
        f = jax.ffi.ffi_call("force", (out_type,))(xm, block_size=np.uint64(block_size), epsilon=np.float32(softening), get_potential=False)[0]
        return f
force_jit = jax.jit(force, static_argnames=("softening", "get_potential", "block_size"))

def ilist_force(x, isplit, interactions=None, irange=None, mass=1., softening=1e-2, get_potential=True, block_size=32, interactions_per_block=None):
    """A convenience interface to ilist_phi. Calling ilist_fphi directly is prefered"""
    xm = jnp.concatenate([x, jnp.broadcast_to(mass, x.shape[:-1])[:,None]], axis=-1)
    
    fphi = ilist_fphi(xm, isplit, interactions, irange, softening=softening, block_size=block_size, interactions_per_block=interactions_per_block)

    if get_potential:
        return fphi[..., :3], fphi[..., 3]
    else:
        return fphi[..., :3]
ilist_force_jit = jax.jit(ilist_force, static_argnames=("softening", "get_potential", "block_size", "interactions_per_block"))

def ilist_fphi_fwd(xm, isplit, interactions=None, irange=None, softening=1e-2, block_size=32, interactions_per_block=None, new=False):
    """
    Calculates forces and potential through an interaction list.
    isplit: offsets in the particle array that defines different nodes. Each node goes from isplit[i] to isplit[i+1], so len(isplit) = nnodes + 1
    interactions: (Nint, 2) array of interactions, with sink node indices and source node indices.
                   Defaults to using all possible interactions
    irange: If given, only consider interactions[irange[0]:irange[1]]. Can be jax.Array. Useful for dynamically sized interaction lists.
    --- optimization parameters ---
    block_size: The number of particles handled per loop of each warp. 32 should be good usually.
    interactions_per_block: The number of interactions handled by each warp block. No need to change this.
                            should be >> 1 to avoid kernel launch overhead, but small enough to keep all SMs busy
    """
    if interactions is None:
        iarange = jnp.arange(0, len(isplit)-1)
        interactions = jnp.stack(jnp.meshgrid(iarange, iarange, indexing='ij'), axis=-1).reshape(-1, 2)
    if irange is None:
        irange = jnp.array([0, len(interactions)], dtype=jnp.int32)
    if interactions_per_block is None:
        interactions_per_block = np.clip(len(interactions) // 8096, 4, 256)

    assert block_size % 32 == 0, "Please keep multiples of 32 for blocksize"
    assert xm.dtype == jnp.float32
    assert isplit.dtype == jnp.int32
    assert interactions.dtype == jnp.int32
    assert interactions.shape[-1] == 2, "Interactions must be a 2D array with shape (N, 2) where N is the number of interactions."
    assert xm.shape[-1] == 4
    assert xm.ndim >= 2
    assert interactions.ndim == 2
    assert softening > 0, "Epsilon must be positive to deal with self-interaction."

    out_type = jax.ShapeDtypeStruct(xm.shape, xm.dtype)

    
    if not new:
        fphi = jax.ffi.ffi_call("ilist_fphi", (out_type,))(xm, isplit, interactions, irange, block_size=np.uint64(block_size), interactions_per_block=np.uint64(interactions_per_block), epsilon=np.float32(softening))[0]
    else:
        fphi = jax.ffi.ffi_call("ilist_fphi_new", (out_type,))(xm, isplit, interactions, irange, block_size=np.uint64(block_size), interactions_per_block=np.uint64(interactions_per_block), epsilon=np.float32(softening))[0]

    fphi = fphi.at[...,3].add(xm[...,3]/softening) # Remove self-interaction from potential
    
    return fphi, (xm, isplit, interactions, irange)

def ilist_fphi_bwd(softening, block_size, interactions_per_block, res, g):
    xm, isplit, interactions, iminmax = res
    if interactions_per_block is None:
        interactions_per_block = np.clip(len(interactions) // 8096, 4, 256)
    out_type = jax.ShapeDtypeStruct(xm.shape, xm.dtype)

    gxm = jax.ffi.ffi_call("ilist_fphi_bwd", (out_type,))(g, xm, isplit, interactions, iminmax, block_size=np.uint64(block_size), interactions_per_block=np.uint64(interactions_per_block), epsilon=np.float32(softening))[0]

    return (gxm, None, None, None)

@partial(jax.custom_vjp, nondiff_argnames=("softening", "block_size", "interactions_per_block"))
def ilist_fphi(xm, isplit, interactions=None, irange=None, softening=1e-2, block_size=32, interactions_per_block=None, new=False):
    fphi, res = ilist_fphi_fwd(xm, isplit, interactions, irange, softening, block_size, interactions_per_block, new=new)
    return fphi
ilist_fphi.defvjp(ilist_fphi_fwd, ilist_fphi_bwd)

ilist_fphi.jit = jax.jit(ilist_fphi, static_argnames=("softening", "block_size", "interactions_per_block"))

def dense_ilist(n, bls):
    bls = 32
    ispl = jnp.clip(jnp.arange(0, (n // bls) + 1) * bls, 0, n)
    iar = jnp.arange(len(ispl)-1)
    ilist = jnp.stack(jnp.meshgrid(iar, iar, indexing="ij"), axis=-1)
    return ispl, ilist.reshape(-1,2)

# ======= Some reference implemetations that can be used for testing =======

@jax.jit
def potential_pure_jax_jit(x, m=1., softening=1e-2):
    rij2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    rinv = jnp.where(rij2 > 0, 1. / jnp.sqrt(rij2 + softening**2), 0.)
    
    return -jnp.sum(rinv * jnp.broadcast_to(m, x.shape[:-1])[None,:], axis=1)

@jax.jit
def force_pure_jax_jit(x, m=1., softening=1e-2):
    dx = x[:, None] - x[None, :]
    rij2 = jnp.sum(dx ** 2, axis=-1, keepdims=True)
    rinv = jnp.where(rij2 > 0, 1. / jnp.sqrt(rij2 + softening**2), 0.)
    
    return -jnp.sum(dx * rinv**3 * jnp.broadcast_to(m, x.shape[:-1])[None,:,None], axis=1)
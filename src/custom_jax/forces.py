import os
import ctypes

import numpy as np

import jax
import jax.numpy as jnp

import importlib.resources

import custom_jax.nb_forces as nb_forces

jax.ffi.register_ffi_target("potential", nb_forces.potential(), platform="CUDA")
jax.ffi.register_ffi_target("force", nb_forces.force(), platform="CUDA")

# ======= Interfaces from CUDA code to Python =======

def potential(x, mass=1., block_size=64, eps=1e-2):
    assert x.dtype == jnp.float32
    assert x.shape[-1] == 3
    assert x.ndim >= 2
    assert np.prod(x.shape[:-1]) % block_size == 0, "The length of x must be divisible by the block size."
    assert eps > 0, "Epsilon must be positive to deal with self-interaction."

    # Reading on GPU will be more efficient if we read pos and mass together as single float4 values
    xm = jnp.concatenate([x, jnp.broadcast_to(1., x.shape[:-1])[:,None]], axis=-1)

    out_type = jax.ShapeDtypeStruct(x.shape[:-1], x.dtype)
    phi = jax.ffi.ffi_call("potential", (out_type,))(xm, block_size=np.uint64(block_size), epsilon=np.float32(eps))
    return phi[0]
potential_jit = jax.jit(potential, static_argnames=("block_size", "eps"))

def force(x, mass=1., block_size=64, eps=1e-2):
    assert x.dtype == jnp.float32
    assert x.shape[-1] == 3
    assert x.ndim >= 2
    assert np.prod(x.shape[:-1]) % block_size == 0, "The length of x must be divisible by the block size."
    assert eps > 0, "Epsilon must be positive to deal with self-interaction."

    xm = jnp.concatenate([x, jnp.broadcast_to(1., x.shape[:-1])[:,None]], axis=-1)

    out_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
    f = jax.ffi.ffi_call("force", (out_type,))(xm, block_size=np.uint64(block_size), epsilon=np.float32(eps))
    return f[0]
force_jit = jax.jit(force, static_argnames=("block_size", "eps"))

# ======= Some reference implemetations that can be used for testing =======

@jax.jit
def potential_pure_jax_jit(x, eps=1e-2):
    rij2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    rinv = jnp.where(rij2 > 0, 1. / jnp.sqrt(rij2 + eps**2), 0.)
    
    return -jnp.sum(rinv, axis=1)

@jax.jit
def force_pure_jax_jit(x, eps=1e-2):
    dx = x[:, None] - x[None, :]
    rij2 = jnp.sum(dx ** 2, axis=-1, keepdims=True)
    rinv = jnp.where(rij2 > 0, 1. / jnp.sqrt(rij2 + eps**2), 0.)
    
    return -jnp.sum(dx * rinv**3, axis=1)
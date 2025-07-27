import os
import ctypes

import numpy as np

import jax
import jax.numpy as jnp

import importlib.resources

import custom_jax.nb_forces as nb_forces

jax.ffi.register_ffi_target("potential", nb_forces.potential(), platform="CUDA")

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
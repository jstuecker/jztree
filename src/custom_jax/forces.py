import os
import ctypes

import numpy as np

import jax
import jax.numpy as jnp

libfile = os.path.join(os.path.dirname(__file__), "lib/lib_forces.so")

# This is a temporary hack for editable mode working with pip install -e .
# Somehow it keeps placing the .so file in the wrong place...
if not os.path.exists(libfile):
  otherpath = "/home/jens/miniconda3/envs/jax/lib/python3.11/site-packages/custom_jax/lib/lib_forces.so"
  if not os.path.exists(otherpath):
    raise FileNotFoundError(f"Could not find {libfile}.")
  else:
    libfile = otherpath

library = ctypes.cdll.LoadLibrary(libfile)

jax.ffi.register_ffi_target("potential", jax.ffi.pycapsule(library.Potential), platform="CUDA")

def potential(x, block_size=64):
  assert x.dtype == jnp.float32
  assert x.shape[-1] == 3
  assert x.ndim >= 2
  assert np.prod(x.shape[:-1]) % block_size == 0, "The length of x must be divisible by the block size."
  
  out_type = jax.ShapeDtypeStruct(x.shape[:-1], x.dtype)
  phi = jax.ffi.ffi_call("potential", (out_type,))(x, n=np.uint64(block_size))
  return phi[0]
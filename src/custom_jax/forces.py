import os
import ctypes

import numpy as np

import jax
import jax.numpy as jnp

import importlib.resources

main_dir = importlib.resources.files("custom_jax")

possible_path1 = os.path.join(main_dir, "lib", "lib_forces.so")
possible_path2 = os.path.join(main_dir, "..", "..", "build", "lib_forces.so")

if os.path.exists(possible_path1):
  libfile = possible_path1
elif os.path.exists(possible_path2):
  libfile = possible_path2
else:
  raise FileNotFoundError(f"Could not find in {possible_path1} or {possible_path2}.")

library = ctypes.cdll.LoadLibrary(libfile)

jax.ffi.register_ffi_target("potential", jax.ffi.pycapsule(library.Potential), platform="CUDA")

def potential(x, mass=1., block_size=64):
  assert x.dtype == jnp.float32
  assert x.shape[-1] == 3
  assert x.ndim >= 2
  assert np.prod(x.shape[:-1]) % block_size == 0, "The length of x must be divisible by the block size."
  
  # Reading on GPU will be more efficient if we read pos and mass together as single float4 values
  xm = jnp.concatenate([x, jnp.broadcast_to(1., x.shape[:-1])[:,None]], axis=-1)
  
  out_type = jax.ShapeDtypeStruct(x.shape[:-1], x.dtype)
  phi = jax.ffi.ffi_call("potential", (out_type,))(xm, n=np.uint64(block_size))
  return phi[0]
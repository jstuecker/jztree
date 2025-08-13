from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import custom_jax.nb_multipoles as nb_multipoles

jax.ffi.register_ffi_target("m2l", nb_multipoles.m2l(), platform="CUDA")

# ======= Multipole Translators =======

def m2l(mp, x, loc, p=1, block_size=64, eps=1e-2):
    assert x.dtype == jnp.float32
    assert mp.dtype == jnp.float32
    assert loc.dtype == jnp.float32
    assert x.shape[-1] == 3
    assert mp.shape[-1] == ((p+3)*(p+2)*(p+1)) // 6
    assert loc.shape[-1] == ((p+3)*(p+2)*(p+1)) // 6
    assert x.ndim >= 2 and mp.ndim >= 2 and loc.ndim >= 2

    out_type = jax.ShapeDtypeStruct(loc.shape, loc.dtype)
    result = jax.ffi.ffi_call("m2l", (out_type,))(x, mp, p=np.int32(p), block_size=np.uint64(block_size), epsilon=np.float32(eps))
    
    return result[0]
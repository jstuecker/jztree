from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import custom_jax.nb_multipoles as nb_multipoles

jax.ffi.register_ffi_target("ilist_m2l", nb_multipoles.ilist_m2l(), platform="CUDA")

# ======= Multipole Translators =======

def ilist_multipole_to_local(mp, x, interactions=None, iminmax=None, p=1, block_size=64, interactions_per_block=None, eps=1e-2):
    assert x.dtype == jnp.float32
    assert mp.dtype == jnp.float32
    assert x.ndim >= 2 and mp.ndim >= 2 and mp.ndim >= 2
    assert x.shape[-1] == 3
    assert mp.shape[-1] == ((p+3)*(p+2)*(p+1)) // 6

    if interactions is None:
        iarange = jnp.arange(0, len(mp))
        interactions = np.stack((iarange, iarange), axis=-1).astype(np.int32)
    if iminmax is None:
        iminmax = jnp.array([0, len(interactions)], dtype=jnp.int32)
    if interactions_per_block is None:
        interactions_per_block = np.clip(len(interactions) // 8096, 4, 256)

    out_type = jax.ShapeDtypeStruct(mp.shape, mp.dtype)
    loc = jax.ffi.ffi_call("ilist_m2l", (out_type,))(x, mp, interactions, iminmax, p=np.int32(p), block_size=np.uint64(block_size), interactions_per_block=np.uint64(interactions_per_block), epsilon=np.float32(eps))[0]
    
    return loc
ilist_multipole_to_local.jit = jax.jit(ilist_multipole_to_local, static_argnames=("p", "block_size", "eps"))
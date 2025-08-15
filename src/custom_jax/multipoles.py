from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import custom_jax.nb_multipoles as nb_multipoles

jax.ffi.register_ffi_target("ilist_m2l", nb_multipoles.ilist_m2l(), platform="CUDA")
jax.ffi.register_ffi_target("ilist_leaf2node_m2l", nb_multipoles.ilist_leaf2node_m2l(), platform="CUDA")

# ======= Multipole Translators =======

def ilist_multipole_to_local(mp, x, interactions=None, iminmax=None, p=1, block_size=32, interactions_per_block=None, eps=1e-2):
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
        interactions_per_block = np.clip(len(interactions) // (8096*block_size), 1, 256)

    out_type = jax.ShapeDtypeStruct(mp.shape, mp.dtype)
    loc = jax.ffi.ffi_call("ilist_m2l", (out_type,))(x, mp, interactions, iminmax, p=np.int32(p), block_size=np.uint64(block_size), interactions_per_block=np.uint64(interactions_per_block), epsilon=np.float32(eps))[0]
    
    return loc
ilist_multipole_to_local.jit = jax.jit(ilist_multipole_to_local, static_argnames=("p", "block_size", "eps"))

def ilist_leaf_to_local(xnodes, xpart, mpart, isplit,  interactions, iminmax=None, 
                        p=1, interactions_per_block=None, eps=1e-2):
    assert xnodes.dtype == jnp.float32
    assert xpart.dtype == jnp.float32
    assert mpart.dtype == jnp.float32
    assert isplit.dtype == jnp.int32
    assert interactions.dtype == jnp.int32
    assert iminmax is None or iminmax.dtype == jnp.int32

    assert xnodes.ndim >= 2 and xpart.ndim >= 2
    xm = jnp.concatenate((xpart, mpart[..., None]), axis=-1)

    ncomb = ((p+3)*(p+2)*(p+1)) // 6

    if iminmax is None:
        iminmax = jnp.array([0, len(interactions)], dtype=jnp.int32)
    if interactions_per_block is None:
        interactions_per_block = np.clip(len(interactions) // (4096), 1, 256)

    out_type = jax.ShapeDtypeStruct(xnodes.shape[:-1] + (ncomb,), xnodes.dtype)
    loc = jax.ffi.ffi_call("ilist_leaf2node_m2l", (out_type,))(
        xnodes, xm, isplit, jnp.abs(interactions), iminmax, p=np.int32(p), 
        interactions_per_block=np.uint64(interactions_per_block), 
        epsilon=np.float32(eps))[0]
    
    return loc
ilist_leaf_to_local.jit = jax.jit(ilist_leaf_to_local, static_argnames=("p", "eps", "interactions_per_block"))
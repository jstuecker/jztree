from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from custom_jax_cuda import ffi_multipoles as ffi_multipoles

jax.ffi.register_ffi_target("IlistM2L", ffi_multipoles.IlistM2L(), platform="CUDA")
jax.ffi.register_ffi_target("IlistLeaf2NodeM2L", ffi_multipoles.IlistLeaf2NodeM2L(), platform="CUDA")

# ======= Multipole Translators =======

def ilist_node_to_node(xnodes, multipoles, interactions, irange=None, p=1, block_size=32, interactions_per_block=None, softening=1e-2):
    assert xnodes.dtype == jnp.float32
    assert multipoles.dtype == jnp.float32
    assert xnodes.ndim >= 2 and multipoles.ndim >= 2 and multipoles.ndim >= 2
    assert xnodes.shape[-1] == 3
    assert multipoles.shape[-1] == ((p+3)*(p+2)*(p+1)) // 6

    if interactions is None:
        iarange = jnp.arange(0, len(multipoles))
        interactions = np.stack((iarange, iarange), axis=-1).astype(np.int32)
    if irange is None:
        irange = jnp.array([0, len(interactions)], dtype=jnp.int32)
    if interactions_per_block is None:
        interactions_per_block = np.clip(len(interactions) // (8096*block_size), 1, 256)

    out_type = jax.ShapeDtypeStruct(multipoles.shape, multipoles.dtype)
    loc = jax.ffi.ffi_call("IlistM2L", (out_type,))(xnodes, multipoles, interactions, irange, p=np.int32(p), block_size=np.uint64(block_size), interactions_per_block=np.uint64(interactions_per_block), epsilon=np.float32(softening))[0]
    
    return loc
ilist_node_to_node.jit = jax.jit(ilist_node_to_node, static_argnames=("p", "block_size", "softening"))

def ilist_leaf_to_node(xnodes, xpart, mpart, isplit,  interactions, irange=None, 
                        p=1, interactions_per_block=None, softening=1e-2):
    assert xnodes.dtype == jnp.float32
    assert xpart.dtype == jnp.float32
    assert mpart.dtype == jnp.float32
    assert isplit.dtype == jnp.int32
    assert interactions.dtype == jnp.int32
    assert irange is None or irange.dtype == jnp.int32

    assert xnodes.ndim >= 2 and xpart.ndim >= 2
    xm = jnp.concatenate((xpart, mpart[..., None]), axis=-1)

    ncomb = ((p+3)*(p+2)*(p+1)) // 6

    if irange is None:
        irange = jnp.array([0, len(interactions)], dtype=jnp.int32)
    if interactions_per_block is None:
        interactions_per_block = np.clip(len(interactions) // (4096), 2, 32)

    out_type = jax.ShapeDtypeStruct(xnodes.shape[:-1] + (ncomb,), xnodes.dtype)
    loc = jax.ffi.ffi_call("IlistLeaf2NodeM2L", (out_type,))(
        xnodes, xm, isplit, jnp.abs(interactions), irange, p=np.int32(p), 
        interactions_per_block=np.uint64(interactions_per_block), 
        epsilon=np.float32(softening), block_size=np.uint64(32))[0]
    
    return loc
ilist_leaf_to_node.jit = jax.jit(ilist_leaf_to_node, static_argnames=("p", "softening", "interactions_per_block"))
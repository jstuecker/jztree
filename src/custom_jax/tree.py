from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

import custom_jax.nb_tree as nb_tree

jax.ffi.register_ffi_target("tree", nb_tree.tree(), platform="CUDA")

def tree(key, block_size=64):
    assert key.dtype == jnp.int32

    out_type = jax.ShapeDtypeStruct(key.shape, jnp.int32)
    phi = jax.ffi.ffi_call("tree", (out_type,))(key, block_size=np.uint64(block_size))
    return phi[0]
potential_jit = jax.jit(tree, static_argnames=("block_size"))
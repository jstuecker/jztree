from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

import custom_jax.nb_tree as nb_tree

jax.ffi.register_ffi_target("argsort", nb_tree.argsort(), platform="CUDA")
jax.ffi.register_ffi_target("i3zsort", nb_tree.i3zsort(), platform="CUDA")
jax.ffi.register_ffi_target("f3zsort", nb_tree.f3zsort(), platform="CUDA")
jax.ffi.register_ffi_target("i3argsort", nb_tree.i3argsort(), platform="CUDA")
jax.ffi.register_ffi_target("i3zmergesort", nb_tree.i3zmergesort(), platform="CUDA")
jax.ffi.register_ffi_target("f3zmergesort", nb_tree.f3zmergesort(), platform="CUDA")

def argsort(key, block_size=64):
    assert key.dtype == jnp.int32

    out_type = jax.ShapeDtypeStruct(key.shape[0:1], jnp.int32)
    isort = jax.ffi.ffi_call("argsort", (out_type,))(key, block_size=np.uint64(block_size))
    return isort[0]
argsort.jit = jax.jit(argsort, static_argnames=("block_size",))

def i3zsort(ids, block_size=64):
    assert ids.dtype == jnp.int32
    assert ids.shape[-1] == 3

    out_type = jax.ShapeDtypeStruct(ids.shape[0:1], jnp.int32)
    isort = jax.ffi.ffi_call("i3zsort", (out_type,))(ids, block_size=np.uint64(block_size))
    return isort[0]
i3zsort.jit = jax.jit(i3zsort, static_argnames=("block_size",))

def f3zsort(x, block_size=64):
    assert x.dtype == jnp.float32
    assert x.shape[-1] == 3

    out_type = jax.ShapeDtypeStruct(x.shape[0:1], jnp.int32)

    isort = jax.ffi.ffi_call("f3zsort", (out_type,))(x, block_size=np.uint64(block_size))
    return isort[0]
f3zsort.jit = jax.jit(f3zsort, static_argnames=("block_size",))

def i3argsort(ids, block_size=64):
    assert ids.dtype == jnp.int32
    assert ids.shape[-1] == 3

    out_type = jax.ShapeDtypeStruct(ids.shape[0:1], jnp.int32)
    isort = jax.ffi.ffi_call("i3argsort", (out_type,))(ids, block_size=np.uint64(block_size))
    return isort[0]
i3argsort.jit = jax.jit(i3argsort, static_argnames=("block_size",))

def i3zmergesort(ids, block_size=64):
    assert ids.dtype == jnp.int32
    assert ids.shape[-1] == 3

    out_type = jax.ShapeDtypeStruct((ids.shape[0],4), jnp.int32)
    isort = jax.ffi.ffi_call("i3zmergesort", (out_type,))(ids, block_size=np.uint64(block_size))
    return isort[0][:,3]
i3zmergesort.jit = jax.jit(i3zmergesort, static_argnames=("block_size",))

def f3zmergesort(x, block_size=64):
    assert x.dtype == jnp.float32
    assert x.shape[-1] == 3

    # To optimize memory layout, we bundle position and id together into a single array
    # We later need to reinterprete the output to extract positions and ids
    out_type = jax.ShapeDtypeStruct((x.shape[0],4), jnp.int32)
    isort = jax.ffi.ffi_call("f3zmergesort", (out_type,))(x, block_size=np.uint64(block_size))

    pos = isort[0][:, :3].view(jnp.float32)
    ids = isort[0][:, 3].view(jnp.int32)

    return pos, ids
f3zmergesort.jit = jax.jit(f3zmergesort, static_argnames=("block_size",))
import numpy as np
import jax
import jax.numpy as jnp
from custom_jax_cuda import ffi_example

jax.ffi.register_ffi_target("SimpleArange", ffi_example.SimpleArange(), platform="CUDA")
jax.ffi.register_ffi_target("SetToConstantCall", ffi_example.SetToConstantCall(), platform="CUDA")

def simple_arange(size: int, add: jnp.ndarray | None = None, p: int = 1) -> jnp.ndarray:
    out = jax.ShapeDtypeStruct((size,), jnp.int32)

    if add is None:
        add = jnp.zeros(out.shape, dtype=out.dtype)
    else:
        assert add.shape == out.shape
        assert add.dtype == out.dtype

    fn = jax.ffi.ffi_call("SimpleArange", (out,))(
        add, size=np.int32(size), p=np.int32(p), block_size=np.uint64(32)
    )
        
    return fn

def set_to_constant(size: int, value: int, tpar: int = 16) -> jnp.ndarray:
    out = jax.ShapeDtypeStruct((size,), jnp.int32)

    fn = jax.ffi.ffi_call("SetToConstantCall", (out,))(
        value=np.int32(value), tpar=np.int32(tpar), block_size=np.uint64(64)
    )

    return fn

# Define a main function so that it is easy to check that this works
if __name__ == "__main__":
    print("(1):", simple_arange(40))
    print("(2):", simple_arange(20, add=jnp.ones((20,), dtype=jnp.int32)))
    print("(3):", simple_arange(20, add=jnp.ones((20,), dtype=jnp.int32), p=3))

    print("(4):", set_to_constant(10, value=5, tpar=15))
    print("(5):", set_to_constant(10, value=7, tpar=32))
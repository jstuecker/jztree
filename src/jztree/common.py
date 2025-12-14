import jax
from jax.experimental import checkify, io_callback
import functools
import jax.numpy as jnp

def conditional_callback(flag, f, *args, **kwargs):
    """Calls a device function f, only if the flag is True. Useful for raising exceptions that 
    truly stop the execution of a jitted program

    returns an integer that is set to 0 if the callback is not triggered. This can be used to force
    the jax graph to resolve the condition before continuing the graph, e.g. as in
    """

    res = jax.lax.cond(
        flag, 
        lambda : io_callback(f, jax.ShapeDtypeStruct((), jnp.int32), *args, **kwargs),
        lambda : jnp.int32(0)
    )

    return res

def offset_sum(num):
    cs = jnp.cumsum(num, axis=0)
    return cs - num, cs[-1]

def cumsum_starting_with_zero(num):
    return jnp.pad(jnp.cumsum(num), (1, 0))

def masked_prefix_sum(mask):
    off, _ = offset_sum(mask)
    off_masked = jnp.where(mask, off, len(mask))
    return off_masked

def inverse_indices(iargsort):
    """Given the indices that would sort an array, return the indices that would unsort it"""
    iunsort = jnp.zeros_like(iargsort)
    iunsort = iunsort.at[iargsort].set(jnp.arange(len(iargsort), dtype=iargsort.dtype))
    return iunsort
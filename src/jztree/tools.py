import inspect
import os
import jax
from jax.experimental import io_callback
import jax.numpy as jnp
from typing import TypeAlias, Any

from .config import LoggingConfig

Pytree: TypeAlias = Any

# ------------------------------------------------------------------------------------------------ #
#                                          Errors and Logs                                         #
# ------------------------------------------------------------------------------------------------ #

def _callsite(depth=2, *, shorten=True):
    """
    depth=2 -> caller of `log` (this -> log -> caller).
    """
    frame = inspect.currentframe()
    # Walk back `depth` frames safely
    for _ in range(depth):
        frame = frame.f_back if frame is not None else None
    if frame is None:
        return "<unknown:0>"

    info = inspect.getframeinfo(frame)
    path = info.filename
    if shorten:
        try:
            path = os.path.relpath(path)
        except Exception:
            pass
        path = os.path.basename(path)  # or keep full relpath if you prefer
    return f"{path}:{info.lineno}"

def log(txt, 
        *args, 
        level : int = 0,
        cfg_log : LoggingConfig | None = None,
        ordered : bool = False,
        partitioned : bool = False,
        **kwargs):
    if cfg_log is not None:
        if level > cfg_log.level:
            return
        if cfg_log.show_loc:
            txt = f"[{_callsite()}] {txt}"

    jax.debug.print(txt, *args, ordered=ordered, partitioned=partitioned, **kwargs)

# ------------------------------------------------------------------------------------------------ #
#                               Some frequently used helper functions                              #
# ------------------------------------------------------------------------------------------------ #

def div_ceil(a, b):
    return (a + b - 1) // b

def cumsum_starting_with_zero(x):
    return jnp.pad(jnp.cumsum(x), (1, 0))

def offset_sum(num):
    cs = jnp.cumsum(num, axis=0)
    return cs - num, cs[-1]

def masked_prefix_sum(mask):
    off, n = offset_sum(mask)
    off_masked = jnp.where(mask, off, len(mask))
    return off_masked, n

def inverse_of_splits(ispl, size):
    """given [0, 4, 7] returns [0,0,0,0,1,1,1] for size=7"""
    mask = jnp.zeros(size, dtype=jnp.int32).at[ispl].add(1)
    return jnp.cumsum(mask) - 1

def inverse_indices(iargsort):
    """Given the indices that would sort an array, return the indices that would unsort it"""
    iunsort = jnp.zeros_like(iargsort)
    iunsort = iunsort.at[iargsort].set(jnp.arange(len(iargsort), dtype=iargsort.dtype))
    return iunsort

def bucket_prefix_sum(key, count=None, num=None):
    """A prefix sum per key: result = sum(count[key[:i] == key[i]]), but jittable
    i.e. the prefix-sum of points with the same index"""
    if num is not None:
        key = jnp.where(jnp.arange(len(key)) < num, key, jnp.iinfo(key.dtype).max)

    isort = jnp.argsort(key, stable=True)
    key_sort = key[isort]
    if count is None:
        csum_sort = jnp.arange(len(key), dtype=key.dtype)
    else:
        count_sort = count[isort]
        csum_sort = jnp.cumsum(count_sort) - count_sort
    ifirst = jnp.searchsorted(key_sort, key_sort, side="left")
    rank = jax.lax.axis_index("gpus")
    
    cdiff = csum_sort - csum_sort[ifirst]
    if num is not None:
        isort = jnp.where(jnp.arange(len(isort)) < num, isort, len(isort))
    invsort = jnp.zeros_like(isort).at[isort].set(jnp.arange(len(isort)))

    return cdiff[invsort]

def smallest_larger_than(values, ind, at_end=None):
    """Assumes values are sorted"""
    ibin = jnp.searchsorted(values, ind, side="right") - 1
    ibin = jnp.where(ibin >= 0, ibin, len(values))

    imin = jnp.full(len(values)-1, jnp.iinfo(jnp.int32).max).at[ibin].min(ind)
    if at_end is not None:
        imin = jnp.pad(imin, (0,1), constant_values=at_end)
    imin = jnp.minimum.accumulate(imin[::-1])[::-1]
    
    return imin

# ------------------------------------------------------------------------------------------------ #
#                                          Scatter Helpers                                         #
# ------------------------------------------------------------------------------------------------ #

def masked_scatter(mask, arr, indices, values):
    indices = jnp.where(mask, indices, len(arr))
    return arr.at[indices].set(values)

def multi_to_dense(x: jax.Array, spl: jax.Array, out_size: int | None = None) -> jax.Array:
    """x[ndev,n], spl[ndev+1] -> x[ndev*n]"""
    ndev = len(x)
    xout = jnp.zeros(((ndev*x.shape[1],) + x.shape[2:]), x.dtype)
    iarange = jnp.arange(x.shape[1])
    idev = jnp.arange(ndev)
    
    indices = spl[idev,None] + iarange[None,:]
    xout = masked_scatter(indices < spl[idev+1,None], xout, indices, x)

    if out_size is not None:
        return xout[:out_size]

    return xout
multi_to_dense.jit = jax.jit(multi_to_dense)

def set_range(arr : jax.Array, values, start, end):
    if(len(arr) / len(values) >= 4):
        # values are much smaller than arr, do a scatter based update
        idx = jnp.arange(len(values)) + start
        idx = jnp.where(idx < end, idx, len(arr))
        return arr.at[idx].set(values, unique_indices=True)
    else:
        # Do a masked update
        idx = jnp.arange(len(arr))
        cond = (idx >= start) & (idx < end)
        cond = cond.reshape((-1,) + (1,) * (values.ndim - 1))
        return jnp.where(cond, values[idx - start], arr)

# ------------------------------------------------------------------------------------------------ #
#                                    Some useful jax constructs                                    #
# ------------------------------------------------------------------------------------------------ #

def fori_dynamic_over_static(lower, upper, body_fun, init_val, *, unroll=None, nstatic=None):
    """Does a loop with a dynamical boundary over a loop with static boundaries

    This function can have two advantages over a standard jax.lax.fori_loop with dynamic
    boundaries: (1) it allows you to unroll the (inner) static loop partially. (E.g. try unroll=4)
    (2) Static loops seem to be a lot faster in jax. Honestly, I don't know why!

    The usage is the same as jax.fori_loop, with the important difference that the loop
    may also be executed a few extra times if (upwer-lower) is not divisible by nstatic.
    So be sure to discard invalid iterations in your function!

    Example:
    def f(iter, x):
        return jnp.where(iter < 10, x + 1, x)

    print(fori_dynamic_over_static(0, 10, f, 0., nstatic=7))
    """

    if nstatic is None:
        return jax.lax.fori_loop(lower, upper, body_fun, init_val, unroll=unroll)
    
    
    def outer_body(iouter, state):
        def inner_body(iinner, state):
            return body_fun(lower + iouter*nstatic + iinner, state)
        return jax.lax.fori_loop(0, nstatic, inner_body, state, unroll=unroll)

    ndynamic = div_ceil(upper - lower, nstatic)
    return jax.lax.fori_loop(0, ndynamic, outer_body, init_val)
import inspect
import os
import numpy as np
import jax
from jax.experimental import io_callback
import jax.numpy as jnp
from typing import TypeAlias, Any, Tuple
from .jax_ext import tree_map_by_len, pytree_len

from .config import LoggingConfig

Pytree: TypeAlias = Any

from jztree_cuda import ffi_tools
jax.ffi.register_ffi_target("RearangeSegments", ffi_tools.RearangeSegments(), platform="CUDA")

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

def size_bytes(x: jax.Array):
    return np.int64(jnp.size(x)) * np.int64(jnp.dtype(x).itemsize)

def rearange_segments(data, seg_spl_out, seg_offset_in, block_size=64):
    data_out = jax.ShapeDtypeStruct(data.shape, data.dtype)

    with jax.enable_x64():
        return jax.ffi.ffi_call("RearangeSegments", (data_out,))(
            data, jnp.astype(seg_spl_out, jnp.int64), jnp.astype(seg_offset_in, jnp.int64),
            size=np.int64(len(data)), dtype_bytes=size_bytes(data[0]),
            grid_size=np.uint64(div_ceil(len(data), block_size)), block_size=np.uint64(block_size)
        )[0]

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

def masked_to_dense(arr: jax.Array, mask, get_inverse=False, get_indices=False, fill_value=0):
    pref, num = offset_sum(mask)
    size = pytree_len(arr)
    pref = jnp.where(mask, pref, size)
    def upd(x):
        return jnp.full(x.shape, fill_value, x.dtype).at[pref].set(x)

    new_arr = tree_map_by_len(upd, arr, len(mask))
    res = [new_arr, num]
    if get_inverse:
        res.append(pref)
    if get_indices:
        ind = jnp.full(size, size, pref.dtype).at[pref].set(jnp.arange(size))
        res.append(ind)
    return res

def inverse_of_splits(ispl, size):
    """given [0, 4, 7] returns [0,0,0,0,1,1,1] for size=7"""
    mask = jnp.zeros(size, dtype=jnp.int32).at[ispl].add(1)
    return jnp.cumsum(mask) - 1

def inverse_indices(iargsort):
    """Given the indices that would sort an array, return the indices that would unsort it"""
    iunsort = jnp.zeros_like(iargsort)
    iunsort = iunsort.at[iargsort].set(jnp.arange(len(iargsort), dtype=iargsort.dtype))
    return iunsort

def masked_inverse(indices, mask):
    """Given the indices that would sort an array, return the indices that would unsort it"""
    inverse = jnp.full_like(indices, len(indices))
    imask = jnp.where(mask, indices, len(indices))
    inverse = inverse.at[imask].set(jnp.arange(len(indices), dtype=indices.dtype))
    return inverse

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
#                                       Ragged Array Helpers                                       #
# ------------------------------------------------------------------------------------------------ #

def ragged_transpose(data: jax.Array, n: jax.Array, axes: Tuple[int]):
    """Transposes a matrix of segments in a (ragged) flat array
    Segments are defined in the flat array as follows:
    segment[i,j,...] = data[offsets[i,j,...]:offsets[i,j,...]+n[i,j,...]]
    where offsets are determined by the cumulative sum over flattened "n"
    if d is a pytree, we apply over all leaves that have the maximum length
    """
    assert n.ndim == len(axes)
    assert pytree_len(data) < 2**31, "Int32 overflow likely..."

    if axes == tuple(range(n.ndim)): # already have this order... nothing to do!
        return data, n

    # Define transposition on segment indices
    iseg = jnp.arange(n.size).reshape(n.shape)
    iseg_from = jnp.transpose(iseg, axes).flatten()

    # determine segment offsets
    n_T = jnp.transpose(n, axes)
    seg_spl_out = jnp.pad(jnp.cumsum(n_T.flatten()), (1,0), constant_values=0)
    seg_off_in = (jnp.cumsum(n.flatten()) - n.flatten())[iseg_from]
    
    # find which particle belongs to which segment in the output and its internal offset
    # idx_seg = jnp.cumsum(jnp.zeros(pytree_len(data), dtype=jnp.int32).at[seg_off_out+n_T.flatten()].add(1))
    # idx = jnp.arange(len(data))
    # idx_delta = idx - seg_off_out[idx_seg]

    # do the transpose through a gather
    # idx_from = seg_off_in[iseg_from[idx_seg]] + idx_delta

    def rearrange(x):
        return rearange_segments(x, seg_spl_out, seg_off_in)
    
    return tree_map_by_len(rearrange, data, pytree_len(data)), n_T

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
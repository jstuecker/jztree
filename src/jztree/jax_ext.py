# ------------------------------------------------------------------------------------------------ #
#              This module contains extensions or variations of jax's build in methods             #
# ------------------------------------------------------------------------------------------------ #

import inspect
import numpy as np
from typing import Tuple, TypeAlias, Any
from functools import wraps
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental import io_callback

Pytree: TypeAlias = Any

# ------------------------------------------------------------------------------------------------ #
#                                       Invalidation Helpers                                       #
# ------------------------------------------------------------------------------------------------ #

def value_for_dtype(dtype, float_val=jnp.nan, int_val=0):
    if dtype.kind == "f":
        return float_val
    else:
        return int_val

def empty_like(x, float_val=jnp.nan, int_val=0, by_len=True):
    def empty_el(xi):
        return jnp.full_like(xi, fill_value=value_for_dtype(xi.dtype, float_val, int_val))

    if by_len:
        return tree_map_by_len(empty_el, x, pytree_len(x))
    else:
        return jax.tree.map(empty_el, x)

def invalidate(arr, mask, invalid_float=jnp.nan, invalid_int=0):
    def inv(x):
        mask_rs = jnp.reshape(mask, mask.shape + (1,)*(x.ndim-1))
        return jnp.where(mask_rs, value_for_dtype(x.dtype, invalid_float, invalid_int), x)
    
    return tree_map_by_len(inv, arr, pytree_len(arr))

# ------------------------------------------------------------------------------------------------ #
#                                          Mapping PyTrees                                         #
# ------------------------------------------------------------------------------------------------ #

def leading_len(x):
    if jnp.size(x) == 1:
        return 1
    else:
        return len(x)

def pytree_len(x):
    """Returns the size of the largest first axis of any leaf of a pytree"""
    leaves = jax.tree_util.tree_leaves(x)

    return max(leading_len(x) for x in leaves)

def tree_map_by_len(fn, tree: Pytree, N: int, axis: int = 0) -> Pytree:
    """
    Like jax.tree_map, but only applies fn to array-like leaves with shape[axis] == N.
    Everything else is copied unchanged.
    """
    def should_map(x):
        return (
            hasattr(x, "shape")
            and x.shape is not None
            and len(x.shape) > axis
            and int(x.shape[axis]) == int(N)
        )

    return jax.tree_util.tree_map(lambda x: fn(x) if should_map(x) else x, tree)

def concatenate_pytrees(xs, nums):
    assert len(nums) == len(xs)
    num_keys = len(xs)
    
    treedef = jax.tree_util.tree_flatten(xs[0])[1]

    leaves = [jax.tree_util.tree_flatten(x)[0] for x in xs]
    new_leaves = []
    for i in range(len(leaves[0])):
        l = [leaves[t][i] for t in range(num_keys)]
        if(jnp.ndim(l[0]) > 0):
            valid_mask = jnp.concatenate([jnp.arange(len(l[t])) < nums[t] for t in range(num_keys)])
            new = jnp.concatenate(l)
            new_leaves.append(
                jnp.compress(valid_mask, new, size=len(valid_mask), axis=0,
                             fill_value=value_for_dtype(new.dtype))
            )
        else:
            new_leaves.append(l)

    return jax.tree_util.tree_unflatten(treedef, new_leaves)

def separate_pytrees(x, key, out_sizes, num):
    size = len(key)
    num_keys = len(out_sizes)

    assert np.max(out_sizes) <= size
    
    key = jnp.where(jnp.arange(size) < num, key, num_keys)
    isort = jnp.argsort(key)

    leaves, treedef = jax.tree_util.tree_flatten(x)

    res, nums = [], []
    offset = 0
    for t in range(0, num_keys):
        new_num = jnp.sum(key == t)
        invalid = jnp.arange(out_sizes[t]) >= new_num
        ifrom = jnp.roll(isort, -offset)[:out_sizes[t]]
        new_leaves = []
        for l in leaves:
            if (jnp.ndim(l) >= 1) and (len(l) == size):
                new_leaves.append(invalidate(l[ifrom], invalid))
            else:
                new_leaves.append(l)
        offset += new_num

        res.append(jax.tree_util.tree_unflatten(treedef, new_leaves))
        nums.append(new_num)
    
    return res, nums

# ------------------------------------------------------------------------------------------------ #
#                                          Raising Errors                                          #
# ------------------------------------------------------------------------------------------------ #

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

def _capture_frames_above_raise_if(*, depth_from_here=2, max_depth=12):
    """
    Return a list[FrameInfo] starting at the user callsite (where raise_if is used),
    then its callers, etc., up to max_depth.
    """
    frame = inspect.currentframe()
    for _ in range(depth_from_here):
        frame = frame.f_back if frame is not None else None
    if frame is None:
        return []

    out = []
    for _ in range(max_depth):
        if frame is None:
            break
        out.append(inspect.getframeinfo(frame))
        frame = frame.f_back
    return out

def _format_trace(frames):
    """
    Python-traceback-like ordering: most recent call last (outermost first).
    We collected from inner->outer, so reverse.
    """
    lines = []
    for fi in reversed(frames):
        lines.append(f'  File "{fi.filename}", line {fi.lineno}, in {fi.function}')
    return lines

def raise_if( pred, msg, *fmt_args, exc=RuntimeError, max_trace_depth=12, **fmt_kwargs, ):
    pred = jnp.asarray(pred)

    # Capture trace-time call chain starting at the *callsite* of raise_if
    frames = _capture_frames_above_raise_if(depth_from_here=2, max_depth=max_trace_depth)

    # The callsite frame is the first element we captured (innermost)
    if frames:
        callsite = f"{frames[0].filename}:{frames[0].lineno}"
    else:
        callsite = "<unknown>:0"

    trace_lines = _format_trace(frames)

    def _raise(*args, **kwargs):
        main = msg.format(*args, **kwargs)

        txt = "\n======== Relevant Error Message =========\n"
        txt += f"{exc.__name__} at {callsite}:\n{main}\n"

        txt = txt + f" Trace (last {max_trace_depth}, tracing time, most recent call last):\n"
        txt = txt + "-----------------------------------------\n"
        txt = txt + "\n".join(trace_lines) + "\n"
        txt = txt + "=========================================\n"

        raise exc(txt)

    def do_raise(_):
        return io_callback( _raise, jax.ShapeDtypeStruct((), jnp.int32), *fmt_args, **fmt_kwargs, )

    return jax.lax.cond(pred, do_raise, lambda _: jnp.int32(0), operand=None)

# ------------------------------------------------------------------------------------------------ #
#                                        General Device Info                                       #
# ------------------------------------------------------------------------------------------------ #

def get_rank_info(axis_name = None) -> Tuple[int, int, str]:
    """Returns rank, ndev and axis_names of the abstract mesh"""
    axis_name = axis_name or jax.sharding.get_abstract_mesh().axis_names
    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    return rank, ndev, axis_name

def pcast_vma(x, vma):
    if hasattr(jax.lax, "pcast"):
        return jax.lax.pcast(x, tuple(vma), to="varying")
    else:
        return jax.lax.pvary(x, tuple(vma))

def pcast_like(x, like):
    if hasattr(jax.lax, "pcast"):
        return jax.lax.pcast(x, tuple(jax.typeof(like).vma), to="varying")
    else:
        return jax.lax.pvary(x, tuple(jax.typeof(like).vma))

# ------------------------------------------------------------------------------------------------ #
#                                       Expanding Shard Map                                        #
# ------------------------------------------------------------------------------------------------ #

def expanding_shard_map(f, *, out_specs=None, in_specs=None, mesh=None, 
                       axis_names=None, check_vma=True, input_tiled=False, output_tiled=False,
                       jit=False):
    """Like jax.shard_map, but allows to choose whether inputs/outputs well be tiled or not
    
    input_tiled: e.g. input (Ndev*N) -> (N) inside mapped function
    not input_tiled:        (Ndev,N) -> (N)
    output_tiled:      (N) inside mapped function -> (Ndev*N) outside
    not output_tiled:  (N) -> (Ndev,N)
    (or no N and (,N) if partition spec is empty)

    Additionally, P(-1) may be used to define a sharding over all axes of the mesh

    These rules will only be applied to arguments with a varying partition spec
    Inputs and outputs with partition spec P() will be kept unchanged
    """
    if mesh is None:
        mesh = jax.sharding.get_abstract_mesh()
    if axis_names is None:
        axis_names = mesh.axis_names
    if in_specs is None:
        in_specs = P(axis_names)
    if out_specs is None:
        out_specs = P(axis_names)

    def insert_all(spec): # We use -1 to map over all mesh axes
        if spec == P(-1):
            return P(axis_names)
        else:
            return spec
    
    in_specs = jax.tree.map(insert_all, in_specs)
    out_specs = jax.tree.map(insert_all, out_specs)
    
    def squeeze_first_dim(x: jax.Array, flag_keep):
        if flag_keep is None or flag_keep: return x
        assert x.shape[0] == 1
        return jnp.reshape(x, jnp.shape(x)[1:])
    
    def expand_first_dim(x: jax.Array, flag_keep):
        if flag_keep is None or flag_keep: return x
        return jnp.reshape(x, (1,) + jnp.shape(x))
    
    def is_constant(spec):
        if spec is None: return True
        return spec == P()

    @jax.shard_map(out_specs=out_specs, in_specs=in_specs, mesh=mesh, axis_names=set(axis_names), check_vma=check_vma)
    def f_smapped(*args):
        if not input_tiled:
            in_specs_broad = jax.tree.broadcast(in_specs, args, is_leaf=lambda x: x is None)
            flag_const = jax.tree.map(is_constant, in_specs_broad)
            args = jax.tree.map(squeeze_first_dim, args, flag_const)
        res = f(*args)
        if not output_tiled:
            out_specs_broad = jax.tree.broadcast(out_specs, res, is_leaf=lambda x: x is None)
            flag_const = jax.tree.map(is_constant, out_specs_broad)
            return jax.tree.map(expand_first_dim, res, flag_const)
        else:
            return res
    
    if jit:
        f_smapped = jax.jit(f_smapped)

    return f_smapped

# ------------------------------------------------------------------------------------------------ #
#                                   Dynamic Shard Map Constructor                                  #
# ------------------------------------------------------------------------------------------------ #

def _flatten_kwargs(f, *args, **kwargs):
    sig = inspect.signature(f)

    # Reject unsupported signatures
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(f"{f.__name__} has keyword-only parameter '{p.name}', cannot flatten.")
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            raise TypeError(f"{f.__name__} has **kwargs parameter '{p.name}', cannot flatten.")

    # Validate/resolve the call (missing/duplicate/unexpected are raised here)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    params = list(sig.parameters.values())
    pos_params = [p for p in params
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                               inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    var_pos = next((p for p in params if p.kind is inspect.Parameter.VAR_POSITIONAL), None)

    fixed = [bound.arguments[p.name] for p in pos_params]
    rest = list(bound.arguments.get(var_pos.name, ())) if var_pos else []

    return fixed + rest

def shard_map_constructor(f, out_specs=None, in_specs=None, default_mesh=None,
                     axis_names=None, check_vma=True, input_tiled=False,
                     output_tiled=False, **jit_kwargs):
    # cache functions, to avoid recompiled if invoking with the same mesh twice
    cache = {}

    def fconstr(mesh=None, jit=False):
        mesh = mesh or default_mesh
        assert mesh is not None, "Please specify a mesh"

        key = (tuple(mesh.axis_names), tuple(mesh.axis_types), mesh.devices.shape, jit)

        if key in cache:
            return cache[key]

        fsm = expanding_shard_map(
            f, out_specs=out_specs, in_specs=in_specs, mesh=mesh, axis_names=axis_names,
            check_vma=check_vma, input_tiled=input_tiled, output_tiled=output_tiled
        )

        # Currently shard_map doesn't support keyword arguments. Fix this through a wrapper
        @wraps(f)
        def fkw(*args, **kwargs):
            return fsm(*_flatten_kwargs(f, *args, **kwargs))
        fkw.__signature__ = inspect.signature(f)

        if jit:
            fkw = jax.jit(fkw, **jit_kwargs)
        
        cache[key] = fkw

        return fkw
    return fconstr

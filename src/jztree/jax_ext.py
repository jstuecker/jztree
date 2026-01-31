# ------------------------------------------------------------------------------------------------ #
#              This module contains extensions or variations of jax's build in methods             #
# ------------------------------------------------------------------------------------------------ #

import inspect
from typing import Tuple, TypeAlias, Any
from functools import wraps
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental import io_callback

Pytree: TypeAlias = Any

# ------------------------------------------------------------------------------------------------ #
#                                          Mapping PyTrees                                         #
# ------------------------------------------------------------------------------------------------ #

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

        key = (tuple(mesh.axis_names), mesh.devices.shape, jit)

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

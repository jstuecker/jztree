import jax
from jax.experimental import checkify, io_callback
import functools
import jax.numpy as jnp

def jzit(f, enable: bool = True, errors = checkify.user_checks, **jit_kwargs):
    """
    Decorate a function so that:
      - checks inside it (via checkify.check) are made jit-safe, and
      - any failure raises on the host after execution.
    If enable=False, it becomes a plain jitted function (fast path).
    """
    
    if not enable:
        return jax.jit(f, **jit_kwargs)

    cf = checkify.checkify(f, errors=errors)
    jf = jax.jit(cf, **jit_kwargs)

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        err, out = jf(*args, **kwargs)
        err.throw()   # materializes + raises if any check failed
        return out
    return wrapped

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
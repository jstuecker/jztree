import pytest
import jax
import jax.numpy as jnp
from jztree.jax_ext import conditional_callback
from jax.experimental import io_callback
from jztree.fof import _masked_min_scatter

@pytest.mark.skip_in_quick
@pytest.mark.parametrize("N", [int(1e5), int(1e6), int(1e7)])
def bench_host_callback(jax_bench, N):
    def f(mode = 0):
        key = jax.random.key(0)

        def fhost(x):
            return 0.

        out = 0.
        for i in range(10):
            key = jax.random.split(key, 1)[0]
            val = jax.random.uniform(key, (N,2))
            vsum = jnp.sum(val + out)
            if mode == 1:
                vsum = vsum + conditional_callback(vsum < 0., fhost, vsum)
            elif mode == 2:
                vsum = vsum + io_callback(fhost, jax.ShapeDtypeStruct((), jnp.float32), vsum)
            out = out + vsum
        return out
    f.jit = jax.jit(f, static_argnames="mode")

    jb = jax_bench(jit_rounds=25, jit_loops=5, jit_warmup=5)
    jb.measure(fn=f, fn_jit=f.jit, mode=0, tag="no-callback")
    jb.measure(fn=f, fn_jit=f.jit, mode=1, tag="cond-callback")
    jb.measure(fn=f, fn_jit=f.jit, mode=2, tag="host-callback")

@pytest.mark.skip_in_quick
@pytest.mark.parametrize("N", [int(1e5), int(1e6)]) # , int(3e6), int(1e7), int(3e7)
def bench_unique(jax_bench, N):
    """This test shows that jnp.unique needs to be avoided, especially on pairs
    it completely obliterates compile and run times when used several times
    """
    
    def f(n=1):
        ind = jax.random.randint(jax.random.key(0), (N,2), 0, 10)
        for i in range(n):
            uq = jnp.unique(ind, size=N, fill_value=-1, return_index=True, return_inverse=True, axis=0)
            ind = ind + uq[0]
        return uq
    f.jit = jax.jit(f, static_argnames="n")
    
    jb = jax_bench(jit_rounds=5, jit_loops=10, jit_warmup=1)
    jb.measure(fn=f, fn_jit=f.jit, n=2, tag="unique")

    jb.configure(jit_rounds=2, jit_loops=2, jit_warmup=1)
    jb.measure(fn=f, fn_jit=f.jit, n=3, tag="unique_n3")
    jb.measure(fn=f, fn_jit=f.jit, n=10, tag="unique_n10")

@pytest.mark.skip_in_quick
@pytest.mark.parametrize("N", [int(1e6), int(1e7)])
def bench_min_scatter(jax_bench, N):
    def f(mode=0):
        if mode == 0:
            ind = jax.random.randint(jax.random.key(0), N, 0, N)
        else:
            ind = jax.random.randint(jax.random.key(0), N, 0, 10)
        mask = ind < N//2
        return _masked_min_scatter(mask, jnp.arange(len(ind)), ind, jnp.arange(len(ind)))
    f.jit = jax.jit(f, static_argnums=0)
    
    jb = jax_bench(jit_rounds=5, jit_loops=10, jit_warmup=1)

    jb.measure(fn_jit=f.jit, mode=0, tag="few-collisions")
    jb.measure(fn_jit=f.jit, mode=1, tag="many-collisions")
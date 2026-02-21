import pytest
import jax
import jax.numpy as jnp
from dataclasses import replace
from jztree.tree import pos_zorder_sort
from jztree.knn import prepare_knn, prepare_knn_z, evaluate_knn, evaluate_knn_z, knn, knn_z

import importlib
has_jaxkd = (importlib.util.find_spec("jaxkd") is not None) and (importlib.util.find_spec("jaxkd_cuda") is not None)

@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("npart", [int(1e5), int(1e6), int(3e6)])
def bench_knn_steps(jax_bench, pos):
    k = 16

    jb = jax_bench(jit_rounds=40, jit_warmup=10)

    posz, idz = jb.measure(fn=pos_zorder_sort, fn_jit=pos_zorder_sort.jit, x=pos, tag="zsort")[1]

    data = jb.measure(fn_jit=prepare_knn.jit, pos0=pos, k=k, tag="prepare")[1]
    data2 = jb.measure(fn_jit=prepare_knn_z.jit, posz=posz, k=k, tag="prepare_z_new")[1]

    jb.measure(fn_jit=evaluate_knn.jit, d=data, tag="eval")
        
    jb.measure(fn_jit=knn.jit, pos0=pos, k=k, tag="total")
    jb.measure(fn_jit=knn_z.jit, posz=posz, k=k, tag="total_z")

    # query particles with a different seed
    pos_q = jax.random.uniform(jax.random.PRNGKey(1), (len(pos),3), dtype=jnp.float32)
    pos_qz = pos_zorder_sort.jit(pos_q)[0]

    jb.measure(fn_jit=evaluate_knn_z.jit, d=data2, posz_query=pos_qz, tag="eval_q_z")
    jb.measure(fn_jit=knn.jit, pos0=pos_q, k=k, pos_query=pos_q, tag="total_q_z")

@pytest.mark.shrink_in_quick(keep_index=3)
@pytest.mark.parametrize("k", [4,8,12,16,32,64])
def bench_knn_k(jax_bench, pos, k):
    jb = jax_bench(jit_rounds=40, jit_warmup=10)

    jb.measure(fn_jit=knn.jit, pos0=pos, k=k)

@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("N", [1e5,3e5,1e6,3e6,1e7])
def bench_knn_N(jax_bench, N):
    jb = jax_bench(jit_rounds=5, jit_loops=5, jit_warmup=1)

    pos = jax.random.uniform(jax.random.key(0), (int(N), 3), dtype=jnp.float32)

    jb.measure(fn_jit=knn.jit, pos0=pos, k=4, tag="jztree_k4")
    jb.measure(fn_jit=knn.jit, pos0=pos, k=16, tag="jztree_k16")

    if has_jaxkd:
        import jaxkd

        def f(pos, k):
            return jaxkd.build_and_query(pos, pos, k=k, cuda=True)
        f.jit = jax.jit(f, static_argnames=["k"])

        jb.measure(fn_jit=f.jit, pos=pos, k=4, tag="jaxkd_cuda_k4")
        jb.measure(fn_jit=f.jit, pos=pos, k=16, tag="jaxkd_cuda_k16")
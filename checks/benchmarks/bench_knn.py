import pytest
import jax
import jax.numpy as jnp
import jztree as jz
from dataclasses import replace
from jztree.tree import pos_zorder_sort

@pytest.mark.parametrize("npart", [int(1e5), int(1e6), int(3e6)])
def bench_knn_steps(jax_bench, pos):
    k = 16

    jb = jax_bench(jit_rounds=40, jit_warmup=10)

    posz, idz = jb.measure(fn=pos_zorder_sort, fn_jit=pos_zorder_sort.jit, x=pos, tag="zsort")[1]

    data = jb.measure(fn_jit=jz.knn.prepare_knn.jit, pos0=pos, k=k, tag="prepare")[1]
    data2 = jb.measure(fn_jit=jz.knn.prepare_knn_z.jit, posz=posz, k=k, tag="prepare_sorted")[1]

    jb.measure(fn_jit=jz.knn.evaluate_knn.jit, d=data, tag="evaluate")
        
    jb.measure(fn_jit=jz.knn.knn.jit, pos0=pos, k=k, tag="total_unsorted")
    jb.measure(fn_jit=jz.knn.knn_z.jit, posz=posz, k=k, tag="total_sorted")
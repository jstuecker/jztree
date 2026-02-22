import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType

from jztree_utils import ics
from jztree.config import KNNConfig
from jztree import knn

def get_mesh(ndev=-1):
    return jax.make_mesh((ndev,), ('gpus',), axis_types=(AxisType.Auto))

@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("N", (int(1e6), int(1e7), int(1e8)))
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_distr_knn(jax_bench, N):
    cfg = KNNConfig()
    cfg.tree.alloc_fac_nodes = 2.0

    ndev = jax.device_count()

    jb = jax_bench(jit_rounds=4, jit_warmup=1)

    mesh = get_mesh(4)

    part = ics.uniform_particles.smap(mesh, jit=True)(N, npad=int(N * 0.4))

    rnn = jb.measure(fn_jit=knn.distr_knn.smap(mesh, jit=True),
        part=part, k=16, output_order="input", result="rad", tag=f"ndev{ndev}"
    )

@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_distr_knn_steps(jax_bench):
    N = int(1e7)

    cfg = KNNConfig()
    cfg.tree.alloc_fac_nodes = 2.0

    ndev = jax.device_count()

    jb = jax_bench(jit_rounds=4, jit_warmup=1)

    mesh = get_mesh(4)

    part = ics.uniform_particles.smap(mesh, jit=True)(N, npad=int(N * 0.4))

    partz, th = jb.measure(fn_jit=knn.distr_zsort_and_tree.smap(mesh, jit=True),
        part=part, cfg_tree=cfg.tree, tag="tree_ndev{ndev}"
    )[1]

    ilist = jb.measure(fn_jit=knn._knn_dual_walk.smap(mesh, jit=True),
        th=th, k=16, alloc_fac_ilist=cfg.alloc_fac_ilist, tag="treewalk_ndev{ndev}"
    )

    rnn = jb.measure(fn_jit=knn.distr_knn.smap(mesh, jit=True),
        part=partz, k=16, output_order="z", result="rad", tag=f"totalz_ndev{ndev}"
    )
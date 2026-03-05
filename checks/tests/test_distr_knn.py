import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType

from jztree_utils import ics
from jztree.config import KNNConfig
from jztree import knn
from jztree.data import squeeze_any, squeeze_particles

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))

@pytest.mark.shrink_in_quick
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def test_distr_knn():
    cfg = KNNConfig()
    cfg.tree.alloc_fac_nodes = 2.0

    part = ics.uniform_particles.smap(mesh, jit=True)(1000000, npad=400000)

    rnn, inn = knn.knn.smap(mesh, jit=True)(
        part, k=4, output_order="input", result="rad_globalidx"
    )
    rnn, inn = squeeze_any((rnn, inn), rnn.shape[1], part.num, part.num_total)

    part_fl = squeeze_particles(part)
    rnn_ref, inn_ref = knn.knn.jit(part_fl.pos, k=4)

    assert jnp.all(inn == inn_ref)
    assert jnp.allclose(rnn, rnn_ref, rtol=1e-6, atol=1e-6)
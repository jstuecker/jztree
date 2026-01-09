import fmdj
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from fmdj.comm import get_rank_info

from jztree.fof import Link, link_distributed, Label, insert_links

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))
sharding = NamedSharding(mesh, P('gpus'))

@pytest.mark.parametrize("nperdev", (16, 256, 5012))
def test_distributed_links(nperdev):
    """Define a random distributed link graph and check against local implementation"""

    ndev = len(jax.devices())
    
    iA = jnp.arange(ndev*nperdev)
    irand = jax.random.randint(jax.random.key(0), len(iA), minval=0, maxval=nperdev//2)
    iB = jnp.maximum(iA - irand, 0)

    links = Link(
        Label(iA // nperdev, iA % nperdev),
        Label(iB // nperdev, iB % nperdev)
    )

    @jax.jit
    @jax.shard_map(out_specs=P("gpus"), in_specs=P("gpus"), mesh=mesh)
    def my_distr_links(links: Link):
        labels = links.a
        igroup = links.a.igroup

        rank, ndev, axis_name = get_rank_info()

        labels = link_distributed(
            igroup, labels, links, len(igroup), jax.lax.pvary(len(igroup), axis_name)
        )
        
        # convert back to global indices
        return labels.igroup + labels.irank * len(igroup)
    
    igroup_a = insert_links(iA, iA, iB, num_links=len(iA)) # fully local operation
    igroup_b = my_distr_links(links)[:ndev*nperdev] # distributed linking

    assert jnp.all(igroup_a == igroup_b)
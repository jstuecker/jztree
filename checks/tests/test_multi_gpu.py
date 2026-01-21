import fmdj
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from fmdj.comm import get_rank_info
import jztree
from fmdj_utils.ics import gaussian_blob
from fmdj.tools import cumsum_starting_with_zero, multi_to_dense

from jztree.fof import Link, link_distributed, Label, insert_links

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))
sharding = NamedSharding(mesh, P('gpus'))

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.shrink_in_quick(keep_index=2)
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

        dev_spl = (jnp.zeros(ndev+1, dtype=jnp.int32)).at[rank+1].set(len(igroup))

        labels = link_distributed(
            igroup, labels, links, dev_spl, len(igroup) + rank*0,
        )
        
        # convert back to global indices
        return labels.igroup + labels.irank * len(igroup)
    
    igroup_a = insert_links(iA, iA, iB, num_links=len(iA)) # fully local operation
    igroup_b = my_distr_links(links)[:ndev*nperdev] # distributed linking

    assert jnp.all(igroup_a == igroup_b)

def particles_and_tree(seed=0):
    cfg = jztree.data.FofConfig()
    rank, ndev, axis_name = get_rank_info()
    npart_tot = 1024*1024*ndev

    part = gaussian_blob(1024*1024, npad=1024*128*3, seed=rank+seed)

    return fmdj.ztree.distr_zsort_and_tree(part, npart_tot, cfg.tree)

@jax.jit
@jax.shard_map(out_specs=P("gpus"), in_specs=P(), mesh=mesh)
def distr_fof(seed):
    rank, ndev, axis_name = get_rank_info()
    partz, th = particles_and_tree(seed)
    igroup = jztree.fof.distr_fof_z_with_tree(partz.pos, th, rlink=0.1, linearize_labels=True)

    num = jax.lax.all_gather(jnp.sum(~jnp.isnan(partz.pos[...,0]), axis=0), axis_name)
    dspl = cumsum_starting_with_zero(num)

    return partz, igroup.reshape(1,-1), dspl.reshape(1,-1)

@pytest.mark.shrink_in_quick
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.parametrize("seed", [0,17,23,99])
def test_distr_fof(seed):
    partz, igroup1, dev_spl = distr_fof(seed)
    # combine arrays into one:
    igroup1 = multi_to_dense.jit(igroup1, dev_spl[0])

    partz = fmdj.ztree.pos_zorder_sort.jit(partz)[0]
    igroup2 = jztree.fof.fof_z.jit(partz.pos, rlink=0.1)

    assert igroup1 == pytest.approx(igroup2, abs=0.1)
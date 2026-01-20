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

def particles_and_tree(seed=0):
    cfg = jztree.data.FofConfig()
    rank, ndev, axis_name = get_rank_info()
    npart_tot = 1024*1024*ndev

    part = gaussian_blob(1024*1024, npad=1024*128*3, seed=rank+seed)

    return fmdj.ztree.distr_zsort_and_tree(part, npart_tot, cfg.tree)

@jax.jit
@jax.shard_map(in_specs=P(), out_specs=P("gpus"), mesh=mesh)
def node_fof(seed=0):
    rank, ndev, axis_name = get_rank_info()
    partz, th = particles_and_tree(seed)

    node_data, ilist = jztree.fof.distr_node_node_fof.jit(th, rlink=0.8)
    labels, spl = node_data.label, node_data.spl
    nnodes = jnp.argmax(spl)

    num = jax.lax.all_gather(nnodes, axis_name)
    dspl = cumsum_starting_with_zero(num)

    igroup = dspl[labels.irank] + labels.igroup

    return partz, igroup.reshape(1,-1), ilist.ispl[-1].reshape(1), dspl.reshape(1,-1)

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.parametrize("seed", [0,11,77,133,1337])
def test_distr_node_node_fof(seed):
    partz, igroup1, numint1, dev_spl = node_fof(seed)
    # combine arrays into one:
    igroup1 = multi_to_dense.jit(igroup1, dev_spl[0])
    numint1 = jnp.sum(numint1)

    cfg = jztree.data.FofConfig()
    partz = fmdj.ztree.pos_zorder_sort(partz)[0]
    th = fmdj.ztree.build_tree_hierarchy.jit(partz, cfg.tree, npart_tot=int(jnp.sum(~jnp.isnan(partz.pos[...,0]))))
    node_data, ilist2 = jztree.fof.node_node_fof.jit(th, rlink=0.8)

    assert igroup1[:len(node_data.label)] == pytest.approx(node_data.label, abs=0.1)
    # So far, there will be more interactions in the multi-gpu case, because the pruning based
    # on identical ids is less efficient, since the global true ids are not perfectly known at that
    # point. For now, just check that we have at least as many interactions:
    assert numint1 >= ilist2.ispl[-1]

@jax.jit
@jax.shard_map(out_specs=P("gpus"), in_specs=P(), mesh=mesh)
def distr_fof(seed):
    rank, ndev, axis_name = get_rank_info()
    partz, th = particles_and_tree(seed)
    igroup = jztree.fof.distr_fof_z_with_tree(partz.pos, th, rlink=0.1, linearize_labels=True)

    num = jax.lax.all_gather(jnp.sum(~jnp.isnan(partz.pos[...,0]), axis=0), axis_name)
    dspl = cumsum_starting_with_zero(num)

    return partz, igroup.reshape(1,-1), dspl.reshape(1,-1)

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.parametrize("seed", [0,17,23,99])
def test_distr_fof(seed):
    partz, igroup1, dev_spl = distr_fof(seed)
    # combine arrays into one:
    igroup1 = multi_to_dense.jit(igroup1, dev_spl[0])

    partz = fmdj.ztree.pos_zorder_sort.jit(partz)[0]
    igroup2 = jztree.fof.fof_z.jit(partz.pos, rlink=0.1)

    assert igroup1 == pytest.approx(igroup2, abs=0.1)
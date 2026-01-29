import pytest
import jax
import jax.numpy as jnp
from dataclasses import replace
from jztree.config import FofConfig
from jztree.tree import distr_zsort_and_tree
from jztree.fof import distr_fof_z_with_tree, distr_node_node_fof, distr_particle_particle_fof
from jztree.comm import get_rank_info
from fmdj_utils.ics import gaussian_blob
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
import numpy as np

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))

def particles_and_tree(N=int(1e6), seed=0):
    cfg = FofConfig()
    rank, ndev, axis_name = get_rank_info()

    part = gaussian_blob(N, npad=N//2, seed=rank+seed)

    return distr_zsort_and_tree(part, cfg.tree)

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("N", [int(1e6), int(1e7), int(3e7), int(1e8), int(3e8)])
def bench_fof(jax_bench, pos, N):

    jb = jax_bench(jit_rounds=2, jit_warmup=1)

    rlink = 0.5*np.cbrt(1./ (N*jax.device_count()))

    @jax.shard_map(in_specs=P("gpus"), out_specs=P("gpus"), mesh=mesh)
    def fof():
        partz, th = particles_and_tree(N)

        labels = distr_fof_z_with_tree(partz.pos, th, rlink=rlink)
        
        return labels
    
    jb.measure(fn_jit=jax.jit(fof), tag="fof")

def smap_jit(f, **kwargs):
    fsm = jax.shard_map(lambda: f(**kwargs), in_specs=P(), out_specs=P("gpus"), mesh=mesh)
    return jax.jit(fsm) # the lambda helps with passing x as keyword argument

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("N", [int(1e6), int(1e7), int(3e7)])
def bench_fof_steps(jax_bench, pos, N):
    jb = jax_bench(jit_rounds=4, jit_warmup=1)

    rlink = 0.5*np.cbrt(1./ (N*jax.device_count()))    

    partz, th = jb.measure(fn_jit=smap_jit(lambda: particles_and_tree(N=N)), tag="ztree")[1]

    cfg = FofConfig()

    @jax.jit
    @jax.shard_map(in_specs=P("gpus"), out_specs=P("gpus"), mesh=mesh)
    def node_node_fof(th):
        return distr_node_node_fof(
            th, rlink=rlink, boxsize=0., alloc_fac_ilist=cfg.alloc_fac_ilist, size_links=len(partz.pos)
        )
    node_data, ilist, link_data = jb.measure(None, node_node_fof, th, tag="node-node")[1]

    @jax.jit
    @jax.shard_map(in_specs=P("gpus"), out_specs=P("gpus"), mesh=mesh)
    def particle_fof(node_data, ilist, link_data):
        return distr_particle_particle_fof(
            node_data, ilist, link_data, partz.pos, rlink=rlink, boxsize=0.
        )

    labels = jb.measure(None, particle_fof, node_data, ilist, link_data, tag="part-part")[1]

    @jax.jit
    @jax.shard_map(in_specs=P("gpus"), out_specs=P("gpus"), mesh=mesh)
    def fof():
        partz, th = particles_and_tree(N)

        labels = distr_fof_z_with_tree(partz.pos, th, rlink=rlink)
        
        return labels

    jb.measure(None, fof, tag="total")
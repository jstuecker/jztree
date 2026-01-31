import pytest
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
import numpy as np

from jztree.config import FofConfig
from jztree.tree import distr_zsort_and_tree
from jztree.fof import distr_fof_z_with_tree, distr_node_node_fof, distr_particle_particle_fof
from jztree.jax_ext import get_rank_info, shard_map_constructor
from jztree_utils import ics

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))

def particles_and_tree(N=int(1e6), seed=0):
    part = ics.gaussian_particles(N, npad=int(N*0.5), seed=seed)

    return distr_zsort_and_tree(part, FofConfig().tree)
particles_and_tree.smap = shard_map_constructor(particles_and_tree,
    out_specs=P(-1), in_specs=(None, None), static_argnums=(0,)
)

def run_fof(N, rlink=0.1):
    partz, th = particles_and_tree(N)

    return distr_fof_z_with_tree(partz.pos, th, rlink=rlink)
run_fof.smap = shard_map_constructor(run_fof, in_specs=(None, None), static_argnums=(0,1))

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("N", [int(1e6), int(1e7), int(3e7), int(1e8), int(3e8)])
def bench_fof(jax_bench, pos, N):

    jb = jax_bench(jit_rounds=4, jit_warmup=1)

    rlink = 0.5*np.cbrt(1./ (N*jax.device_count()))

    jb.measure(fn_jit=run_fof.smap(mesh, jit=True), N=N, rlink=rlink, tag="fof")

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("N", [int(1e6), int(1e7), int(3e7)])
def bench_fof_steps(jax_bench, pos, N):
    jb = jax_bench(jit_rounds=4, jit_warmup=1)

    rlink = 0.5*np.cbrt(1./ (N*jax.device_count()))

    partz, th = jb.measure(None, particles_and_tree.smap(mesh, jit=True), N, tag="ztree")[1]

    def node_node_fof(th, partz):
        cfg = FofConfig()
        return distr_node_node_fof(
            th, rlink=rlink, boxsize=0., alloc_fac_ilist=cfg.alloc_fac_ilist, size_links=len(partz.pos)
        )
    node_node_fof = shard_map_constructor(node_node_fof)(mesh, jit=True)
    node_data, ilist, link_data = jb.measure(None, node_node_fof, th, partz, tag="node-node")[1]

    def particle_fof(node_data, ilist, link_data, partz):
        return distr_particle_particle_fof(
            node_data, ilist, link_data, partz.pos, rlink=rlink, boxsize=0.
        )
    particle_fof = shard_map_constructor(particle_fof)(mesh, jit=True)
    labels = jb.measure(None, particle_fof, node_data, ilist, link_data, partz, tag="part-part")[1]

    jb.measure(fn_jit=run_fof.smap(mesh, jit=True), N=N, rlink=rlink, tag="total")
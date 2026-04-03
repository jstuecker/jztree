import pytest
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
import numpy as np

from jztree.config import FofConfig
from jztree.tree import zsort_and_tree
from jztree.fof import distr_fof_labels_z_with_tree, _distr_fof_dual_walk, _distr_fof_leaf2leaf
from jztree.fof import _fof_catalogue_from_groups, _distr_fof_order, distr_fof_and_catalogue
from jztree.jax_ext import get_rank_info, shard_map_constructor
from jztree_utils import ics

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))

def particles_and_tree(N=int(1e6), seed=0):
    part = ics.gaussian_particles(N, npad=int(N*0.5), seed=seed)
    partz, th = zsort_and_tree(part, FofConfig().tree)
    return part, partz, th
particles_and_tree.smap = shard_map_constructor(particles_and_tree,
    out_specs=P(-1), in_specs=(None, None), static_argnums=(0,)
)

def run_fof(N, rlink=0.1):
    part, partz, th = particles_and_tree(N)

    return distr_fof_labels_z_with_tree(partz.pos, th, rlink=rlink)
run_fof.smap = shard_map_constructor(run_fof, in_specs=(None, None), static_argnums=(0,1))

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("N", [int(1e6), int(1e7), int(3e7), int(1e8), int(3e8)])
def bench_fof(jax_bench, pos, N):
    ndev = jax.device_count()

    jb = jax_bench(jit_rounds=4, jit_warmup=1)

    rlink = 0.5*np.cbrt(1./ (N*jax.device_count()))

    jb.measure(fn_jit=run_fof.smap(mesh, jit=True), N=N, rlink=rlink, tag=f"fof_{ndev}")

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("ndev", [i for i in (256,128,64,32,16,8,4,2,1) if i <= jax.device_count()])
def bench_fof_steps(jax_bench, pos, ndev):
    jb = jax_bench(jit_rounds=4, jit_warmup=1)

    N = 512**3
    mesh = jax.make_mesh((ndev,), ("gpus",), AxisType.Explicit)
    rlink = 0.5*np.cbrt(1./ (N*ndev))

    part, partz, th = jb.measure(None, particles_and_tree.smap(mesh, jit=True), N, tag=f"ztree")[1]

    def node_node_fof(th, partz):
        cfg = FofConfig()
        return _distr_fof_dual_walk(
            th, rlink=rlink, boxsize=0., alloc_fac_ilist=cfg.alloc_fac_ilist, size_links=int(len(partz.pos)*0.5)
        )
    node_node_fof = shard_map_constructor(node_node_fof)(mesh, jit=True)
    node_data, ilist, link_data = jb.measure(None, node_node_fof, th, partz, tag=f"node-node")[1]

    def particle_fof(node_data, ilist, link_data, partz):
        return _distr_fof_leaf2leaf(
            node_data, ilist, link_data, partz.pos, rlink=rlink, boxsize=0.
        )
    particle_fof = shard_map_constructor(particle_fof)(mesh, jit=True)
    labels = jb.measure(None, particle_fof, node_data, ilist, link_data, partz, tag=f"part-part")[1]

    partf, counts = jb.measure(None, _distr_fof_order.smap(mesh, jit=True),
        labels, partz, tag=f"fof-order"
    )[1]

    cata = jb.measure(None, _fof_catalogue_from_groups.smap(mesh, jit=True),
        partf, counts, tag=f"catalogue"
    )[1]

    
    jb.measure(fn_jit=distr_fof_and_catalogue.smap(mesh, jit=True), part=part, rlink=rlink, tag=f"total")
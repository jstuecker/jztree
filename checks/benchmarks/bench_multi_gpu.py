import pytest
import jax
import jax.numpy as jnp
import jztree as jz
from dataclasses import replace
from jztree.tree import pos_zorder_sort
import fmdj
from fmdj.comm import get_rank_info
from fmdj_utils.ics import gaussian_blob
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))

def ftest():
    return jax.random.uniform(jax.random.key(0), 100)

@pytest.mark.parametrize("N", [int(1e6)])
def bench_smap_test(jax_bench, pos, N):

    jb = jax_bench(jit_rounds=2, jit_warmup=1)

    @jax.shard_map(in_specs=P("gpus"), out_specs=P("gpus"), mesh=mesh)
    def smapped():
        
        jb.measure(fn_jit=jax.jit(ftest), tag="shard_map_test")

    smapped()

def particles_and_tree(N, seed=0):
    cfg = jz.data.FofConfig()
    rank, ndev, axis_name = get_rank_info()
    npart_tot = N*ndev

    part = gaussian_blob(N, npad=N//2, seed=rank+seed)

    return fmdj.ztree.distr_zsort_and_tree(part, npart_tot, cfg.tree)

@pytest.mark.parametrize("N", [int(1e6), int(1e7), int(3e7), int(1e8)])
def bench_fof(jax_bench, pos, N):

    jb = jax_bench(jit_rounds=2, jit_warmup=1)

    @jax.shard_map(in_specs=P("gpus"), out_specs=P("gpus"), mesh=mesh)
    def fof():
        jb.measure(fn_jit=jax.jit(ftest), tag="tmp")

        partz, th = particles_and_tree(N)

        labels = jz.fof.distr_fof_z_with_tree(partz.pos, th, rlink=0.1)
        
        return labels
    
    jb.measure(fn_jit=jax.jit(fof), tag="fof")
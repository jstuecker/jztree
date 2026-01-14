import pytest
import jax
import jax.numpy as jnp
import jztree as jz
from dataclasses import replace
from jztree.tree import pos_zorder_sort
import fmdj

@pytest.mark.parametrize("N", [int(1e6), int(3e6)])
def bench_fof_steps(jax_bench, pos, N):
    jb = jax_bench(jit_rounds=5, jit_warmup=2)

    boxsize = 1.0
    rlink = 0.2 * boxsize / N**(1/3)
    pos = jax.random.uniform(jax.random.PRNGKey(0), (N, 3), minval=0.0, maxval=boxsize)

    posz, idz = jb.measure(fn=pos_zorder_sort, fn_jit=pos_zorder_sort.jit, x=pos, tag="zsort")[1]

    cfg = jz.fof.FofConfig()
    th = jb.measure(
        fn_jit=fmdj.ztree.build_tree_hierarchy.jit, 
        part=posz, cfg_tree=cfg.tree, tag="tree"
    )[1]

    node_data, ilist = jb.measure(fn_jit=jz.fof.node_node_fof.jit, 
        th=th, rlink=rlink, boxsize=boxsize, alloc_fac_ilist=cfg.alloc_fac_ilist, tag="node_node"
    )[1]
    jb.measure(fn_jit=jz.fof.particle_particle_fof.jit, 
        node_data=node_data, ilist=ilist, posz=posz, rlink=rlink, boxsize=boxsize,
        tag="particle_particle"
    )[1]

    jb.measure(fn_jit=jz.fof.fof.jit, pos=pos, rlink=rlink, boxsize=boxsize, tag="total")

@pytest.mark.parametrize("N", [int(1e5), int(3e5), int(1e6), int(3e6), int(1e7)])
def bench_fof_uniform(jax_bench, pos, N):
    jb = jax_bench(jit_rounds=40, jit_warmup=10)

    boxsize = 1.0
    rlink = 0.2 * boxsize / N**(1/3)
    pos = jax.random.uniform(jax.random.PRNGKey(0), (N, 3), minval=0.0, maxval=boxsize)

    jb.measure(fn_jit=jz.fof.fof.jit, pos=pos, rlink=rlink, boxsize=boxsize)

@pytest.mark.parametrize("ngrid", [16, 32, 64, 128, 256])
def bench_fof_cosmo(jax_bench, pos, ngrid):
    jb = jax_bench(jit_rounds=40, jit_warmup=10)

    from fmdj_utils.ics import discodj_sim
    pos = discodj_sim.jit(ngrid).pos

    boxsize = 100.
    rlink = 0.2 * boxsize / ngrid

    jb.measure(fn_jit=jz.fof.fof.jit, pos=pos, rlink=rlink, boxsize=boxsize)
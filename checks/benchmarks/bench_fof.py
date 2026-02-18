import pytest
import jax
import jax.numpy as jnp
from dataclasses import replace
from jztree.config import FofConfig
from jztree.tree import pos_zorder_sort, build_tree_hierarchy
from jztree.fof import node_node_fof, particle_particle_fof, fof_labels
import importlib
has_discodj = importlib.util.find_spec("discodj") is not None

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("N", [int(1e6), int(3e6)])
def bench_fof_steps(jax_bench, pos, N):
    jb = jax_bench(jit_rounds=5, jit_warmup=2)

    boxsize = 1.0
    rlink = 0.2 * boxsize / N**(1/3)
    pos = jax.random.uniform(jax.random.PRNGKey(0), (N, 3), minval=0.0, maxval=boxsize)

    posz, idz = jb.measure(fn=pos_zorder_sort, fn_jit=pos_zorder_sort.jit, x=pos, tag="zsort")[1]

    cfg = FofConfig()
    th = jb.measure(
        fn_jit=build_tree_hierarchy.jit, 
        part=posz, cfg_tree=cfg.tree, tag="tree"
    )[1]

    node_data, ilist = jb.measure(fn_jit=node_node_fof.jit, 
        th=th, rlink=rlink, boxsize=boxsize, alloc_fac_ilist=cfg.alloc_fac_ilist, tag="node_node"
    )[1]
    jb.measure(fn_jit=particle_particle_fof.jit, 
        node_data=node_data, ilist=ilist, posz=posz, rlink=rlink, boxsize=boxsize,
        tag="particle_particle"
    )[1]

    jb.measure(fn_jit=fof_labels.jit, pos=pos, rlink=rlink, boxsize=boxsize, tag="total")

@pytest.mark.shrink_in_quick(keep_index=2)
@pytest.mark.parametrize("N", [int(1e5), int(3e5), int(1e6), int(3e6), int(1e7)])
def bench_fof_uniform(jax_bench, pos, N):
    jb = jax_bench(jit_rounds=40, jit_warmup=10)

    boxsize = 1.0
    rlink = 0.2 * boxsize / N**(1/3)
    pos = jax.random.uniform(jax.random.PRNGKey(0), (N, 3), minval=0.0, maxval=boxsize)

    jb.measure(fn_jit=fof_labels.jit, pos=pos, rlink=rlink, boxsize=boxsize)

def dj_sim(ngrid, boxsize):
    from discodj import DiscoDJ

    dj = DiscoDJ(dim=3, res=ngrid, boxsize=boxsize)
    dj = dj.with_timetables()
    pkstate = dj.with_linear_ps()
    ics = dj.with_ics(pkstate, seed=0)
    lpt_state = dj.with_lpt(ics, n_order=1)
    sim_ini = dj.with_lpt_ics(lpt_state, n_order=1, a_ini=0.02)
    X, P, a = dj.run_nbody(
        sim_ini, a_end=1.0, n_steps=16, res_pm=ngrid, stepper="bullfrog"
    )

    return X, P, a
dj_sim.jit = jax.jit(dj_sim, static_argnames=("ngrid", "boxsize"))

@pytest.mark.shrink_in_quick(keep_index=3)
@pytest.mark.skipif(not has_discodj, reason="requires discodj module installed")
@pytest.mark.parametrize("ngrid", [16, 32, 64, 128, 256])
def bench_fof_cosmo(jax_bench, pos, ngrid):
    jb = jax_bench(jit_rounds=40, jit_warmup=10)
    
    boxsize = 100.
    pos = dj_sim.jit(ngrid, boxsize)[0].reshape(-1,3)

    cfg = FofConfig(alloc_fac_ilist=16)

    jb.measure(fn_jit=fof_labels.jit, pos=pos, rlink=0.2*boxsize/ngrid, boxsize=boxsize, cfg=cfg)
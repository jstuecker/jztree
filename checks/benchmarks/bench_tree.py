import pytest
from dataclasses import replace
from jztree.config import TreeConfig
from jztree.tree import build_tree_hierarchy, zsort
from jztree_utils import ics
import jax
import jax.numpy as jnp

@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("npart", [1024*128,1024*1024, 1024*1024*8])
def bench_build_tree_hierarchy(jax_bench, npart):
    pos_mass = ics.uniform_particles(npart)
    pos_mass_z = zsort(pos_mass)[0]

    jb = jax_bench(jit_rounds=10, jit_warmup=5, eager_rounds=0, eager_warmup=0)

    cfg_tree = TreeConfig(mass_centered=False)

    jb.measure(
        fn=build_tree_hierarchy, fn_jit=build_tree_hierarchy.jit, 
        part=pos_mass_z, cfg_tree=cfg_tree, tag="geom_centered")
    
    cfg_tree.mass_centered = True
    jb.measure(
        fn=build_tree_hierarchy, fn_jit=build_tree_hierarchy.jit, 
        part=pos_mass_z, cfg_tree=cfg_tree, tag="mass_centered")

@pytest.mark.shrink_in_quick(keep_index=2)
@pytest.mark.parametrize("npart", [1024*128,1024*1024, 1024*1024*8, 1024*1024*32])
def bench_zsort(jax_bench, npart):
    jb = jax_bench(jit_rounds=5, jit_loops=5, jit_warmup=1)

    def f():
        pos = ics.uniform_particles(npart)
        return zsort(pos)
    f.jit = jax.jit(f)

    jb.measure(fn_jit=f.jit, tag="zsort")
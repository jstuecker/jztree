import pytest
from dataclasses import replace
from jztree.config import TreeConfig
from jztree.tree import build_tree_hierarchy, pos_zorder_sort
from jztree_utils import ics
import jax
import jax.numpy as jnp

@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("npart", [1024*128,1024*1024, 1024*1024*8])
def bench_build_tree_hierarchy(jax_bench, npart):
    pos_mass = ics.uniform_particles(npart)
    pos_mass_z = pos_zorder_sort(pos_mass)[0]

    jb = jax_bench(jit_rounds=10, jit_warmup=5, eager_rounds=0, eager_warmup=0)

    cfg_tree = TreeConfig(mass_centered=False)

    jb.measure(
        fn=build_tree_hierarchy, fn_jit=build_tree_hierarchy.jit, 
        part=pos_mass_z, cfg_tree=cfg_tree, tag="geom_centered")
    
    cfg_tree.mass_centered = True
    jb.measure(
        fn=build_tree_hierarchy, fn_jit=build_tree_hierarchy.jit, 
        part=pos_mass_z, cfg_tree=cfg_tree, tag="mass_centered")

@pytest.mark.shrink_in_quick(keep_index=3)
@pytest.mark.parametrize("npart", [1024*128,1024*1024, 1024*1024*8, 1024*1024*32])
def bench_zsort(jax_bench, npart):
    pos_mass = ics.uniform_particles(npart)
    jb = jax_bench(jit_rounds=50, jit_warmup=5)

    jb.measure(fn_jit=pos_zorder_sort.jit, x=pos_mass.pos, radix=False, tag="zsort")

    jb.measure(fn_jit=pos_zorder_sort.jit, x=pos_mass.pos, radix=True, tag="zsort_radix")

    irand = jax.random.randint(jax.random.key(0), npart, 0, 128)
    @jax.jit
    def argsort(i):
        return jnp.argsort(i, stable=False)
    jb.measure(fn_jit=argsort, i=irand, tag="argsort")
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from jztree.jax_ext import get_rank_info, expanding_shard_map, shard_map_constructor
from jztree.config import TreeConfig
from jztree.data import Pos, PosMass, TreeHierarchy, squeeze_particles
from jztree.tree import zsort, distr_zsort, adjust_domain_for_nodesize
from jztree.tree import detect_leaf_boundaries, build_tree_hierarchy, zsort_and_tree
from jztree_utils import ics

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))
sharding = NamedSharding(mesh, P('gpus'))

ndev = len(jax.devices())

def _distr_zsort():
    rank, ndev, axis_name = get_rank_info()
    pos = jax.random.uniform(jax.random.key(rank), (int(4096+1024),3))
    pos = pos.at[-1024:].set(jnp.nan)

    part = Pos(pos=pos, num=4096, num_total=4096*ndev)

    partz = distr_zsort(part)[0]
    return part, partz
_distr_zsort.smapped = expanding_shard_map(_distr_zsort, jit=True, mesh=mesh)

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def test_mutli_zsort():
    part, partz = _distr_zsort.smapped()

    partz_mult = squeeze_particles(partz)
    partz_sing = zsort(squeeze_particles(part))[0]

    assert len(partz_sing.pos) == len(partz_mult.pos) == partz_sing.num == partz_mult.num
    assert partz_mult.pos == pytest.approx(partz_sing.pos, abs=1e-5)

def _distr_coarsen(partz: Pos):
    partz, dataz, lvl_bound = adjust_domain_for_nodesize(partz, 256)
    ispl = detect_leaf_boundaries(partz.pos, leaf_size=256, lvl_bound=lvl_bound, npart=partz.num)

    # Convert splits to global splits
    rank, ndev, axis_name = get_rank_info()
    neach = jax.lax.all_gather(ispl[-1], axis_name)
    offset = jnp.cumsum(neach)[rank] - ispl[-1]
    ispl = ispl + offset

    return partz, ispl
_distr_coarsen.smapped = expanding_shard_map(_distr_coarsen, mesh=mesh, jit=True)

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def test_multi_leaves():
    part, partz = _distr_zsort.smapped()
    partz_new, ispl_multi = _distr_coarsen.smapped(partz)
    
    assert jnp.all(squeeze_particles(partz).pos == squeeze_particles(partz_new).pos)

    ispl0 = detect_leaf_boundaries(squeeze_particles(partz).pos, leaf_size=256)

    # Have to remove duplicates, since splits at device boundaries appear twice
    def remove_duplicates(i):
        return i[jnp.where(i[1:] != i[:-1])]

    assert jnp.all(remove_duplicates(ispl_multi.flatten()) == remove_duplicates(ispl0))

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def test_tree_properties():
    # Builds a tree hierarchy once on a single device and once accross multiple devices
    # We check their consistency by performing a couple of reduction operations

    cfg_tree = TreeConfig()
    cfg_tree.mass_centered = True
    cfg_tree.alloc_fac_nodes = 2.0

    # Build a reference structure
    partref = ics.uniform_particles.smap(mesh, jit=True)(1024*128, npad=1024*32)
    partrefz = zsort(squeeze_particles(partref))[0]

    thref = build_tree_hierarchy(partrefz, cfg_tree)

    def reductions(th: TreeHierarchy, axis_name=None):
        # Do a couple of reductions on node-properties to check identity of tree structures
        # We use some squares to make it unlikely that the sum agrees if there are any differences
        # in the node definitions
        res = []
        for ispl in th.ispl_n2n, th.ispl_n2l:
            for n in range(th.ispl_n2n.nlevels()):
                res.append(jnp.sum((ispl.get(n)[1:] - ispl.get(n)[:-1])**2))

        lvlsum = jnp.sum(jnp.where(th.lvl.data > -1000, th.lvl.data, 0))
        csum = jnp.nansum(th.geom_cent.data**2)
        msum = jnp.nansum(th.mass.data**2)
        mcsum = jnp.nansum(th.mass_cent.data**2)
        res = res + [lvlsum, csum, msum, mcsum]

        if axis_name is None:
            return jnp.array(res)
        else:
            return jnp.array([jax.lax.psum(x, axis_name) for x in res])

    def distr_reductions(part):
        rank, ndev, axis_name = get_rank_info()
        partz, th = zsort_and_tree(part, cfg_tree)

        return reductions(th, axis_name)
    distr_reductions.smap = shard_map_constructor(distr_reductions, out_specs=P())
    
    results1 = reductions(thref)
    results2 = distr_reductions.smap(mesh, jit=True)(partref)

    assert results1[:-3] == pytest.approx(results2[:-3], rel=1e-7) # integer sums, be very strict
    assert results1[-3:] == pytest.approx(results2[-3:], rel=1e-6) # floating point sums
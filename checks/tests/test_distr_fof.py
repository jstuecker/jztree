import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from jztree.jax_ext import get_rank_info, expanding_shard_map, shard_map_constructor
from jztree.config import FofConfig
from jztree.tools import cumsum_starting_with_zero, multi_to_dense
from jztree.data import ParticleData, Link, Label, flatten_particles, pad_particles
from jztree.data import squeeze_particles, expand_particles, squeeze_catalogue, sort_catalogue
from jztree.tree import zsort_and_tree, zsort
from jztree.fof import _distr_link, _insert_links, distr_fof_labels, fof_labels
from jztree.fof import fof_and_catalogue, distr_fof_and_catalogue
from jztree_utils import ics
import importlib
has_discodj = importlib.util.find_spec("discodj") is not None

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))
sharding = NamedSharding(mesh, P('gpus'))

@pytest.mark.skipif((jax.device_count() <= 1) or (jax.device_count() > 4), reason="Only single node")
@pytest.mark.shrink_in_quick(keep_index=2)
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

        dev_spl = (jnp.zeros(ndev+1, dtype=jnp.int32)).at[rank+1].set(len(igroup))

        labels = _distr_link(
            igroup, labels, links, dev_spl, len(igroup) + rank*0,
        )
        
        # convert back to global indices
        return labels.igroup + labels.irank * len(igroup)
    
    igroup_a = _insert_links(iA, iA, iB, num_links=len(iA)) # fully local operation
    igroup_b = my_distr_links(links)[:ndev*nperdev] # distributed linking

    assert jnp.all(igroup_a == igroup_b)

def my_distr_fof_labels(seed):
    # almost everything gets linked so we need a bit larger allocation than usual:
    cfg = FofConfig(alloc_fac_distr_links=0.1)
    part = ics.gaussian_particles(1024*1024, npad=1024*256, seed=seed)
    partz, igroup = distr_fof_labels(part, rlink=0.03, linearize_labels=True, cfg=cfg)
    return partz, igroup
my_distr_fof_labels.smapped = expanding_shard_map(my_distr_fof_labels, 
    in_specs=P(), input_tiled=True, mesh=mesh, jit=True
)

@pytest.mark.shrink_in_quick
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.parametrize("seed", [3,17,23,99])
def test_labels_vs_single(seed):
    partz, igroup1 = my_distr_fof_labels.smapped(seed)
    dev_spl = cumsum_starting_with_zero(partz.num)
    # combine arrays into one:
    igroup1 = multi_to_dense(igroup1, dev_spl, out_size=dev_spl[-1])
    partz = squeeze_particles(partz)

    partz = zsort.jit(partz)[0]
    igroup2 = fof_labels.jit(partz.pos, rlink=0.03)

    assert igroup1 == pytest.approx(igroup2, abs=0.1)

@pytest.mark.shrink_in_quick
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.parametrize("seed", [0,17,23,99])
def test_catalogue_vs_single(seed):
    # almost everything gets linked so we need a bit larger allocation than usual:
    cfg = FofConfig(alloc_fac_distr_links=0.1)
    part = ics.gaussian_particles.smap(mesh, jit=True)(1024*1024, npad=1024*256, seed=seed)
    partf, cata1 = distr_fof_and_catalogue.smap(mesh, jit=True)(part, rlink=0.05, cfg=cfg)

    p2, cata2 = fof_and_catalogue.jit(squeeze_particles(partf), rlink=0.05)
    
    # Sort by masses and equalize shapes
    assert np.sum(cata1.ngroups) == np.sum(cata2.ngroups)

    cata1 = sort_catalogue(squeeze_catalogue(cata1, offset_mode="global", nparts=partf.num))
    cata2 = sort_catalogue(squeeze_catalogue(cata2))

    assert jnp.all(cata1.count == cata2.count)
    assert jnp.all(cata1.offset.astype(jnp.int32) == cata2.offset)
    assert cata1.mass == pytest.approx(cata2.mass, abs=0.1)
    assert cata1.com_pos == pytest.approx(cata2.com_pos, abs=0.01)
    assert cata1.com_inertia_radius == pytest.approx(cata2.com_inertia_radius, abs=0.01)

@pytest.mark.skip_in_quick
@pytest.mark.skipif(not has_discodj, reason="requires discodj module installed")
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def test_discodj_fof():
    mesh = jax.make_mesh((jax.device_count(),), ("gpus",), (AxisType.Explicit,))
    ndev = jax.device_count()
    boxsize = 1000.

    part = ics.multi_gpu_dj_sim.jit(num_per_device=128**3)
    part = expand_particles(part, ndev)

    rlink = 0.2 * boxsize / np.cbrt(part.num_total)

    def distr_fof(part: ParticleData):
        ndev = jax.lax.axis_size("gpus")

        part = pad_particles(part, int(part.num_total // ndev * 0.5))
        cfg = FofConfig()
        cfg.tree.alloc_fac_nodes = 1.2
        part_fof, cata = distr_fof_and_catalogue(part, rlink=rlink, boxsize=1000., cfg=cfg)
        return part_fof, cata
    distr_fof = expanding_shard_map(distr_fof, mesh=mesh, jit=True)
    
    sharding = jax.sharding.NamedSharding(mesh, P())
    cata = jax.device_put(distr_fof(part)[1], sharding)

    cata = sort_catalogue(squeeze_catalogue(cata))

    print(cata.mass[0:20])

    assert (jnp.max(cata.mass) >= 1e15) & (jnp.max(cata.mass) <= 1e17)
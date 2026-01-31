import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from jztree.comm import get_rank_info, expanding_shard_map
from fmdj_utils.ics import gaussian_blob
from jztree.config import FofConfig
from jztree.tools import cumsum_starting_with_zero, multi_to_dense
from jztree.data import ParticleData, Link, Label, flatten_particles, pad_particles
from jztree.data import squeeze_particles, expand_particles, squeeze_catalogue, sort_catalogue
from jztree.tree import distr_zsort_and_tree, pos_zorder_sort
from jztree.fof import link_distributed, insert_links, distr_fof_z_with_tree, fof_labels_z
from jztree.fof import fof_and_catalogue, distr_fof_and_catalogue
from jztree.tools import tree_map_by_len
import importlib
has_discodj = importlib.util.find_spec("discodj") is not None

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))
sharding = NamedSharding(mesh, P('gpus'))

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
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

        labels = link_distributed(
            igroup, labels, links, dev_spl, len(igroup) + rank*0,
        )
        
        # convert back to global indices
        return labels.igroup + labels.irank * len(igroup)
    
    igroup_a = insert_links(iA, iA, iB, num_links=len(iA)) # fully local operation
    igroup_b = my_distr_links(links)[:ndev*nperdev] # distributed linking

    assert jnp.all(igroup_a == igroup_b)

def particles(seed=0):
    rank, ndev, axis_name = get_rank_info()
    part = gaussian_blob(1024*1024, npad=1024*128*3, seed=rank+seed)
    part.mass = 1.
    return part

def particles_and_tree(seed=0):
    cfg = FofConfig()

    return distr_zsort_and_tree(particles(seed), cfg.tree)

@jax.jit
def distr_fof(seed):
    rank, ndev, axis_name = get_rank_info()
    partz, th = particles_and_tree(seed)
    igroup = distr_fof_z_with_tree(partz.pos, th, rlink=0.1, linearize_labels=True)

    num = jax.lax.all_gather(jnp.sum(~jnp.isnan(partz.pos[...,0]), axis=0), axis_name)
    dspl = cumsum_starting_with_zero(num)

    return partz, igroup.reshape(1,-1), dspl.reshape(1,-1)
distr_fof.smapped = expanding_shard_map(distr_fof, in_specs=P(), input_tiled=True, mesh=mesh, jit=True)

@pytest.mark.shrink_in_quick
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.parametrize("seed", [0,17,23,99])
def test_distr_fof(seed):
    partz, igroup1, dev_spl = distr_fof.smapped(seed)
    # combine arrays into one:
    igroup1 = multi_to_dense.jit(igroup1, dev_spl[0])

    partz = pos_zorder_sort.jit(partz)[0]
    igroup2 = fof_labels_z.jit(partz.pos, rlink=0.1)

    assert igroup1 == pytest.approx(igroup2, abs=0.1)

# @jax.jit
# @jax.shard_map(out_specs=P("gpus"), in_specs=P(), mesh=mesh)
def distr_fof_cata(seed):
    part = particles(seed)
    return distr_fof_and_catalogue(part, rlink=0.05, cfg=FofConfig())
distr_fof_cata.smapped = expanding_shard_map(distr_fof_cata, input_tiled=True, mesh=mesh, in_specs=P())

@jax.jit
def fof_cata(part):
    return fof_and_catalogue(part, rlink=0.05, cfg=FofConfig())

@pytest.mark.shrink_in_quick
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.parametrize("seed", [0,17,23,99])
def test_catalogue_vs_single(seed):
    partf, cata1 = jax.jit(distr_fof_cata.smapped)(seed)
    p2, cata2 = fof_cata(squeeze_particles(partf))
    
    # Sort by masses and equalize shapes
    assert np.sum(cata1.ngroups) == np.sum(cata2.ngroups)

    cata1 = sort_catalogue(squeeze_catalogue(cata1, offset_mode="global", nparts=partf.num))
    cata2 = sort_catalogue(squeeze_catalogue(cata2))

    assert jnp.all(cata1.count == cata2.count)
    assert jnp.all(cata1.offset == cata2.offset)
    assert cata1.mass == pytest.approx(cata2.mass, abs=0.1)
    assert cata1.com_pos == pytest.approx(cata2.com_pos, abs=0.01)
    assert cata1.com_inertia_radius == pytest.approx(cata2.com_inertia_radius, abs=0.01)

def _particle_mass(omega_m: float, boxsize: float, npart: int) -> float:
    G = 43.007105731706317
    Hubble = 100.0
    return 1e10 * omega_m * 3 * Hubble * Hubble / (8 * np.pi * G) * boxsize ** 3 / npart

def fistr_dj_sim() -> ParticleData:
    from discodj import DiscoDJ
    from discodj.core.scatter_and_gather import ScatterGatherProperties

    ndev = jax.device_count()

    nres = np.int64(((np.cbrt(512**3 * ndev))//ndev)*ndev)

    print(f"total grid dim {nres}, particles per GPU {np.cbrt(nres**3/ndev):.2f}**3")
    
    scat = ScatterGatherProperties(
        res=nres,
        res_pm=nres,
        num_devices=ndev,
        use_distributed_scatter_gather=True,
        use_vjp_gather=False,
        use_vjp_scatter=False,
        scatter_gather_check=False
    )

    boxsize = 1000.

    dj = DiscoDJ(dim=3, res=scat.res, boxsize=boxsize)
    dj = dj.with_timetables()
    pkstate = dj.with_linear_ps()
    ics = dj.with_ics(pkstate, seed=0)
    lpt_state = dj.with_lpt(ics, n_order=1)
    sim_ini = dj.with_lpt_ics(lpt_state, n_order=1, a_ini=0.02)
    X, P, a = dj.run_nbody(
        sim_ini, a_end=1.0, n_steps=16, res_pm=scat.res_pm, stepper="bullfrog",
        scatter_gather_props=scat
    )

    part = ParticleData(
        pos=X.reshape(-1,3),
        mass=_particle_mass(0.3, boxsize, nres**3),
        vel=P.reshape(-1,3),
        num_total=nres**3
    )

    return part

@pytest.mark.skipif(not has_discodj, reason="requires discodj module installed")
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def test_discodj_fof():
    mesh = jax.make_mesh((jax.device_count(),), ("gpus",))
    ndev = jax.device_count()
    boxsize = 1000.

    part = jax.jit(fistr_dj_sim)()
    part = expand_particles(part, ndev)

    rlink = 0.2 * boxsize / np.cbrt(part.num_total)

    def distr_fof(part: ParticleData):
        rank = jax.lax.axis_index("gpus")
        ndev = jax.lax.axis_size("gpus")

        part = pad_particles(part, int(part.num_total // ndev * 0.5))
        
        part_fof, cata = distr_fof_and_catalogue(part, rlink=rlink, boxsize=1000.)
        return part_fof, cata
    distr_fof = expanding_shard_map(distr_fof, jit=True, mesh=mesh)
       
    part_fof, cata = distr_fof(part)

    masses = cata.mass

    print("Masses:")
    masses = jax.device_put(masses[:,0:20].flatten(), jax.NamedSharding(mesh, P()))
    print(masses)

    assert (jnp.max(masses) >= 1e15) & (jnp.max(masses) <= 1e17)
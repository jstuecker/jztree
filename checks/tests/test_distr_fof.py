import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from jztree.comm import get_rank_info
from fmdj_utils.ics import gaussian_blob
from jztree.config import FofConfig
from jztree.tools import cumsum_starting_with_zero, multi_to_dense
from jztree.data import ParticleData, Link, Label
from jztree.tree import distr_zsort_and_tree, pos_zorder_sort
from jztree.fof import link_distributed, insert_links, distr_fof_z_with_tree, fof_labels_z
from jztree.fof import distr_fof_order, fof_catalogue_from_groups, fof_order
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

def particles_and_tree(seed=0):
    cfg = FofConfig()
    rank, ndev, axis_name = get_rank_info()
    npart_tot = 1024*1024*ndev

    part = gaussian_blob(1024*1024, npad=1024*128*3, seed=rank+seed)

    return distr_zsort_and_tree(part, cfg.tree)

@jax.jit
@jax.shard_map(out_specs=P("gpus"), in_specs=P(), mesh=mesh)
def distr_fof(seed):
    rank, ndev, axis_name = get_rank_info()
    partz, th = particles_and_tree(seed)
    igroup = distr_fof_z_with_tree(partz.pos, th, rlink=0.1, linearize_labels=True)

    num = jax.lax.all_gather(jnp.sum(~jnp.isnan(partz.pos[...,0]), axis=0), axis_name)
    dspl = cumsum_starting_with_zero(num)

    return partz, igroup.reshape(1,-1), dspl.reshape(1,-1)

@pytest.mark.shrink_in_quick
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.parametrize("seed", [0,17,23,99])
def test_distr_fof(seed):
    partz, igroup1, dev_spl = distr_fof(seed)
    # combine arrays into one:
    igroup1 = multi_to_dense.jit(igroup1, dev_spl[0])

    partz = pos_zorder_sort.jit(partz)[0]
    igroup2 = fof_labels_z.jit(partz.pos, rlink=0.1)

    assert igroup1 == pytest.approx(igroup2, abs=0.1)

@jax.jit
@jax.shard_map(out_specs=P("gpus"), in_specs=P(), mesh=mesh)
def distr_fof_cata(seed):
    rank, ndev, axis_name = get_rank_info()
    partz, th = particles_and_tree(seed)
    label = distr_fof_z_with_tree(partz.pos, th, rlink=0.1)
    
    part_fof, counts = distr_fof_order(label, partz)
    
    return partz, fof_catalogue_from_groups(part_fof, counts)

@jax.jit
def fof_cata(part):
    partz = pos_zorder_sort(part)[0]
    igroup = fof_labels_z(partz.pos, rlink=0.1)
    part, counts = fof_order(igroup, partz)
    return fof_catalogue_from_groups(part, counts)

@pytest.mark.shrink_in_quick
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.parametrize("seed", [0,17,23,99])
def test_distr_catalogue(seed):
    partz, cata1 = distr_fof_cata(seed)
    cata2 = fof_cata(partz)
    
    # Sort by masses and equalize shapes
    assert np.sum(cata1.ngroups) == np.sum(cata2.ngroups)
    num = np.sum(cata1.ngroups)
    isort1 = jnp.argsort(cata1.count, descending=True)
    isort2 = jnp.argsort(cata2.count, descending=True)
    cata1 = tree_map_by_len(lambda x: x[isort1][:num], cata1, len(cata1.count))
    cata2 = tree_map_by_len(lambda x: x[isort2][:num], cata2, len(cata2.count))

    assert jnp.all(cata1.count == cata2.count)
    assert cata1.mass == pytest.approx(cata2.mass, abs=0.1)
    assert cata1.com_pos == pytest.approx(cata2.com_pos, abs=0.01)
    assert cata1.com_inertia_radius == pytest.approx(cata2.com_inertia_radius, abs=0.01)

def _particle_mass(omega_m: float, boxsize: float, npart: int) -> float:
    G = 43.007105731706317
    Hubble = 100.0
    return 1e10 * omega_m * 3 * Hubble * Hubble / (8 * np.pi * G) * boxsize ** 3 / npart

def fistr_dj_sim():
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

    return X, P, a

@pytest.mark.skipif(not has_discodj, reason="requires discodj module installed")
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def test_discodj_fof():
    mesh = jax.make_mesh((jax.device_count(),), ("gpus",))
    ndev = jax.device_count()
    boxsize = 1000.

    pos, vel, a = jax.jit(fistr_dj_sim)()
    pos, vel = pos.reshape(-1,3), vel.reshape(-1,3)
    npart_tot = len(pos)
    nper_dev = npart_tot // ndev

    part = ParticleData(pos=pos, vel=vel)
    
    rlink = 0.2 * boxsize / np.cbrt(npart_tot)

    @jax.jit
    @jax.shard_map(out_specs=P("gpus"), in_specs=P("gpus"), mesh=mesh)
    def distr_fof(part):
        rank = jax.lax.axis_index("gpus")
        ndev = jax.lax.axis_size("gpus")

        part = ParticleData(
            pos=jnp.pad(part.pos, ((0,np.int32(nper_dev * 0.5)), (0,0)), constant_values=jnp.nan),
            vel=jnp.pad(part.vel, ((0,np.int32(nper_dev * 0.5)), (0,0)), constant_values=jnp.nan),
            num=nper_dev, num_total=ndev*nper_dev
        )

        cfg = FofConfig()
        cfg.tree.alloc_fac_nodes = 1.2

        partz, th = distr_zsort_and_tree(part, cfg.tree)
        labels = distr_fof_z_with_tree(partz.pos, th, rlink=rlink)

        # Warning!
        # This Reduction looses particles lying on other tasks
        # Fix this later!
        idx = jnp.where(labels.irank == rank, labels.igroup, len(labels.igroup))
        counts = jnp.zeros(len(labels.igroup)).at[idx].add(1)
        counts = jnp.sort(counts, descending=True)
        
        return partz, labels, counts
       
    partz, labels, counts = distr_fof(part)

    masses = counts.reshape(ndev,-1) * _particle_mass(0.3, 1000., npart_tot)

    print("Masses:")
    masses = jax.device_put(masses[:,0:20].flatten(), jax.NamedSharding(mesh, P()))
    print(masses)

    assert (jnp.max(masses) >= 1e15) & (jnp.max(masses) <= 1e17)
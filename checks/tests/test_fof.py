import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import importlib
from pathlib import Path
from jztree.config import FofConfig
from jztree.data import ParticleData, FofCatalogue, squeeze_catalogue, sort_catalogue
from jztree.tree import pos_zorder_sort
from jztree.fof import fof_labels, fof_is_superset, fof_labels_z, fof_order, fof_catalogue_from_groups
import h5py
import hdf5plugin

has_hfof = importlib.util.find_spec("hfof") is not None

@pytest.mark.skipif(not has_hfof, reason="requires hfof module installed")
def test_vs_hfof_uniform():
    from hfof import fof as fof_hfof

    boxsize = 1.0

    pos = jax.random.uniform(jax.random.PRNGKey(0), (1000000, 3), minval=0.0, maxval=boxsize)

    rlink = 0.8 * boxsize / len(pos)**(1/3)

    igr_jz = fof_labels.jit(pos, rlink=rlink, boxsize=boxsize)
    igr_hfof = fof_hfof(pos, rlink, boxsize=boxsize)

    # Since labels may differ, test set wise >= from both directions
    assert fof_is_superset(igr_hfof, igr_jz)
    assert fof_is_superset(igr_jz, igr_hfof)

    group_sizes_jz = jnp.sort(jnp.bincount(igr_jz, minlength=len(pos)))[::-1]
    group_sizes_hfof = jnp.sort(jnp.bincount(igr_hfof, minlength=len(pos)))[::-1]

    assert group_sizes_jz == pytest.approx(group_sizes_hfof)

def _particle_mass(omega_m: float, boxsize: float, npart: int) -> float:
    G = 43.007105731706317
    Hubble = 100.0
    return 1e10 * omega_m * 3 * Hubble * Hubble / (8 * np.pi * G) * boxsize ** 3 / npart

@pytest.fixture()
def camels_data():
    path_snap = Path(__file__).resolve().parent.parent / "data/CAMELS_snapshot.hdf5"
    path_groups = Path(__file__).resolve().parent.parent /"data/CAMELS_groups.hdf5"

    if not path_snap.exists() or not path_groups.exists():
        raise ValueError("Camels data not found, please download by executing ../prepare_tests.py")
    
    with h5py.File(path_snap) as fsnap:
        boxsize = fsnap["Header"].attrs["BoxSize"] / 1e3
        pos = fsnap['PartType1']['Coordinates'][:] / 1000
        vel = fsnap['PartType1']['Velocities'][:]
        mass = fsnap["Header"].attrs["MassTable"][1]
    part = ParticleData(pos=pos, vel=vel, mass=mass)
    
    with h5py.File(path_groups) as fgroup:
        pos_h = fgroup['Group']['GroupPos'][:] / 1000
        vel_h = fgroup['Group']['GroupVel'][:]
        count_h = fgroup['Group']['GroupLen'][:]
        mass_h = fgroup['Group']['GroupMass'][:]
    cata = FofCatalogue(ngroups=len(pos_h), mass=mass_h, count=count_h, com_pos=pos_h, com_vel=vel_h)
    cata = sort_catalogue(cata)

    rlink = 0.2 * boxsize / 256

    return part, cata, boxsize, rlink

@pytest.fixture()
def camels_jz_fof(camels_data):
    part, _, boxsize, rlink = camels_data

    cfg = FofConfig()
    cfg.tree.alloc_fac_nodes = 1.2
    
    partz, idz = pos_zorder_sort.jit(part)
    igr_jz = fof_labels_z.jit(partz.pos, rlink, boxsize=boxsize, cfg=cfg)

    part_fof, counts = fof_order(igr_jz, partz)
    cata = fof_catalogue_from_groups(part_fof, counts, boxsize=boxsize)
    cata = sort_catalogue(squeeze_catalogue(cata))

    return partz, igr_jz, part_fof, cata, boxsize, rlink

def test_CAMELS(camels_data, camels_jz_fof):
    cata_cam = camels_data[1]
    cata = camels_jz_fof[3]

    print("\n","Comparing CAMELS and jz_tree","\n")

    print(f"# of halos: {cata_cam.ngroups} vs {cata.ngroups}")
    print(f"# of minimal mass (32 particles) halos: {jnp.sum(cata_cam.count == 32), jnp.sum(cata.count == 32)}")
    print(f"# particles in haloes: {jnp.sum(cata_cam.count)} vs {jnp.sum(cata.count)}")

    print("Counts in heaviest 25 halos:")
    print("CAMELS:\n", cata_cam.count[:25])
    print("jz_tree:\n", cata.count[:25],"\n")

    print("Mass in heaviest 25 halos:")
    print("CAMELS:\n", cata_cam.mass[:25])
    print("jz_tree:\n", cata.mass[:25],"\n")

    print("Positions 5 most massive:")
    print("CAMELS:\n", cata_cam.com_pos[:5])
    print("jz_tree:\n", cata.com_pos[:5],"\n")

    print("Velocities 5 most massive:")
    print("CAMELS:\n", cata_cam.com_vel[:5])
    print("jz_tree:\n", cata.com_vel[:5],"\n")

@pytest.mark.skipif(not has_hfof, reason="requires hfof module installed")
def test_vs_hfof_camels(camels_jz_fof):
    from hfof import fof

    partz, igr_jz, part_fof, cata, boxsize, rlink = camels_jz_fof
    igr_hfof = fof(partz.pos, rlink, boxsize=boxsize)

    # uniquely map every jzfof-label to an hfof-label
    label_map = jnp.zeros(partz.pos.shape[0], dtype=jnp.int32).at[igr_jz].set(igr_hfof)
    label_map_rev = jnp.arange(partz.pos.shape[0], dtype=jnp.int32).at[igr_hfof].set(igr_jz)

    igr_hfof_jz = label_map[igr_jz]
    igr_jz_hfof = label_map_rev[igr_hfof]
    group_sizes_jz = jnp.sort(jnp.bincount(igr_jz, minlength=partz.pos.shape[0]))[::-1]
    group_sizes_hfof = jnp.sort(jnp.bincount(igr_hfof, minlength=partz.pos.shape[0]))[::-1]

    assert jnp.all(group_sizes_jz == pytest.approx(group_sizes_hfof)), "group size mismatch"
    assert jnp.all(igr_hfof_jz == igr_hfof), "group label mismatch"
    assert jnp.all(igr_jz_hfof == igr_jz), "group label mismatch"
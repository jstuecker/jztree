import pytest
import jztree.fof
import jax
import jax.numpy as jnp
import pytest
import importlib
from pathlib import Path
from jztree.data import ParticleData
from fmdj.ztree import pos_zorder_sort
import h5py
import hdf5plugin

has_hfof = importlib.util.find_spec("hfof") is not None

@pytest.mark.skipif(not has_hfof, reason="requires hfof module installed")
def test_vs_hfof_uniform():
    from hfof import fof

    boxsize = 1.0

    pos = jax.random.uniform(jax.random.PRNGKey(0), (1000000, 3), minval=0.0, maxval=boxsize)

    rlink = 0.8 * boxsize / len(pos)**(1/3)

    igr_jz = jztree.fof.fof.jit(pos, rlink=rlink, boxsize=boxsize)
    igr_hfof = fof(pos, rlink, boxsize=boxsize)

    # uniquely map every jzfof-label to an hfof-label
    label_map = jnp.zeros(len(pos), dtype=jnp.int32).at[igr_jz].set(igr_hfof)
    label_map_rev = jnp.arange(len(pos), dtype=jnp.int32).at[igr_hfof].set(igr_jz)

    igr_hfof_jz = label_map[igr_jz]
    igr_jz_hfof = label_map_rev[igr_hfof]

    assert igr_hfof_jz == pytest.approx(igr_hfof)
    assert igr_jz_hfof == pytest.approx(igr_jz)

    group_sizes_jz = jnp.sort(jnp.bincount(igr_jz, minlength=len(pos)))[::-1]
    group_sizes_hfof = jnp.sort(jnp.bincount(igr_hfof, minlength=len(pos)))[::-1]

    # print(group_sizes_jz[0:10])

    assert group_sizes_jz == pytest.approx(group_sizes_hfof)

@pytest.fixture()
def camels_data():
    file_path_snap = Path(__file__).resolve().parent.parent / "data/CAMELS_snapshot.hdf5"
    file_path_groups = Path(__file__).resolve().parent.parent /"data/CAMELS_groups.hdf5"

    if not file_path_snap.exists() or not file_path_groups.exists():
        raise ValueError("Camels data not found, please download by executing ../prepare_tests.py")
    
    file_snap = h5py.File(file_path_snap)['PartType1']
    pos = file_snap['Coordinates'][:] / 1000
    vel = file_snap['Velocities'][:] 
    
    file_groups = h5py.File(file_path_groups)['Group']
    pos_h = file_groups['GroupPos'][:] / 1000
    vel_h = file_groups['GroupVel'][:]
    group_len = file_groups['GroupLen'][:]

    return pos, vel, pos_h, vel_h, group_len

@pytest.fixture()
def camels_jz_fof(camels_data):
    pos, vel, pos_h, vel_h, group_len = camels_data

    res = 256

    boxsize = 25
    b = 0.2
    rlink = b * boxsize / res
    cfg = jztree.data.FofConfig()
    cfg.tree.alloc_fac_nodes = 2
    
    particles = ParticleData(jnp.asarray(pos), jnp.asarray(vel))
    
    particlesz, idz = pos_zorder_sort.jit(particles)
    igr_jz = jztree.fof.fof_z.jit(particlesz.pos, rlink, boxsize=boxsize, cfg=cfg)

    return particlesz, igr_jz, rlink, boxsize

def test_CAMELS(camels_data, camels_jz_fof):
    pos, vel, pos_h, vel_h, group_len = camels_data
    particlesz, igr_jz, rlink, boxsize = camels_jz_fof

    print("\n","Comparing CAMELS and jz_tree","\n")

    results = jztree.fof.fof_reduction(particlesz, igr_jz, boxsize=boxsize, Nmin=32)
    results.npart = results.npart[:results.ngroups]
    results.pos = results.pos[:results.ngroups]
    results.vel = results.vel[:results.ngroups]

    sort_by_mass_CAMELS = jnp.argsort(group_len)
    sort_by_mass_jz_tree = jnp.argsort(results.npart)
    
    print("# of halos indentified:")
    print("CAMELS:\n", pos_h.shape[0])
    print("jz_tree:\n", results.ngroups,"\n")

    print("# particles in heaviest 25 halos:")
    print("CAMELS:\n", group_len[sort_by_mass_CAMELS][-25:])
    print("jz_tree:\n", results.npart[sort_by_mass_jz_tree][-25:],"\n")
    
    print("# of minimal mass (32 particles) halos:")
    print("CAMELS:\n", jnp.sum(group_len == 32))
    print("jz_tree:\n", jnp.sum(results.npart == 32),"\n")

    
    print("Total number of particles in all halos combined:")
    print("CAMELS:\n", jnp.sum(group_len))
    print("jz_tree:\n", jnp.sum(results.npart))
    print("difference (CAMELS - jz_tree):\n", jnp.sum(group_len) - jnp.sum(results.npart[:results.ngroups]),"\n")

    print("Positions 5 most massive:")
    print("CAMELS:\n", pos_h[sort_by_mass_CAMELS][-5:])
    print("jz_tree:\n", results.pos[sort_by_mass_jz_tree][-5:],"\n")

    print("Velocities 5 most massive:")
    print("CAMELS:\n", vel_h[sort_by_mass_CAMELS][-5:])
    print("jz_tree:\n", results.vel[sort_by_mass_jz_tree][-5:],"\n")

@pytest.mark.skipif(not has_hfof, reason="requires hfof module installed")
def test_vs_hfof_camels(camels_data, camels_jz_fof):
    from hfof import fof

    pos, vel, pos_h, vel_h, group_len = camels_data
    particlesz, igr_jz, rlink, boxsize = camels_jz_fof
    
    igr_hfof = fof(particlesz.pos, rlink, boxsize=boxsize)

    # uniquely map every jzfof-label to an hfof-label
    label_map = jnp.zeros(particlesz.pos.shape[0], dtype=jnp.int32).at[igr_jz].set(igr_hfof)
    label_map_rev = jnp.arange(particlesz.pos.shape[0], dtype=jnp.int32).at[igr_hfof].set(igr_jz)

    igr_hfof_jz = label_map[igr_jz]
    igr_jz_hfof = label_map_rev[igr_hfof]
    group_sizes_jz = jnp.sort(jnp.bincount(igr_jz, minlength=particlesz.pos.shape[0]))[::-1]
    group_sizes_hfof = jnp.sort(jnp.bincount(igr_hfof, minlength=particlesz.pos.shape[0]))[::-1]

    assert jnp.all(group_sizes_jz == pytest.approx(group_sizes_hfof)), "group size mismatch"
    assert jnp.all(igr_hfof_jz == igr_hfof), "group label mismatch"
    assert jnp.all(igr_jz_hfof == igr_jz), "group label mismatch"
    # print(group_sizes_jz)
    # print(group_sizes_hfof)
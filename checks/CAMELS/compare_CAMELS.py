import pytest
import jztree.fof
from fmdj.ztree import pos_zorder_sort
import jax
import jax.numpy as jnp
import pytest
import importlib
import matplotlib.pyplot as plt

from jztree.data import ParticleData

import h5py
import hdf5plugin

import os
import urllib.request

url_snap = "https://users.flatironinstitute.org/~camels/Sims/Astrid_DM/CV/CV_0/snapshot_090.hdf5"
file_path_snap = "CAMELS_snapshot.hdf5"

url_groups = "https://users.flatironinstitute.org/~camels/Sims/Astrid_DM/CV/CV_0/groups_090.hdf5"
file_path_groups = "CAMELS_groups.hdf5"

if not os.path.exists(file_path_snap):
    print("File not found. Downloading...")
    urllib.request.urlretrieve(url_snap, file_path_snap)
if not os.path.exists(file_path_groups):
    print("File not found. Downloading...")
    urllib.request.urlretrieve(url_groups, file_path_groups)

def test_fof_CAMELS():
    print("\n","Comparing CAMELS and jz_tree","\n")
    res = 256
    boxsize = 25 # Mpc/h
    
    file_snap = h5py.File(file_path_snap)['PartType1']
    pos = file_snap['Coordinates'][:] / 1000
    vel = file_snap['Velocities'][:] 
    
    file_groups = h5py.File(file_path_groups)['Group']
    pos_h = file_groups['GroupPos'][:] / 1000
    vel_h = file_groups['GroupVel'][:]
    group_len = file_groups['GroupLen'][:]

    b = 0.2
    rlink = b * boxsize / res
    cfg = jztree.data.FofConfig()
    cfg.tree.alloc_fac_nodes = 2
    
    particles = ParticleData(jnp.asarray(pos), jnp.asarray(vel))
    del pos, vel
    
    # run jztree-fof
    particlesz, idz = pos_zorder_sort.jit(particles)
    igr_jz = jztree.fof.fof_z.jit(particlesz.pos, rlink, boxsize=boxsize, cfg=cfg)
    del particles

    results = fof_reduction(particlesz, igr_jz, boxsize=boxsize, Nmin=32)
    
    print("# of halos indentified:")
    print("CAMELS:", pos_h.shape[0])
    print("jz_tree:", results.ngroups,"\n")

    print("# particles in heaviest 25 halos:")
    print("CAMELS:", jnp.sort(group_len)[-25:])
    print("jz_tree:", jnp.sort(results.npart[:results.ngroups])[-25:],"\n")

    print("# of minimal mass (32 particles) halos:")
    print("CAMELS:", jnp.sum(group_len == 32))
    print("jz_tree:", jnp.sum(results.npart[:results.ngroups] == 32),"\n")

    
    print("Total number of particles in all halos combined:")
    print("CAMELS", jnp.sum(group_len))
    print("jz_tree:", jnp.sum(results.npart[:results.ngroups]))
    print("difference (CAMELS - jz_tree):", jnp.sum(group_len) - jnp.sum(results.npart[:results.ngroups]),"\n")

    # has_hfof = importlib.util.find_spec("hfof") is not None
    # if has_hfof:
    #     print("Comparing to hfof")
    #     from hfof import fof
    #     igr_hfof = fof(particlesz.pos, rlink, boxsize=boxsize)
    
    #     # uniquely map every jzfof-label to an hfof-label
    #     label_map = jnp.zeros(particlesz.pos.shape[0], dtype=jnp.int32).at[igr_jz].set(igr_hfof)
    #     label_map_rev = jnp.arange(particlesz.pos.shape[0], dtype=jnp.int32).at[igr_hfof].set(igr_jz)
    
    #     igr_hfof_jz = label_map[igr_jz]
    #     igr_jz_hfof = label_map_rev[igr_hfof]
    #     group_sizes_jz = jnp.sort(jnp.bincount(igr_jz, minlength=particlesz.pos.shape[0]))[::-1]
    #     group_sizes_hfof = jnp.sort(jnp.bincount(igr_hfof, minlength=particlesz.pos.shape[0]))[::-1]
    
    #     print("Group sizes consitent:", jnp.all(group_sizes_jz == pytest.approx(group_sizes_hfof)))
    #     print("Group labels consistent:", jnp.all(igr_hfof_jz == igr_hfof), jnp.all(igr_jz_hfof == igr_jz))
    #     print(group_sizes_jz)
    #     print(group_sizes_hfof)

test_fof_CAMELS()
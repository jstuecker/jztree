import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from jztree.data import PosMass, ParticleData, pad_particles
from jztree.comm import get_rank_info
from jztree.jax_ext import shard_map_constructor

def uniform_particles(N, boxsize=1.0, total_mass=1., seed=0, npad=0):
    rank, ndev, axis_name = get_rank_info()

    pos = jax.random.uniform(jax.random.PRNGKey(seed + rank), (N,3), dtype=jnp.float32)
    posmass = PosMass(pos=pos, mass=total_mass/(N*ndev), num=N, num_total=ndev*N)

    return pad_particles(posmass, npad)
uniform_particles.smap = shard_map_constructor(uniform_particles,
    in_specs=(None, None, None, None, None), out_specs=P(-1), static_argnums=(0,4)
)

def gaussian_particles(N, scale=1.0, total_mass=1., seed=0, npad=0):
    rank, ndev, axis_name = get_rank_info()

    pos = jax.random.normal(jax.random.PRNGKey(seed + rank), (N,3), dtype=jnp.float32) * scale
    posmass = PosMass(pos=pos, mass=total_mass/(N*ndev), num=N, num_total=ndev*N)

    return pad_particles(posmass, npad)
gaussian_particles.smap = shard_map_constructor(gaussian_particles,
    in_specs=(None, None, None, None, None), out_specs=P(-1), static_argnums=(0,4)
)

def hernquist_particles(N, a=1., M=1., anisotropy=0., seed=None):
    import aegis

    if seed is not None:
        np.random.seed(seed)
    prof = aegis.profiles.HernquistProfile(a=a, M=M, anisotropy=anisotropy)
    pos, vel, mass = prof.sample_particles(N, result="pos_vel_m", rpmin=1e-6*a, ramax=1e6*a)
    return ParticleData(pos=pos, mass=mass, vel=vel)

def discodj_particles(res):
    from discodj_examples.simulations import disco_sim
    pos = disco_sim(res=res, res_pm=res)[1].reshape(-1,3)

    mass = jnp.ones(len(pos), dtype=pos.dtype) / res**3
    return PosMass(pos=pos, mass=mass)
discodj_particles.jit = jax.jit(discodj_particles, static_argnames=("res"))
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from jztree.data import PosMass, ParticleData, pad_particles
from jztree.comm import get_rank_info
from jztree.jax_ext import shard_map_constructor

def uniform_particles(N, total_mass=1., seed=0, npad=0, dim=3):
    rank, ndev, axis_name = get_rank_info()

    pos = jax.random.uniform(jax.random.PRNGKey(seed + rank), (N,dim), dtype=jnp.float32)
    posmass = PosMass(pos=pos, mass=total_mass/(N*ndev), num=N, num_total=ndev*N)

    return pad_particles(posmass, npad)
uniform_particles.smap = shard_map_constructor(uniform_particles,
    in_specs=(None, None, None, None, None), out_specs=P(-1), static_argnums=(0,3,4)
)

def gaussian_particles(N, scale=1.0, total_mass=1., seed=0, npad=0, dim=3):
    rank, ndev, axis_name = get_rank_info()

    pos = jax.random.normal(jax.random.PRNGKey(seed + rank), (N,dim), dtype=jnp.float32) * scale
    posmass = PosMass(pos=pos, mass=total_mass/(N*ndev), num=N, num_total=ndev*N)

    return pad_particles(posmass, npad)
gaussian_particles.smap = shard_map_constructor(gaussian_particles,
    in_specs=(None, None, None, None, None, None), out_specs=P(-1), static_argnums=(0,4,5)
)

def hernquist_posmass(N, a=1., total_mass=1., seed=0, npad=0, dim=3, rmax=None):
    rank, ndev, axis_name = get_rank_info()

    key_r, key_dir = jax.random.split(jax.random.PRNGKey(seed + rank))
    u = jax.random.uniform(key_r, (N,), dtype=jnp.float32)
    if rmax is not None:
        umax = (rmax / (rmax + a)) ** 2
        u = u * jnp.asarray(umax, dtype=u.dtype)
    u = jnp.minimum(u, jnp.asarray(1. - jnp.finfo(u.dtype).eps, dtype=u.dtype))

    sqrt_u = jnp.sqrt(u)
    r = a * sqrt_u / (1. - sqrt_u)

    if dim == 3:
        key_mu, key_phi = jax.random.split(key_dir)
        mu = jax.random.uniform(key_mu, (N,), minval=-1., maxval=1., dtype=jnp.float32)
        phi = jax.random.uniform(key_phi, (N,), minval=0., maxval=2. * jnp.pi, dtype=jnp.float32)
        sin_theta = jnp.sqrt(jnp.maximum(0., 1. - mu * mu))
        direction = jnp.stack(
            (sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), mu), axis=-1
        )
    elif dim == 2:
        phi = jax.random.uniform(key_dir, (N,), minval=0., maxval=2. * jnp.pi, dtype=jnp.float32)
        direction = jnp.stack((jnp.cos(phi), jnp.sin(phi)), axis=-1)
    else:
        direction = jax.random.normal(key_dir, (N, dim), dtype=jnp.float32)
        direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)

    pos = direction * r[:, None]
    mass = total_mass / (N * ndev)
    part = PosMass(pos=pos, mass=mass, num=N, num_total=N * ndev)
    return pad_particles(part, npad)
hernquist_posmass.smap = shard_map_constructor(hernquist_posmass,
    in_specs=(None, None, None, None, None, None, None), out_specs=P(-1), static_argnums=(0,4,5,6)
)

def hernquist_particles(N, a=1., M=1., anisotropy=0., seed=None):
    import aegis

    if seed is not None:
        np.random.seed(seed)
    prof = aegis.profiles.HernquistProfile(a=a, M=M, anisotropy=anisotropy)
    pos, vel, mass = prof.sample_particles(N, result="pos_vel_m", rpmin=1e-6*a, ramax=1e6*a)
    return ParticleData(pos=pos, mass=mass, vel=vel)

def discodj_particles(res, boxsize=100.):
    from discodj import DiscoDJ
    dj = DiscoDJ(dim=3, res=res, boxsize=boxsize)
    dj = dj.with_timetables()
    dj = dj.with_linear_ps()
    dj = dj.with_ics()
    dj = dj.with_lpt(n_order=1)
    X, P, a = dj.run_nbody(a_ini=0.02, a_end=1.0, n_steps=10, res_pm=res, stepper="bullfrog")
    pos = X.reshape(-1,3)

    mass = jnp.ones(len(pos), dtype=pos.dtype) / res**3
    return PosMass(pos=pos, mass=mass)
discodj_particles.jit = jax.jit(discodj_particles, static_argnames=("res", "boxsize"))

def multi_gpu_dj_sim(boxsize = 1000., num_per_device=512**3) -> ParticleData:
    from discodj import DiscoDJ
    from discodj.core.scatter_and_gather import ScatterGatherProperties

    def _particle_mass(omega_m: float, boxsize: float, npart: int) -> float:
        G = 43.007105731706317
        Hubble = 100.0
        return 1e10 * omega_m * 3 * Hubble * Hubble / (8 * np.pi * G) * boxsize ** 3 / npart

    ndev = jax.device_count()

    nres = np.int64(((np.cbrt(num_per_device * ndev))//ndev)*ndev)

    # print(f"total grid dim {nres}, particles per GPU {np.cbrt(nres**3/ndev):.2f}**3")
    
    scat = ScatterGatherProperties(
        res=nres,
        res_pm=nres,
        num_devices=ndev,
        use_distributed_scatter_gather=True,
        use_vjp_gather=False,
        use_vjp_scatter=False,
        scatter_gather_check=False
    )

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
multi_gpu_dj_sim.jit = jax.jit(multi_gpu_dj_sim, static_argnums=(0,1))


# ------------------------------------------------------------------------------------------------ #
#                                              Samples                                             #
# ------------------------------------------------------------------------------------------------ #

def cosmo_2d_sample():
    from importlib.resources import files

    file = files("jztree_utils") / "data" / "pos2d_cosmo_128_100.npy"
    return np.load(file)

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
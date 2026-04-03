import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax
import jax.numpy as jnp
import numpy as np

import jfof.fofkd_exp as fof_e
import jfof.fofkd

from pytest_jax_bench import JaxBench

def djsim(ngrid, boxsize, pad=0):
    from discodj import DiscoDJ
    dj = DiscoDJ(dim=3, res=ngrid, boxsize=boxsize)
    dj = dj.with_timetables()
    pk = dj.with_linear_ps()
    ic = dj.with_ics(pk)
    lpt = dj.with_lpt(ic, n_order=1)
    sim_ini = dj.with_lpt_ics(lpt, n_order=1, a_ini=0.02)
    X, P, a = dj.run_nbody(sim_ini, a_end=1.0, n_steps=10, res_pm=ngrid, stepper="bullfrog")
    pos = X.reshape(-1,3) % boxsize
    if pad > 0:
        pos = jnp.pad(pos, ((0,pad), (0,0)), constant_values=boxsize+1.)
    return pos
djsim.jit = jax.jit(djsim, static_argnames=("ngrid", "boxsize", "pad"))

jb = JaxBench(jit_rounds=10, jit_warmup=1)

res = dict(ngrid = [], t = [])
for ngrid, pad in ((64,0), (88,0), (128,1), (180,0), (252,0), (360,0), (512,1)):
    x = djsim.jit(ngrid, ngrid*1., pad=pad)

    # timing1 = jb.measure(fn_jit=jfof.fofkd.fof_clusters_jit, pos=x, b=0.2, cuda=True, write=False)[0]
    
    fof_fn = fof_e.make_fof_frozen(k=8, max_iters=300, min_size=20, b=0.20, cuda=True)
    fof_fn = jax.jit(fof_e.wrap_pos_arg(fof_fn))
    timing2 = jb.measure(fn_jit=fof_fn, pos=x, write=False)[0]

    res["ngrid"].append(ngrid)
    res["t"].append(timing2.jit_mean_ms)

np.savez("out/jfof_timing.npz", **res)
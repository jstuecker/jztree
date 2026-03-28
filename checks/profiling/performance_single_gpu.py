import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import io
import matplotlib.pyplot as plt

from pytest_jax_bench import JaxBench
import jax
import jax.numpy as jnp
import jztree as jz
from jztree_utils import ics

def get_part(n, mode="uniform", seed=0):
    if mode == "grid":
        xi = jnp.arange(int(np.cbrt(n)))*(1./n)
        return jnp.stack(jnp.meshgrid(xi, xi, xi, indexing="ij"), axis=-1).reshape(-1,3)
    elif mode == "uniform":
        return jax.random.uniform(jax.random.key(seed), (int(n),3), dtype=jnp.float32)
    elif mode == "normal":
        return jax.random.normal(jax.random.key(seed), (int(n),3), dtype=jnp.float32)
    elif mode == "discodj" or mode == "discodj_per":
        res = int(np.cbrt(n))
        return ics.discodj_particles.jit(res, boxsize=1.*res).pos

def bench_distributions():
    jb = JaxBench(jit_rounds=10, jit_warmup=1)

    ns = np.logspace(5, 8, 7)
    ns = ((np.cbrt(ns) // 4) * 4)**3 # round a bit so discodj is happy
    res = dict(n=ns, discodj=[], discodj_per=[], grid=[], uniform=[], normal=[])

    for mode in "discodj", "discodj_per", "grid", "uniform", "normal":
        for n in ns:
            x = get_part(n, mode=mode)

            cfg = jz.config.KNNConfig()
            cfg.tree.alloc_fac_nodes = 1.2

            if "per" in mode:
                boxsize = int(np.cbrt(n))*1.
            else:
                boxsize = 0.

            timing, (rnn, inn) = jb.measure(fn_jit=jz.knn.knn.jit, part=x, k=16, write=False, cfg=cfg, boxsize=boxsize)
            res[mode].append(timing.jit_mean_ms)

    return res

if os.path.exists("out/knn_distributions.npz"):
    res = np.load("out/knn_distributions.npz")
else:
    res = bench_distributions()
    np.savez("out/knn_distributions.npz", **res)

plt.figure(figsize=(5,3.5))
plt.loglog(res["n"], res["grid"], label="grid", marker="o")
plt.loglog(res["n"], res["uniform"], label="uniform", marker="o")
plt.loglog(res["n"], res["normal"], label="normal", marker="o")
plt.loglog(res["n"], res["discodj"], label="cosm. (no wrapping)", marker="o")
plt.loglog(res["n"], res["discodj_per"], label="cosm. (with wrapping)", marker="o")

nlin = np.logspace(6.5, 8.)
plt.loglog(nlin, nlin / 3.5e5, ls="dashed", color="black")
plt.annotate("linear scaling", (1.2e7, 2.2e1))

plt.legend()
plt.xlabel("N")
plt.ylabel("Time [ms]")

plt.savefig("out/knn_distributions.pdf", bbox_inches="tight")

def bench_k():
    jb = JaxBench(jit_rounds=10, jit_warmup=1)

    n = int(1e6)
    ks = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

    res = dict(k=ks, uniform=[])

    for mode in "uniform",: 
        for k in ks:
            x = get_part(n, mode=mode)

            cfg = jz.config.KNNConfig(alloc_fac_ilist=400.) # need larger ilist for k=1024
            cfg.tree.alloc_fac_nodes = 1.2

            timing, (rnn, inn) = jb.measure(fn_jit=jz.knn.knn.jit, part=x, k=k, write=False, cfg=cfg)
            res[mode].append(timing.jit_mean_ms)

    return res

if os.path.exists("out/knn_k.npz"):
    res = np.load("out/knn_k.npz")
else:
    res = bench_k()
    np.savez("out/knn_k.npz", **res)

plt.figure(figsize=(5,3.5))
plt.loglog(res["k"], res["uniform"], marker="o", label=r"$N = 10^6$")
plt.xlabel("k")
plt.ylabel("Time [ms]")
plt.legend()

klin = np.logspace(2, 3)
plt.loglog(klin, klin / 10., ls="dashed", color="black")
plt.annotate("linear\nscaling", (3e2, 1.6e1))

plt.savefig("out/knn_k.pdf", bbox_inches="tight")

def bench_query():
    jb = JaxBench(jit_rounds=10, jit_warmup=1)

    ns = np.logspace(5, 8, 7)[:-1]
    res = dict(n=ns)

    mode = "uniform"
    # for mode in "uniform",:
    for nsrc in (None, int(1e6), int(1e7)):
        res[str(nsrc)] = []
        for n in ns:
            if nsrc is None:
                x = get_part(n, mode=mode, seed=0)
                xq = None
            else:
                x = get_part(nsrc, mode=mode, seed=0)
                xq = get_part(n, mode=mode, seed=1)

            cfg = jz.config.KNNConfig(alloc_fac_ilist=380.)
            cfg.tree.alloc_fac_nodes = 1.2

            timing, (rnn, inn) = jb.measure(fn_jit=jz.knn.knn.jit, part=x, k=16, write=False, cfg=cfg, part_query=xq)
            res[str(nsrc)].append(timing.jit_mean_ms)

        # timing, (rnn, inn) = jb.measure(fn_jit=jz.knn.knn.jit, part=x, k=16, write=False, cfg=cfg)
        # res["selfq"] = timing.jit_mean_ms

    return res

if os.path.exists("out/knn_query.npz"):
    res = np.load("out/knn_query.npz")
else:
    res = bench_query()
    np.savez("out/knn_query.npz", **res)

plt.figure(figsize=(5,3.5))
plt.loglog(res["n"], res[str(int(1e6))], marker="o", label=r"$N_{\rm{src}} = 10^6$")
plt.axhline(res[str(None)][2], color="C0", ls="dashed", xmin=0, xmax=0.37)
plt.loglog(res["n"], res[str(int(1e7))], marker="o", label=r"$N_{\rm{src}} = 10^7$")
plt.loglog(res["n"], res[str(None)], marker="o", label=r"$x_{\rm{src}} = x_{\rm{query}}$", ls="dashed", color="black")
plt.axhline(res[str(None)][4], color="C1", ls="dashed", xmin=0, xmax=0.75)
plt.xlabel(r"$N_{\rm{query}}$")
plt.ylabel("Time [ms]")
plt.legend()
plt.xlim(1e5, None)

plt.savefig("out/knn_query.pdf", bbox_inches="tight")
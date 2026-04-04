import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))

import numpy as np
import io
import matplotlib.pyplot as plt

from pytest_jax_bench import JaxBench
import jax
import jax.numpy as jnp
import jztree as jz
from jztree_utils import ics

# jax.distributed.initialize()

def bench_ndev(ndev):
    jb = JaxBench(jit_rounds=10, jit_warmup=1)

    ns = np.geomspace(64**3, 512**3, 7)
    ns = ((np.cbrt(ns*ndev)  // 4) * 4)**3 // ndev
    ns = ((np.cbrt(ns*ndev)  // ndev) * ndev)**3 // ndev

    print(ns, np.cbrt(ns*ndev), np.cbrt(ns))

    mesh = jax.make_mesh((ndev,), axis_names=("gpus",), axis_types=jax.sharding.AxisType.Explicit)
    res = dict(time=[], n=[])

    for n in ns:
        if jax.process_index() == 0:
            print(ndev, n)
        
        cfg = jz.config.FofConfig()
        if n <= 128**3:
            cfg.tree.alloc_fac_nodes = 2.0
            padding = 0.8
        else:
            cfg.tree.alloc_fac_nodes = 1.3
            padding = 0.5

        rlink = 0.2
        
        if ndev > 1:
            boxsize = np.cbrt(n * ndev)*1.
            part = ics.multi_gpu_dj_sim.jit(boxsize=boxsize, num_per_device=n)
            res["n"].append(len(part.pos))

            def pad(part):
                part = jz.data.pad_particles(part, int(n*padding))
                return part
            pad = jz.jax_ext.expanding_shard_map(pad, mesh=mesh, jit=True)

            part = jz.data.expand_particles(part, ndev)
            part = pad(part)
            f = jz.fof.fof_and_catalogue.smap(mesh, jit=True)
            timing = jb.measure(
                fn_jit=f, part=part, rlink=rlink, boxsize=boxsize, write=False, cfg=cfg
            )[0]
        else:
            ng = (np.cbrt(n)//4)*4
            part = ics.discodj_particles(ng, boxsize=1.*ng)
            res["n"].append(len(part.pos))

            f = jz.fof.fof_and_catalogue.jit
            timing = jb.measure(
                fn_jit=f, part=part, rlink=rlink, boxsize=boxsize, write=False, cfg=cfg
            )[0]

        res["time"].append(timing.jit_mean_ms)

    return res

# ndevices = jax.device_count()
ndevices = 4
fname = f"out/fof_devices_{ndevices}.npz"

if not os.path.exists(fname):
    res = bench_ndev(ndevices)
    if jax.process_index() == 0:
        np.savez(fname, **res)

print(512**3 / 1e8)

plt.figure(figsize=(5,3.5))
for ndev in (1,4,16,64):
    color = plt.get_cmap("viridis")(np.log2(ndev) / 6)
    res = np.load(f"out/fof_devices_{ndev}.npz")
    label = f"{ndev}GPU{'s' if ndev > 1 else ''}"
    plt.loglog(res["n"]/ndev, res["time"], marker="o", label=label, color=color)

nlin = np.logspace(7., 8.)
plt.loglog(nlin, nlin / 2.e5, ls="dashed", color="black")
plt.annotate("linear scaling", (2.5e7, 8e1))
plt.legend()
plt.xlabel("N per GPU")
plt.ylabel("Time [ms]")

plt.savefig("out/fof_devices.pdf", bbox_inches="tight")

# ------------------------------------------------------------------------------------------------ #
#                                           Library plot                                           #
# ------------------------------------------------------------------------------------------------ #

data = np.load("out/fof_results_gadget_hfof.npz")

plt.figure(figsize=(5,3.5))

plt.loglog(data["n"], data["hfof_ms"], marker="o", label="hfof [1CPU]")
plt.loglog(data["n"], data["g4_peano_n1_ms"] + data["g4_fofcomplete_n1_ms"], marker="o", label="Gadget4 [1CPU]")
plt.loglog(data["n"], data["g4_peano_n32_ms"] + data["g4_fofcomplete_n32_ms"], marker="o", label="Gadget4 [32CPU]")

res = np.load("out/jfof_timing.npz")
plt.loglog(res["ngrid"]**3, res["t"], marker="o", label="jfof [1GPU]")

print("jfof", res["t"][-1], res["t"][-1]/1243.826)

print(np.cbrt(data["n"]), (data["g4_peano_n32_ms"] + data["g4_fofcomplete_n32_ms"])[-1], data["jzfof_ms"][-1])
for ndev in (1,):
    color = plt.get_cmap("viridis")(np.log2(ndev) / 6)
    res = np.load(f"out/fof_devices_{ndev}.npz")

    print(res["time"])
    plt.loglog(res["n"], res["time"], label=f"jz-tree [{ndev}GPU{'s' if ndev > 1 else ''}]", marker="o", color="black")

print((data["g4_peano_n1_ms"] + data["g4_fofcomplete_n1_ms"])[-1])
print((data["hfof_ms"] + data["hfof_ms"])[-1])
print((data["g4_peano_n32_ms"] + data["g4_fofcomplete_n32_ms"])[-1])
print(res["time"][-1])
print(6583.27 / 1243.826)

print((data["g4_peano_n1_ms"] + data["g4_fofcomplete_n1_ms"])[-1]/res["time"][-1])
print((data["hfof_ms"] + data["hfof_ms"])[-1]/res["time"][-1])
print((data["g4_peano_n32_ms"] + data["g4_fofcomplete_n32_ms"])[-1]/res["time"][-1])


plt.xlim(None, 2e8)
plt.xlabel("N")
plt.ylabel("Time [ms]")

ax = plt.gca()
ax2 = ax.secondary_xaxis("top", functions=(lambda x: np.cbrt(x), lambda x: x**3))
ngs = [64, 88, 128, 180, 252, 360, 512]
lab = [rf"${ng}^3$" for ng in ngs]
ax2.set_xticks(ngs, lab)
import matplotlib.ticker as ticker
ax2.xaxis.set_minor_locator(ticker.NullLocator())

plt.legend()

plt.savefig("out/fof_libraries.pdf", bbox_inches="tight")

print(data)
print(data.keys())

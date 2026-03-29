import os
import numpy as np
import io
import matplotlib.pyplot as plt

from pytest_jax_bench import JaxBench
import jax
import jax.numpy as jnp
import jztree as jz
from jztree_utils import ics

jax.distributed.initialize()

def bench_ndev(ndevices):
    jb = JaxBench(jit_rounds=4, jit_warmup=1)

    ns = np.logspace(5, 8, 7)[::-1]

    ndevs = 2**np.arange(0, 10)
    ndevs = ndevs[ndevs <= ndevices][::-1]

    res = dict(ndevs=ndevs, ns=ns)

    for ndev in ndevs:
        mesh = jax.make_mesh((ndev,), axis_names=("gpus",), axis_types=jax.sharding.AxisType.Explicit)
        res[str(ndev)] = []
        for n in ns:
            if jax.process_index() == 0:
                print(ndev, n)
            
            pad, nfac = (1.0, 4.0) if n <= 1e7 else (0.6, 2.5)

            cfg = jz.config.KNNConfig()
            cfg.tree.alloc_fac_nodes = nfac
            
            if ndev > 1:
                x = ics.uniform_particles.smap(mesh, jit=True)(int(n), npad=int(max(n*pad, 1e6)))
                f = jz.knn.knn.smap(mesh, jit=True)
                timing = jb.measure(
                    fn_jit=f, part=x, k=16, write=False, cfg=cfg, result="rad", output_order="z"
                )[0]
            else:
                x = ics.uniform_particles(int(n))
                timing = jb.measure(
                    fn_jit=jz.knn.knn.jit, part=x, k=16, write=False, cfg=cfg,
                    result="rad", output_order="z"
                )[0]

            res[str(ndev)].append(timing.jit_mean_ms)

    return res

ndevices = jax.device_count()
fname = f"out/knn_devices_{ndevices}.npz"

if not os.path.exists(fname):
    res = bench_ndev(ndevices)
    if jax.process_index() == 0:
        np.savez(fname, **res)
else:
    res = np.load(fname)

if jax.process_index() == 0:
    plt.figure(figsize=(5,3.5))
    for ndev in (4,2,1):
        # color = plt.get_cmap("viridis")(np.log2(ndev) / 5)
        plt.loglog(res["ns"], res[str(ndev)], label=f"{(ndev + 3)//4} Node, {ndev} GPU", marker="o")
    plt.legend()
    plt.xlabel("N per GPU")
    plt.ylabel("Time [ms]")
    plt.savefig("out/knn_devices.pdf", bbox_inches="tight")
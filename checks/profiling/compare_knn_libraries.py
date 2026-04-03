import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault(
    "CONDA_PREFIX",
    "/leonardo/home/userexternal/jstuecke/miniforge3/envs/cu12.2",
) # cupy reads this variable, but VSCode doesn't provide it, so we set it explicitly

import numpy as np
import io
import matplotlib.pyplot as plt

import cupy as cp
from cupy_knn import LBVHIndex

from pytest_jax_bench import JaxBench
import jax
import jax.numpy as jnp
import jztree as jz
from scipy.spatial import KDTree
import time

print(jax.devices())

def bench_scipy():
    ts = []

    # ns = jnp.logspace(3, 8, 11)[:-1]
    ns = jnp.logspace(3, 7, 9)#[:-1]
    for n in ns:
        x = np.random.uniform(size=(int(n), 3))
        xq = np.random.uniform(size=(int(n), 3))

        t0 = time.perf_counter()
        for i in range(4):
            tree = KDTree(x)
            tree.query(xq, k=30, workers=32)
        ts.append((time.perf_counter() - t0) / 4. * 1e3)
    return dict(ns=ns, ts=ts)

if os.path.exists("out/scipy.npz"):
    res = np.load("out/scipy.npz")
else:
    res = bench_scipy()
    np.savez("out/scipy.npz", **res)

plt.figure(figsize=(5,3.5))
plt.plot(res["ns"], res["ts"], label="scipy [32CPU]", marker="o")

res = np.load("out/faiss.npy")
ns, ts = res[:,0], res[:,1]
plt.loglog(ns, ts, marker="o", label="faiss")


def bench_jz():
    ts = []
    jb = JaxBench(jit_rounds=10, jit_warmup=1)

    ns = jnp.logspace(3, 8, 11)[:-1] # have to skip 1e8 with k=30

    for n in ns:
        x = jax.random.uniform(jax.random.key(0), (int(n),3), dtype=jnp.float32)
        xq = jax.random.uniform(jax.random.key(1), (int(n),3), dtype=jnp.float32)

        res, (rnn, inn) = jb.measure(fn_jit=jz.knn.knn.jit, part=x, k=30, write=False, part_query=xq)
        ts.append(res.jit_mean_ms)

    return ns, ts

def bench_cupy_knn():
    ts = []
    ns = jnp.logspace(3, 7, 9)

    cp.cuda.Device(0).use()
    k = 30

    # warm up kernel compilation / allocator
    xw = cp.random.uniform(size=(1024, 3), dtype=cp.float32)
    xqw = cp.random.uniform(size=(1024, 3), dtype=cp.float32)
    index = LBVHIndex()
    index.build(xw)
    index.prepare_knn_default(k)
    _ = index.query_knn(xqw)
    cp.cuda.runtime.deviceSynchronize()

    for n in ns:
        n = int(n)

        # generate directly on GPU so host->device transfer is excluded
        x = cp.random.uniform(size=(n, 3), dtype=cp.float32)
        xq = cp.random.uniform(size=(n, 3), dtype=cp.float32)
        cp.cuda.runtime.deviceSynchronize()

        t0 = time.perf_counter()
        for _ in range(4):
            index = LBVHIndex()
            index.build(x)
            index.prepare_knn_default(k)
            _ = index.query_knn(xq)
        cp.cuda.runtime.deviceSynchronize()

        ts.append((time.perf_counter() - t0) / 4.0 * 1e3)

        # optional, helps reduce fragmentation for large sweeps
        cp.get_default_memory_pool().free_all_blocks()

    return dict(ns=np.asarray(ns), ts=ts)

if os.path.exists("out/cupyknn.npz"):
    res = np.load("out/cupyknn.npz")
else:
    res = bench_cupy_knn()
    np.savez("out/cupyknn.npz", **res)

plt.loglog(res["ns"], res["ts"], marker="o", label="cupy-knn")

def bench_jaxkd():
    import jaxkd

    ts = []
    jb = JaxBench(jit_rounds=3, jit_warmup=1)

    ns = jnp.logspace(3, 8, 11)[:-1]

    def f(pos, posq, k):
        return jaxkd.build_and_query(pos, pos, k=k, cuda=True)
    f.jit = jax.jit(f, static_argnames=["k"])

    for n in ns:
        x = jax.random.uniform(jax.random.key(0), (int(n),3), dtype=jnp.float32)
        xq = jax.random.uniform(jax.random.key(1), (int(n),3), dtype=jnp.float32)

        res, (rnn, inn) = jb.measure(fn_jit=f.jit, pos=x, posq=xq, k=30, write=False)
        ts.append(res.jit_mean_ms)

    return ns, ts

if os.path.exists("out/jaxkd.npz"):
    res = np.load("out/jaxkd.npz")
    ns, ts = res["ns"], res["ts"]
else:
    ns, ts = bench_jaxkd()
    np.savez("out/jaxkd.npz", ns=ns,ts=ts)

plt.plot(ns, ts, label="jaxkd-cuda", marker="o")

with open("out/clover.txt", "r") as f:
    lines = f.read()
for c in ("/", ",", "(", "[", ",", "]", ")"):
    lines = lines.replace(c, "")

arr = np.loadtxt(io.StringIO(lines), dtype=np.int64)

ns, ts = arr[:,1], 0.5*(arr[:,3] + arr[:,4])/1e6
plt.xlabel("N")
plt.ylabel("Time [ms]")
plt.loglog(ns, ts, marker="o", label="clover")

if os.path.exists("out/jz.npz"):
    res = np.load("out/jz.npz")
    ns, ts = res["ns"], res["ts"]
else:
    ns, ts = bench_jz()
    np.savez("out/jz.npz", ns=ns,ts=ts)

plt.plot(ns, ts, label="jz-tree", marker="o", color="black")

plt.legend()

plt.ylim(0.2, 1e4)
plt.xlim(1e3,4e7)
plt.grid("on")

nlin = np.logspace(6, 7.5)
plt.loglog(nlin, nlin/5e5, color="black", ls="dashed")
plt.annotate("linear\nscaling", (3e6, 1.5))

plt.savefig("out/libraries.pdf", bbox_inches="tight")

# ------------------------------------------------------------------------------------------------ #
#                                            Dimensions                                            #
# ------------------------------------------------------------------------------------------------ #


plt.figure(figsize=(4,3.))

res = np.load("out/faiss_dim.npy")
ds, ts = res[:,0], res[:,1]
plt.semilogy(ds, ts, marker="o", label="faiss")

def bench_jz_dim():
    ts = []
    jb = JaxBench(jit_rounds=10, jit_warmup=1)

    N = int(1e6)
    ds = np.array((2,3,4,5,6,7,8))

    for d in ds:
        x = jax.random.uniform(jax.random.key(0), (N,d), dtype=jnp.float32)

        cfg = jz.config.KNNConfig(alloc_fac_ilist = 300*(d/3)**5)
        print(d, cfg.alloc_fac_ilist)

        res, (rnn, inn) = jb.measure(fn_jit=jz.knn.knn.jit, part=x, k=16, write=False, cfg=cfg)
        ts.append(res.jit_mean_ms)

    return ds, ts

if os.path.exists("out/jzdim.npz"):
    res = np.load("out/jzdim.npz")
    ds, ts = res["ds"], res["ts"]
else:
    ds, ts = bench_jz_dim()
    np.savez("out/jzdim.npz", ds=ds,ts=ts)

plt.semilogy(ds, ts, label="jz-tree", marker="o", color="black")

plt.legend()
plt.xlabel("dimension")
plt.ylabel("Time [ms]")

plt.savefig("out/dimensions.pdf", bbox_inches="tight")

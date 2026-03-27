import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import io
import matplotlib.pyplot as plt

from pytest_jax_bench import JaxBench
import jax
import jax.numpy as jnp
import jztree as jz

print(jax.devices())

res = np.load("out/faiss.npy")
ns, ts = res[:,0], res[:,1]
plt.loglog(ns, ts, marker="o", label="FAISS")

with open("out/clover.txt", "r") as f:
    lines = f.read()
for c in ("/", ",", "(", "[", ",", "]", ")"):
    lines = lines.replace(c, "")

arr = np.loadtxt(io.StringIO(lines), dtype=np.int64)

ns, time = arr[:,1], 0.5*(arr[:,3] + arr[:,4])/1e6
plt.xlabel("N")
plt.ylabel("Time [ms]")
plt.loglog(ns, time, marker="o", label="Clover KNN")


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

plt.plot(ns, ts, label="jaxkd[cuda]", marker="o")

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

plt.savefig("out/libraries.pdf", bbox_inches="tight")

plt.figure()

res = np.load("out/faiss_dim.npy")
ds, ts = res[:,0], res[:,1]
plt.semilogy(ds, ts, marker="o", label="FAISS")

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

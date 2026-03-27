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

with open("out/clover.txt", "r") as f:
    lines = f.read()
for c in ("/", ",", "(", "[", ",", "]", ")"):
    lines = lines.replace(c, "")

arr = np.loadtxt(io.StringIO(lines), dtype=np.int64)

ns, time = arr[:,1], 0.5*(arr[:,3] + arr[:,4])/1e6
plt.xlabel("N")
plt.ylabel("Time [ms]")
plt.loglog(ns, time, marker="o", label="Clover KNN")

res = np.load("out/faiss.npy")
ns, ts = res[:,0], res[:,1]
plt.loglog(ns, ts, marker="o", label="FAISS")

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

    ns = jnp.logspace(3, 7, 9)

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

plt.savefig("out/libraries.pdf", bbox_inches="tight")
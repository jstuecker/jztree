import time
from pathlib import Path

import faiss
import numpy as np

OUT = Path("out/faiss.npy")
OUT.parent.mkdir(parents=True, exist_ok=True)

d = 3
k = 30
repeats = 5
ns = np.rint(np.asarray(np.logspace(3, 6, 7))).astype(np.int64)

res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))

results = []
for n in ns:
    rng = np.random.default_rng(int(n))
    xb = rng.random((n, d), dtype=np.float32)
    xq = rng.random((n, d), dtype=np.float32)

    # warmup
    w = min(n, 1024)
    index.reset()
    index.add(xb[:w])
    index.search(xq[:w], k)
    res.syncDefaultStreamCurrentDevice()

    times = []
    for _ in range(repeats):
        index.reset()
        t0 = time.perf_counter()
        index.add(xb)
        index.search(xq, k)
        res.syncDefaultStreamCurrentDevice()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    results.append((int(n), float(np.median(times) * 1e3)))
    print(f"n={n:>8d}  total={np.median(times):.6f}s")

np.save(OUT, np.array(results, dtype=np.float64))
print(f"\nWrote {OUT}")

# dimension test:

OUT = Path("out/faiss_dim.npy")
OUT.parent.mkdir(parents=True, exist_ok=True)

k = 16
repeats = 5
n = int(1e6)
ds = (2,3,4,5,6,7,8)

results = []
for d in ds:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))

    rng = np.random.default_rng(int(n))
    xb = rng.random((n, d), dtype=np.float32)
    # xq = rng.random((n, d), dtype=np.float32)

    # warmup
    w = min(n, 1024)
    index.reset()
    index.add(xb[:w])
    index.search(xb[:w], k)
    res.syncDefaultStreamCurrentDevice()

    times = []
    for _ in range(repeats):
        index.reset()
        t0 = time.perf_counter()
        index.add(xb)
        index.search(xb, k)
        res.syncDefaultStreamCurrentDevice()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    results.append((int(d), float(np.median(times) * 1e3)))
    print(f"d={d:>8d}  total={np.median(times):.6f}s")

np.save(OUT, np.array(results, dtype=np.float64))
print(f"\nWrote {OUT}")
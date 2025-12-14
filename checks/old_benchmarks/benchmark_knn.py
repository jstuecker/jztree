import jax
import jax.numpy as jnp
import jztree as jz
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from fmdj_utils.utility import Tee, Timer
import sys
import numpy as np

sys.stdout = Tee(sys.stdout, open("logs/knn.log", "a+"))

print(f"============== Query = Build ==============")
timer = Timer(verbose=True, loops=100, print_compile=True, print_warmup=False)

boxsize = 0.

k = 8

for N in (1e5,3e5,1e6,2e6,4e6,7e6,1e7):
    N = int(N)
    timer.set_tag(N=N)
    pos0 = jax.random.uniform(jax.random.PRNGKey(0), (N,3), minval=0, maxval=1, dtype=jnp.float32).block_until_ready()
    posz, idz = timer.timeit_jit(jz.tree.pos_zorder_sort.jit, pos0)

    timer.timeit_jit(jz.knn.knn.jit, pos0, k=k, name="total (unsorted)")
    data = timer.timeit_jit(jz.knn.prepare_knn.jit, pos0, k=k, name="prepare (unsorted)")
    timer.timeit_jit(jz.knn.evaluate_knn_z.jit, data, name="evaluate (unsorted)")

    timer.timeit_jit(jz.knn.knn_z.jit, posz, k=k, name="total (sorted)")
    data = timer.timeit_jit(jz.knn.prepare_knn_z.jit, posz, k=k, name="prepare (sorted)")
    timer.timeit_jit(jz.knn.evaluate_knn_z.jit, data, name="evaluate (sorted)")

timer.plot_timings("N")
plt.savefig("logs/knn.pdf")
# plt.show()

print(f"============== Query != Build ==============")
timer = Timer(verbose=True, loops=100, print_compile=True, print_warmup=False)

plt.figure()

for N in (1e5,3e5,1e6,2e6,4e6,7e6,1e7):
    N = int(N)
    timer.set_tag(Nquery=N)
    posQ = jax.random.uniform(jax.random.PRNGKey(1), (N//10,3), minval=0, maxval=1, dtype=jnp.float32).block_until_ready()
    poszQ, idz = jz.tree.pos_zorder_sort.jit(posQ)

    for Nbuild in 2e5, 1e6, 5e6:
        Nbuild = int(Nbuild)
        pos0 = jax.random.uniform(jax.random.PRNGKey(0), (Nbuild,3), minval=0, maxval=1, dtype=jnp.float32).block_until_ready()
        posz = jz.tree.pos_zorder_sort.jit(pos0)[0]
        data = jz.knn.prepare_knn.jit(posz, k=k)

        timer.timeit_jit(jz.knn.evaluate_knn_z.jit, data, posz_query=poszQ, name=f"Nbuild={Nbuild:.1e}")

timer.plot_timings("Nquery")
plt.title(f"Query Timings (k={k})")
plt.savefig("logs/knn_query.pdf")
# plt.show()
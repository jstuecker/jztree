import jax
import jax.numpy as jnp
import custom_jax as cj
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from fmdj.utility import Tee, Timer
import sys
import numpy as np

boxsize = 0.
k = 16
N = 1024*1024
read = False

sys.stdout = Tee(sys.stdout, open("logs/knn_ilist.log", "a+"))

print(f"============== Lowest level Ilist build (1024*1024), compress dense ilist ==============")
timer = Timer(verbose=True, loops=500, print_compile=False, print_warmup=False)
timer.set_tag(N=N, k=k)

pos0 = jax.random.uniform(jax.random.PRNGKey(0), (N,3), minval=0, maxval=1, dtype=jnp.float32)
posz, idz = cj.tree.pos_zorder_sort.jit(pos0)


spl, nleaf, llvl, xleaf, numleaves = cj.tree.summarize_leaves.jit(posz, max_size=32)
spl2, nleaf2, llvl2, xleaf2, numleaves2 = cj.tree.summarize_leaves.jit(
    xleaf, max_size=32, nleaf=nleaf, num_part=len(posz), ref_fac=8)
il, ir2l2, ispl = cj.knn.build_ilist_recursive.jit(
    xleaf2, llvl2, nleaf2, max_size=32, refine_fac=8, num_part=len(posz), k=16, boxsize=boxsize)

par = (xleaf, llvl, nleaf, spl2, il, ir2l2, ispl)

il, ir2l2, ispl = cj.knn.build_ilist_knn.jit(*par, alloc_fac=256, k=k, boxsize=boxsize)
ispl.block_until_ready()

il2, ir2l2, ispl2 = cj.knn.build_ilist_knn.jit(*par, alloc_fac=256, k=k, boxsize=boxsize)

rnn, inn = cj.knn.ilist_knn_search.jit(posz, spl, il, ir2l2, ispl, k=k, boxsize=boxsize)

tree = cKDTree(np.array(posz), boxsize=boxsize if boxsize > 0 else None)
rknn2, iknn2 = tree.query(posz, k=k)
print(jnp.allclose(rnn, rknn2))
print(jnp.allclose(inn, iknn2), jnp.sum(inn != iknn2))

for rfac in 2, 4, 8, 16, 32, 48:
    timer.set_tag(rfac=rfac)
    spl2, nleaf2, llvl2, xleaf2, numleaves2 = cj.tree.summarize_leaves.jit(
        xleaf, max_size=32*rfac, nleaf=nleaf, num_part=len(posz), ref_fac=rfac)
    if read:
        il = jnp.load(f"logs/tmp/ilist_knn_ilist_{N}_{rfac}.npy")
        ir2l = jnp.load(f"logs/tmp/ilist_knn_r2il_{N}_{rfac}.npy")
        ispl = jnp.load(f"logs/tmp/ilist_knn_ispl_{N}_{rfac}.npy")
    else:
        il, ir2l, ispl = cj.knn.build_ilist_recursive.jit(
            xleaf2, llvl2, nleaf2, max_size=32*rfac, refine_fac=rfac, num_part=len(posz), k=16, boxsize=boxsize)
        jnp.save(f"logs/tmp/ilist_knn_ilist_{N}_{rfac}.npy", np.array(il))
        jnp.save(f"logs/tmp/ilist_knn_r2il_{N}_{rfac}.npy", np.array(ir2l))
        jnp.save(f"logs/tmp/ilist_knn_ispl_{N}_{rfac}.npy", np.array(ispl))
    il.block_until_ready()
    par = (xleaf, llvl, nleaf, spl2, il, ir2l, ispl)
    # timer.timeit_jit(cj.knn.build_ilist_knn.jit, *par, alloc_fac=222, k=k, sort=False, name="nosort")
    timer.timeit_jit(cj.knn.build_ilist_knn.jit, *par, alloc_fac=222, k=k)

    il,ir2l,ispl = cj.knn.build_ilist_knn.jit(*par, alloc_fac=222, k=k, boxsize=boxsize)
    if read:
        ir2lref = jnp.load(f"logs/tmp/ilist_knn_radii_{N}_{rfac}.npy")
        ilref = jnp.load(f"logs/tmp/ilist_knn_ilout_{N}_{rfac}.npy")
        isplref = jnp.load(f"logs/tmp/ilist_knn_isplout_{N}_{rfac}.npy")

        print(jnp.allclose(il[:ispl[-1]], ilref[:ispl[-1]]), 
              jnp.allclose(ispl, isplref),
              jnp.allclose(ir2l[:ispl[-1]], ir2lref[:ispl[-1]], rtol=1e-3))
        # print(radii[0:6], radiiref[0:6])
        # print(ispl[0:7], isplref[0:7])

    else:
        jnp.save(f"logs/tmp/ilist_knn_radii_{N}_{rfac}.npy", np.array(ir2l))
        jnp.save(f"logs/tmp/ilist_knn_ilout_{N}_{rfac}.npy", np.array(il))
        jnp.save(f"logs/tmp/ilist_knn_isplout_{N}_{rfac}.npy", np.array(ispl))


    # timer.timeit_jit(cj.knn.build_ilist_knn.jit, *par, alloc_fac=180, k=k, sort=True, name="sort")
print("")
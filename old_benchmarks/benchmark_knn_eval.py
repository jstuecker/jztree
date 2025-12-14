import jax
import jax.numpy as jnp
import fmdj
import jztree as jz
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from fmdj.utility import Tee, Timer
import sys
import numpy as np

sys.stdout = Tee(sys.stdout, open("logs/knn_eval.log", "a+"))

print(f"============== simplify refinement (sc. 2048, cleanup) ==============")
timer = Timer(verbose=True, loops=100, print_compile=False, print_warmup=False)

boxsize = 0.

def prepare_ilist(N=1024*1024, k=16, max_leaf_size=32):
    pos0 = jax.random.uniform(jax.random.PRNGKey(0), (N,3), minval=0, maxval=1, dtype=jnp.float32)
    posz = jz.tree.pos_zorder_sort.jit(pos0)[0]
    spl, nleaf, llvl, xleaf, numleaves = jz.tree.summarize_leaves.jit(posz, max_size=max_leaf_size)
    il, ir2l, ispl = jz.knn.build_ilist_recursive(
        xleaf, llvl, nleaf, max_size=max_leaf_size, refine_fac=8, num_part=len(posz), k=k, boxsize=boxsize, alloc_fac=256)
    return posz, spl, xleaf, llvl, il, ir2l, ispl

posz, leaf_isplit, leaf_cent, leaf_level, il, ir2l, isplit = prepare_ilist(N=1024*1024, k=16, max_leaf_size=128)
rnn, inn = jz.knn.ilist_knn_search.jit(posz, leaf_isplit, il, ir2l, isplit, k=16, boxsize=boxsize)
tree = cKDTree(posz)
rknn2, iknn2 = tree.query(posz, k=16)

print(jnp.allclose(rnn, rknn2))
print(jnp.allclose(inn, iknn2), jnp.sum(inn != iknn2))

for k in (4,8,12,16,32,64):
    timer.set_tag(N=1024**2, k=k)
    posz, leaf_isplit, leaf_cent, leaf_level, il, ir2l, isplit = prepare_ilist(N=1024*1024, k=k)
    if k >= 16:
        timer.loops = 50

    rnn, inn = timer.timeit_jit(jz.knn.ilist_knn_search.jit, posz, leaf_isplit, il, ir2l, isplit, k=k, boxsize=boxsize)

    # rnn, inn = timer.timeit_jit(jz.knn.knn.jit, posz, k=k, boxsize=boxsize, alloc_fac=256, 
    #                             name="full knn l16", max_leaf_size=16)
    rnn, inn = timer.timeit_jit(jz.knn.knn_old.jit, posz, k=k, boxsize=boxsize, alloc_fac=256, 
                                name="full knn l32", max_leaf_size=32)
    rnn, inn = timer.timeit_jit(jz.knn.knn_old.jit, posz, k=k, boxsize=boxsize, alloc_fac=256, 
                                name="full knn l48", max_leaf_size=48)
    # rnn, inn = timer.timeit_jit(jz.knn.knn.jit, posz, k=k, boxsize=boxsize, alloc_fac=256, 
    #                             name="full knn l64", max_leaf_size=64)
    # rnn, inn = timer.timeit_jit(jz.knn.knn.jit, posz, k=k, boxsize=boxsize, alloc_fac=256, 
    #                             name="full knn l96", max_leaf_size=96)
    # rnn, inn = timer.timeit_jit(jz.knn.knn.jit, posz, k=k, boxsize=boxsize, alloc_fac=256, 
    #                             name="full knn l128", max_leaf_size=128)
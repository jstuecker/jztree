import pytest
import jax
import jax.numpy as jnp
import jztree as jz

import importlib
has_jaxkd = (importlib.util.find_spec("jaxkd") is not None) and (importlib.util.find_spec("jaxkd_cuda") is not None)

@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("npart", [int(1e5), int(1e6), int(3e6), int(3e7)])
def bench_knn_steps(jax_bench, pos):
    k = 16
    cfg = jz.config.KNNConfig()

    jb = jax_bench(jit_rounds=40, jit_warmup=10)

    posz, idz = jb.measure(fn_jit=jz.tree.zsort.jit, x=pos, tag="zsort")[1]
    th = jb.measure(fn_jit=jz.tree.build_tree_hierarchy.jit, tag="tree",
                    part=posz, cfg_tree=cfg.tree)[1]
    
    ilist = jb.measure(fn_jit=jz.knn._knn_dual_walk.jit, tag="treewalk",
                       th=th, k=k, alloc_fac_ilist=cfg.alloc_fac_ilist)[1]
    
    jb.measure(fn_jit=jz.knn._segment_sort.jit, tag="ilist_sort",
               spl=ilist.ispl, key=ilist.rad2, val=ilist.iother)[1]

    rnn, inn = jb.measure(fn_jit=jz.knn._knn_leaf2leaf.jit, tag="leaf2leaf",
                          ilist=ilist, splT=th.splits_leaf_to_part(), xT=posz, k=k)[1]
    inverse = jnp.zeros_like(idz).at[idz].set(jnp.arange(len(idz)))
    def reorder(rnn, inn):
        return jax.tree.map(lambda x: x[inverse], (rnn, inn))
    jb.measure(fn_jit=jax.jit(reorder), tag="reorder", rnn=rnn, inn=inn)
    
    jb.measure(fn_jit=jz.knn.knn.jit, tag="total_z", part=posz, k=k, th=th)
    jb.measure(fn_jit=jz.knn.knn.jit, tag="total", part=pos, k=k)

@pytest.mark.shrink_in_quick(keep_index=5)
@pytest.mark.parametrize("k", [2,8,12,23,32,64,128,220])
def bench_knn_k(jax_bench, pos, k):
    jb = jax_bench(jit_rounds=10, jit_warmup=5)

    jb.measure(fn_jit=jz.knn.knn.jit, part=pos, k=k)

@pytest.mark.shrink_in_quick(keep_index=2)
@pytest.mark.parametrize("N", [1e5,3e5,1e6,3e6,1e7])
def bench_knn_N(jax_bench, N):
    jb = jax_bench(jit_rounds=5, jit_loops=5, jit_warmup=1)

    pos = jax.random.uniform(jax.random.key(0), (int(N), 3), dtype=jnp.float32)

    jb.measure(fn_jit=jz.knn.knn.jit, part=pos, k=4, tag="jztree_k4")
    jb.measure(fn_jit=jz.knn.knn.jit, part=pos, k=16, tag="jztree_k16")

    if has_jaxkd:
        import jaxkd

        def f(pos, k):
            return jaxkd.build_and_query(pos, pos, k=k, cuda=True)
        f.jit = jax.jit(f, static_argnames=["k"])

        jb.measure(fn_jit=f.jit, pos=pos, k=4, tag="jaxkd_cuda_k4")
        jb.measure(fn_jit=f.jit, pos=pos, k=16, tag="jaxkd_cuda_k16")

@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("query_frac", [0.2,1.0,5.0])
def bench_knn_query(jax_bench, query_frac):
    N = int(1e6)
    jb = jax_bench(jit_rounds=5, jit_loops=5, jit_warmup=1)

    cfg = jz.config.KNNConfig(alloc_fac_ilist=300.)

    pos = jax.random.uniform(jax.random.key(0), (int(N), 3), dtype=jnp.float32)
    posq = jax.random.uniform(jax.random.key(0), (int(N*query_frac), 3), dtype=jnp.float32)

    jb.measure(fn_jit=jz.knn.knn.jit, part=pos, part_query=posq, k=16, cfg=cfg, tag="equal_distr")

    posq = jax.random.uniform(
        jax.random.key(0), (int(N*query_frac), 3), dtype=jnp.float32, minval=0.3, maxval=0.8
    )
    jb.measure(fn_jit=jz.knn.knn.jit, part=pos, part_query=posq, k=16, cfg=cfg, tag="diff_distr")

def bench_knn_setup(jax_bench):
    N = 1e6
    jb = jax_bench(jit_rounds=5, jit_loops=5, jit_warmup=1)

    cfg = jz.config.KNNConfig()
    cfg.tree.regularization = None

    print("\nUniform:")
    with jz.stats.statistics() as st:
        pos = jax.random.uniform(jax.random.key(0), (int(N), 3), dtype=jnp.float32)
        jb.measure(fn_jit=jz.knn.knn.jit, part=pos, k=16, tag="uniform")
        st.print_suggestions(cfg)

    print("\nGaus:")
    with jz.stats.statistics() as st:
        pos = jax.random.normal(jax.random.key(0), (int(N), 3), dtype=jnp.float32)
        jb.measure(fn_jit=jz.knn.knn.jit, part=pos, k=16, tag="gaus")
        st.print_suggestions(cfg)

    print("\nGaus (regularized):")
    cfg.tree.regularization = jz.config.RegularizationConfig()
    with jz.stats.statistics() as st:
        pos = jax.random.normal(jax.random.key(0), (int(N), 3), dtype=jnp.float32)
        jb.measure(fn_jit=jz.knn.knn.jit, part=pos, k=16,  tag="gaus(reg)")
        st.print_suggestions(cfg)

@pytest.mark.parametrize("dim", [2,3])
def bench_knn_dtype_dim(jax_bench, dim):
    jb = jax_bench(jit_rounds=5, jit_loops=4, jit_warmup=1)

    N = int(1e6)
    k = 16

    pos = jax.random.uniform(jax.random.key(0), (N,dim))
    jb.measure(fn_jit=jz.knn.knn.jit, part=pos, k=16, tag="float")

    with jax.enable_x64():
        pos = jax.random.uniform(jax.random.key(0), (N,dim), dtype=jnp.float64)
        jb.measure(fn_jit=jz.knn.knn.jit, part=pos, k=16, tag="double")
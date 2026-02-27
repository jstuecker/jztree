import jax
import jax.numpy as jnp
import pytest
import numpy as np
from scipy.spatial import cKDTree
from jztree.config import KNNConfig
from jztree.knn import knn_z, _segment_sort, prepare_knn, evaluate_knn_z, prepare_knn_z, evaluate_knn, knn
from jztree.tree import pos_zorder_sort

def get_pos(N=5555, duplicate=False, xmin=0., xmax=1., seed=1, dim=3, dtype=jnp.float32):
    pos0 = jax.random.uniform(jax.random.PRNGKey(seed), (N,dim), dtype=dtype, minval=xmin, maxval=xmax)
    if duplicate:
        pos0 = jnp.concatenate((pos0, pos0, pos0, pos0))
    
    return pos0

@pytest.mark.skip_in_quick
def test_segment_sort():
    spl = jnp.insert(jnp.sort(jax.random.randint(jax.random.PRNGKey(0), (5000,), 0, 1000000)), 0, 0)
    print(f"\n max seg: {jnp.max(spl[1:] - spl[:-1])}, mean seg: {jnp.mean(spl[1:] - spl[:-1])}")

    r = jax.random.uniform(jax.random.PRNGKey(1), (spl[-1],), minval=0, maxval=1)

    ikey  = jnp.arange(len(r))
    rnew, inew = _segment_sort(spl, r, ikey, smem_size=64)

    # for this test, we can emulate a segmented sort through a lexsort in jax
    # (it takes a factor 10 longer though)
    iseg = jnp.digitize(inew, spl) - 1
    inew2 = jnp.lexsort((r, iseg))

    print(f"Ids different {jnp.sum(inew != inew2)}/{len(inew)} (this is ok, if radii are identical):")
    assert jnp.all(rnew == r[inew2])

def check_against_ckdtree(posz, k=16, boxsize=0.):
    cfg = KNNConfig()
    rnn, inn = knn_z(posz, k=k, boxsize=boxsize, cfg=cfg)

    tree = cKDTree(np.array(posz), boxsize=boxsize)
    rnn2, inn2 = tree.query(np.array(posz), k=k)

    assert jnp.all(rnn[:,1:] >= rnn[:,:-1]), "Radii should be sorted"
    assert jnp.allclose(rnn, rnn2, rtol=1e-4), "Only small differences may arise due to float64 precision in cKDTree"

    degenerate_radii = jnp.sum(rnn[:,1:] == rnn[:,:-1])
    nids_diff = jnp.sum(inn2[:,:-1] != inn[:,:-1])
    print(f"Ids different: {nids_diff}. Expected up to: {degenerate_radii}")

    assert  nids_diff/len(posz) <= 1e-2, "A lot of ids are different (some expected due to degenerate radii)"

@pytest.mark.skip_in_quick
@pytest.mark.parametrize("xmin,xmax", [(-0.3,0.7), (0.1, 0.4), (0.25,0.5), (-1, 0), (0, 1e6), (-1, 1), (-0.5, 1.)])
def test_domain(xmin, xmax):
    posz, idz = pos_zorder_sort.jit(get_pos(N=1024*256, xmin=xmin, xmax=xmax))

    check_against_ckdtree(posz)

@pytest.mark.shrink_in_quick(keep_index=3)
@pytest.mark.parametrize("k", [4,8,12,16,32,64])
def test_k(k):
    posz, idz = pos_zorder_sort.jit(get_pos(N=1024*256, xmin=0., xmax=10.))

    check_against_ckdtree(posz, k=k)

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("dim", [2,3])
def test_dim(dim):
    k = 13
    posz, idz = pos_zorder_sort.jit(get_pos(N=1024*256, xmin=0., xmax=10., dim=dim))

    check_against_ckdtree(posz, k=k)

def test_double():
    k = 13
    with jax.enable_x64():
        posz, idz = pos_zorder_sort.jit(get_pos(N=1024*256, xmin=0., xmax=10., dtype=jnp.float64))
        check_against_ckdtree(posz, k=k)

@pytest.mark.skip_in_quick
@pytest.mark.parametrize("boxsize", [0.03,1.,170.])
def test_boxsize(boxsize):
    posz, idz = pos_zorder_sort.jit(get_pos(N=1024*256, xmin=0., xmax=boxsize))

    check_against_ckdtree(posz, boxsize=boxsize)

@pytest.mark.skip_in_quick
@pytest.mark.parametrize("npart", [1e5, 1e6, 4e6])
def test_npart(npart):
    posz, idz = pos_zorder_sort.jit(get_pos(N=int(npart), xmin=-1., xmax=1.))

    check_against_ckdtree(posz)

@pytest.mark.skip_in_quick
def test_query_skip():
    pos0 = get_pos(1024*128, xmin=0., xmax=1.0)

    data = prepare_knn.jit(pos0, k=16)

    rnnz, innz = evaluate_knn_z.jit(data)
    rnnz2, innz2 = evaluate_knn_z.jit(data, posz_query=data.posz[::2])

    assert jnp.all(rnnz[::2] == rnnz2)
    print(jnp.all(innz[::2] == innz2))

def test_io_order():
    # tests (and demonstrates) the different output/input ordering options
    pos0 = get_pos(1024*128, xmin=0., xmax=1.0)
    posz, idz = pos_zorder_sort.jit(pos0)

    data = prepare_knn.jit(pos0, k=16)
    dataz = prepare_knn_z.jit(posz, k=16)

    rnn00, inn00 = evaluate_knn.jit(data)    # ids and outputs in original order
    rnn0z, inn0z = evaluate_knn_z.jit(data)   # ids original order, outputs in z-order
    rnnzz, innzz = evaluate_knn_z.jit(dataz)  # ids and outputs in z-order

    print("radii only depent on output order:")
    assert jnp.all(rnn00[data.partz.id] == rnn0z)
    assert jnp.all(rnn0z == rnnzz)

    print("ids also depent on input order used to define the data")
    assert jnp.all(inn00[data.partz.id] == inn0z)
    assert jnp.all(inn0z == data.partz.id[innzz])

@pytest.mark.skip_in_quick
def test_twice_knn():
    """Test that running the knn twice works. (Might fail with stream capture problems.)"""
    pos0 = get_pos(1024*128, xmin=0., xmax=1.0)

    def twice_knn(pos0):
        rnn, inn = knn(pos0, k=16)
        pos0 = pos0 + rnn[:,0:1] * 1e-3 # Basically adds zero, since nearest neighbor distance is 0
        rnn, inn = knn(pos0, k=16)
        return rnn, inn
    twice_knn.jit = jax.jit(twice_knn)

    rnn, inn = twice_knn.jit(pos0)

    assert jnp.all(rnn[:,1:] >= rnn[:,:-1]), "Radii should be sorted"

@pytest.mark.skip_in_quick
def test_twice_query():
    """Test that running the knn twice works. (Might fail with stream capture problems.)"""
    pos0 = get_pos(1024*128, xmin=0., xmax=1.0, seed=1)
    posa = get_pos(1024*32, xmin=0., xmax=1.0, seed=2)
    posb = get_pos(1024*32, xmin=0., xmax=1.0, seed=3)

    def twice_knn(pos0, posa, posb):
        rnna, inn = knn(pos0, k=16, pos_query=posa)
        rnnb, inn = knn(pos0, k=16, pos_query=posb)
        return 0.5*(rnna+rnnb), inn
    twice_knn.jit = jax.jit(twice_knn)

    rnn, inn = twice_knn.jit(pos0, posa, posb)

    assert jnp.all(rnn[:,1:] >= rnn[:,:-1]), "Radii should be sorted"

@pytest.mark.skip_in_quick
@pytest.mark.slow
def test_scan_knn():
    pos0 = get_pos(1024*128, xmin=0., xmax=1.0)

    def myknn(pos0, i):
        rnn, inn = knn(pos0, k=16)
        return pos0+1e-3, jnp.mean(rnn)
    myknn.jit = jax.jit(myknn)

    pos, rmean = jax.lax.scan(myknn, pos0, jnp.arange(0,10))

    rmean2 = jnp.mean(knn(pos, k=16)[0])

    assert jnp.allclose(rmean, rmean2)

@pytest.mark.skip_in_quick
def test_scan_query():
    pos0 = get_pos(1024*128, xmin=0., xmax=1.0, seed=1)
    posa = get_pos(1024*32, xmin=0., xmax=1.0, seed=2)
    posb = get_pos(1024*32, xmin=0., xmax=1.0, seed=3)

    def myknn(carry, i):
        pos0, posa, posb = carry
        rnna, inna = knn(pos0, k=16, pos_query=posa)
        rnnb, innb = knn(pos0, k=16, pos_query=posb)
        return (pos0+rnna[0,0], posa+rnnb[0,0], posb+rnnb[0,0]), jnp.mean(rnna + rnnb)
    # myknn.jit = jax.jit(myknn)

    res, rmean = jax.lax.scan(myknn, (pos0, posa, posb), jnp.arange(0,10))
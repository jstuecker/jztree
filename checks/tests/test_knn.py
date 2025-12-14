import jax
import jax.numpy as jnp
import jztree as jz
import pytest
import numpy as np
from scipy.spatial import cKDTree

def test_segment_sort():
    spl = jnp.insert(jnp.sort(jax.random.randint(jax.random.PRNGKey(0), (5000,), 0, 1000000)), 0, 0)
    print(f"\n max seg: {jnp.max(spl[1:] - spl[:-1])}, mean seg: {jnp.mean(spl[1:] - spl[:-1])}")

    r = jax.random.uniform(jax.random.PRNGKey(1), (spl[-1],), minval=0, maxval=1)

    ikey  = jnp.arange(len(r))
    rnew, inew = jz.knn.segment_sort(r, ikey, spl, smem_size=64)

    # for this test, we can emulate a segmented sort through a lexsort in jax
    # (it takes a factor 10 longer though)
    iseg = jnp.digitize(inew, spl) - 1
    inew2 = jnp.lexsort((r, iseg))

    print(f"Ids different {jnp.sum(inew != inew2)}/{len(inew)} (this is ok, if radii are identical):")
    assert jnp.all(rnew == r[inew2])

def get_pos(N=5555, duplicate=False, xmin=0., xmax=1., seed=1):
    pos0 = jax.random.uniform(jax.random.PRNGKey(seed), (N,3), dtype=jnp.float32, minval=xmin, maxval=xmax)
    if duplicate:
        pos0 = jnp.concatenate((pos0, pos0, pos0, pos0))
    
    return pos0

def test_search_sorted_z():
    posz, idz = jz.tree.pos_zorder_sort.jit(get_pos(1387, xmin=0.1, xmax=0.4))
    posz2, idz2 = jz.tree.pos_zorder_sort.jit(get_pos(2222, xmin=0.1, xmax=0.4))

    iself = jz.tree.search_sorted_z.jit(posz, posz)
    assert jnp.all(iself == jnp.arange(len(posz), dtype=jnp.int32))

    i2 = jz.tree.search_sorted_z.jit(posz, posz2)
    pos_ins = jnp.insert(posz, i2, posz2, axis=0)
    pos_ins_ref = jz.tree.pos_zorder_sort.jit(pos_ins)[0]
    assert jnp.all(pos_ins == pos_ins_ref), "If indices were right, we should already be in z-order"

def test_leaf_search():
    posz = jz.tree.pos_zorder_sort(get_pos(144387))[0]
    # Create some reduced leaves
    spl, nleaf, llvl, xleaf, numleaves = jz.tree.summarize_leaves.jit(posz, max_size=32)

    # Check whether we can learn the right leaf numbers just from the leaf positions
    ileaf = jz.tree.search_sorted_z(xleaf, posz, leaf_search=True)
    spl2 = jnp.searchsorted(ileaf, jnp.arange(len(xleaf)+1), side="left")

    assert jnp.all(spl == spl2), "Leaf ranges should be identical"

def check_against_ckdtree(posz, k=16, boxsize=0.):
    cfg = jz.knn.KNNConfig()
    rnn, inn = jz.knn.knn_z(posz, k=k, boxsize=boxsize, cfg=cfg)

    tree = cKDTree(np.array(posz), boxsize=boxsize)
    rnn2, inn2 = tree.query(np.array(posz), k=k)

    assert jnp.all(rnn[:,1:] >= rnn[:,:-1]), "Radii should be sorted"
    assert jnp.allclose(rnn, rnn2, rtol=1e-4), "Only small differences may arise due to float64 precision in cKDTree"

    degenerate_radii = jnp.sum(rnn[:,1:] == rnn[:,:-1])
    nids_diff = jnp.sum(inn2[:,:-1] != inn[:,:-1])
    print(f"Ids different: {nids_diff}. Expected up to: {degenerate_radii}")

    assert  nids_diff/len(posz) <= 1e-2, "A lot of ids are different (some expected due to degenerate radii)"

@pytest.mark.parametrize("xmin,xmax", [(0.1, 0.4), (-0.3,0.3), (0.25,0.5), (-1, 0), (0, 1e6), (-1, 1), (-0.5, 1.)])
def test_domain(xmin, xmax):
    posz, idz = jz.tree.pos_zorder_sort.jit(get_pos(N=1024*256, xmin=xmin, xmax=xmax))

    check_against_ckdtree(posz)

@pytest.mark.parametrize("k", [4,8,12,16,32,64])
def test_k(k):
    posz, idz = jz.tree.pos_zorder_sort.jit(get_pos(N=1024*256, xmin=0., xmax=10.))

    check_against_ckdtree(posz, k=k)

@pytest.mark.parametrize("boxsize", [0.03,1.,170.])
def test_boxisze(boxsize):
    posz, idz = jz.tree.pos_zorder_sort.jit(get_pos(N=1024*256, xmin=0., xmax=boxsize))

    check_against_ckdtree(posz, boxsize=boxsize)

@pytest.mark.parametrize("npart", [1e5, 1e6, 4e6])
def test_npart(npart):
    posz, idz = jz.tree.pos_zorder_sort.jit(get_pos(N=int(npart), xmin=-1., xmax=1.))

    check_against_ckdtree(posz)

def test_query_skip():
    pos0 = get_pos(1024*128, xmin=0., xmax=1.0)

    data = jz.knn.prepare_knn.jit(pos0, k=16)

    rnnz, innz = jz.knn.evaluate_knn_z.jit(data)
    rnnz2, innz2 = jz.knn.evaluate_knn_z.jit(data, posz_query=data.posz[::2])

    assert jnp.all(rnnz[::2] == rnnz2)
    print(jnp.all(innz[::2] == innz2))

def test_io_order():
    # tests (and demonstrates) the different output/input ordering options
    pos0 = get_pos(1024*128, xmin=0., xmax=1.0)
    posz, idz = jz.tree.pos_zorder_sort.jit(pos0)

    data = jz.knn.prepare_knn.jit(pos0, k=16)
    dataz = jz.knn.prepare_knn_z_new.jit(posz, k=16)

    rnn00, inn00 = jz.knn.evaluate_knn.jit(data)    # ids and outputs in original order
    rnn0z, inn0z = jz.knn.evaluate_knn_z.jit(data)   # ids original order, outputs in z-order
    rnnzz, innzz = jz.knn.evaluate_knn_z.jit(dataz)  # ids and outputs in z-order

    print("radii only depent on output order:")
    assert jnp.all(rnn00[data.idz] == rnn0z)
    assert jnp.all(rnn0z == rnnzz)

    print("ids also depent on input order used to define the data")
    assert jnp.all(inn00[data.idz] == inn0z)
    assert jnp.all(inn0z == data.idz[innzz])

def test_twice_knn():
    """Test that running the knn twice works. (Might fail with stream capture problems.)"""
    pos0 = get_pos(1024*128, xmin=0., xmax=1.0)

    def twice_knn(pos0):
        rnn, inn = jz.knn.knn(pos0, k=16)
        pos0 = pos0 + rnn[:,0:1] * 1e-3 # Basically adds zero, since nearest neighbor distance is 0
        rnn, inn = jz.knn.knn(pos0, k=16)
        return rnn, inn
    twice_knn.jit = jax.jit(twice_knn)

    rnn, inn = twice_knn.jit(pos0)

    assert jnp.all(rnn[:,1:] >= rnn[:,:-1]), "Radii should be sorted"

def test_twice_query():
    """Test that running the knn twice works. (Might fail with stream capture problems.)"""
    pos0 = get_pos(1024*128, xmin=0., xmax=1.0, seed=1)
    posa = get_pos(1024*32, xmin=0., xmax=1.0, seed=2)
    posb = get_pos(1024*32, xmin=0., xmax=1.0, seed=3)

    def twice_knn(pos0, posa, posb):
        rnna, inn = jz.knn.knn(pos0, k=16, pos_query=posa)
        rnnb, inn = jz.knn.knn(pos0, k=16, pos_query=posb)
        return 0.5*(rnna+rnnb), inn
    twice_knn.jit = jax.jit(twice_knn)

    rnn, inn = twice_knn.jit(pos0, posa, posb)

    assert jnp.all(rnn[:,1:] >= rnn[:,:-1]), "Radii should be sorted"

def test_scan_knn():
    pos0 = get_pos(1024*128, xmin=0., xmax=1.0)

    def myknn(pos0, i):
        rnn, inn = jz.knn.knn(pos0, k=16)
        return pos0+1e-3, jnp.mean(rnn)
    myknn.jit = jax.jit(myknn)

    pos, rmean = jax.lax.scan(myknn, pos0, jnp.arange(0,10))

    rmean2 = jnp.mean(jz.knn.knn(pos, k=16)[0])

    assert jnp.allclose(rmean, rmean2)

def test_scan_query():
    pos0 = get_pos(1024*128, xmin=0., xmax=1.0, seed=1)
    posa = get_pos(1024*32, xmin=0., xmax=1.0, seed=2)
    posb = get_pos(1024*32, xmin=0., xmax=1.0, seed=3)

    def myknn(carry, i):
        pos0, posa, posb = carry
        rnna, inna = jz.knn.knn(pos0, k=16, pos_query=posa)
        rnnb, innb = jz.knn.knn(pos0, k=16, pos_query=posb)
        return (pos0+rnna[0,0], posa+rnnb[0,0], posb+rnnb[0,0]), jnp.mean(rnna + rnnb)
    # myknn.jit = jax.jit(myknn)

    res, rmean = jax.lax.scan(myknn, (pos0, posa, posb), jnp.arange(0,10))
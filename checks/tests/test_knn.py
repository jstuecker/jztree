import jax
import jax.numpy as jnp
import pytest
import numpy as np
from scipy.spatial import cKDTree
from jztree.config import KNNConfig
import jztree as jz
from jztree_utils import ics

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
    rnew, inew = jz.knn._segment_sort(spl, r, ikey, smem_size=64)

    # for this test, we can emulate a segmented sort through a lexsort in jax
    # (it takes a factor 10 longer though)
    iseg = jnp.digitize(inew, spl) - 1
    inew2 = jnp.lexsort((r, iseg))

    print(f"Ids different {jnp.sum(inew != inew2)}/{len(inew)} (this is ok, if radii are identical):")
    assert jnp.all(rnew == r[inew2])

def check_against_ckdtree(pos, k=16, boxsize=0., pos_query=None):
    cfg = KNNConfig()

    rnn, inn = jz.knn.knn(pos, k=k, boxsize=boxsize, result="rad_globalidx", part_query=pos_query)

    tree = cKDTree(np.array(pos), boxsize=boxsize)
    if pos_query is None:
        pos_query = pos
    rnn2, inn2 = tree.query(np.array(pos_query), k=k)

    assert jnp.all(rnn[:,1:] >= rnn[:,:-1]), "Radii should be sorted"
    assert jnp.allclose(rnn, rnn2, rtol=1e-4), "Only small differences may arise due to float64 precision in cKDTree"

    degenerate_radii = jnp.sum(rnn[:,1:] == rnn[:,:-1])
    nids_diff = jnp.sum(inn2[:,:-1] != inn[:,:-1])
    print(f"Ids different: {nids_diff}. Expected up to: {degenerate_radii}")

    assert  nids_diff/len(pos_query) <= 1e-2, "A lot of ids are different (some expected due to degenerate radii)"

@pytest.mark.skip_in_quick
@pytest.mark.parametrize("xmin,xmax", [(-0.3,0.7), (0.1, 0.4), (0.25,0.5), (-1, 0), (0, 1e6), (-1, 1), (-0.5, 1.)])
def test_domain(xmin, xmax):
    posz, idz = jz.tree.zsort.jit(get_pos(N=1024*256, xmin=xmin, xmax=xmax))

    check_against_ckdtree(posz)

# @pytest.mark.shrink_in_quick(keep_index=6)
@pytest.mark.parametrize("k", [4,8,12,16,32,53,133])
def test_k(k):
    posz, idz = jz.tree.zsort.jit(get_pos(N=1024*256, xmin=0., xmax=10.))

    check_against_ckdtree(posz, k=k)

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("dim", [2,3])
def test_dim(dim):
    k = 13
    posz, idz = jz.tree.zsort.jit(get_pos(N=1024*256, xmin=0., xmax=10., dim=dim))

    check_against_ckdtree(posz, k=k)

def test_double():
    k = 13
    with jax.enable_x64():
        posz, idz = jz.tree.zsort.jit(get_pos(N=1024*256, xmin=0., xmax=10., dtype=jnp.float64))
        check_against_ckdtree(posz, k=k)

@pytest.mark.skip_in_quick
@pytest.mark.parametrize("boxsize", [0.03,1.,170.])
def test_boxsize(boxsize):
    posz, idz = jz.tree.zsort.jit(get_pos(N=1024*256, xmin=0., xmax=boxsize))

    check_against_ckdtree(posz, boxsize=boxsize)

@pytest.mark.skip_in_quick
@pytest.mark.parametrize("npart", [1e5, 1e6, 4e6])
def test_npart(npart):
    posz, idz = jz.tree.zsort.jit(get_pos(N=int(npart), xmin=-1., xmax=1.))

    check_against_ckdtree(posz)

def test_query():
    pos = get_pos(N=int(1e5), xmin=-1., xmax=1.)
    posq = get_pos(N=int(1.3e5), xmin=-0.5, xmax=0.5)

    check_against_ckdtree(pos, pos_query=posq)

@pytest.mark.parametrize("boxsize", [None, 1.0])
@pytest.mark.parametrize("query_n", [None, 32], ids=["self", "part_query"])
@pytest.mark.parametrize("k", [8, 64, 117])
def test_lattice_ties(k, query_n, boxsize):
    # Dyadic coordinates give exact float32 distance ties. k > 32 also tests
    # ties spanning multiple passes of KnnLeaf2LeafKernel.
    def lattice(n):
        g = (np.arange(n, dtype=np.float32) + 0.5) / n
        x, y = np.meshgrid(g, g, indexing="ij")
        return np.column_stack((x.ravel(), y.ravel()))

    pos = lattice(16)
    pos_query = None if query_n is None else lattice(query_n)
    query = pos if pos_query is None else pos_query
    radii, indices = jz.knn.knn(
        jnp.asarray(pos), k=k, boxsize=boxsize,
        part_query=None if pos_query is None else jnp.asarray(pos_query),
    )
    radii, indices = np.asarray(radii), np.asarray(indices)

    assert indices.shape == (len(query), k)
    assert np.all((indices >= 0) & (indices < len(pos)))
    assert np.all(np.diff(np.sort(indices, axis=1), axis=1) > 0), \
        "Each row must contain k distinct particle indices"

    reference_radii, _ = cKDTree(pos, boxsize=boxsize).query(query, k=k)
    assert np.all(np.diff(radii, axis=1) >= 0), "Radii must be sorted"
    np.testing.assert_allclose(radii, reference_radii, rtol=1e-6, atol=1e-7)

    # Check indices as well as radii; matching distances alone cannot detect
    # duplicates on a lattice. Do not prescribe which tied cutoff IDs to use.
    delta = pos[None, :, :].astype(np.float64) - query[:, None, :]
    if boxsize is not None:
        delta -= boxsize * np.rint(delta / boxsize)
    distances = np.sqrt(np.sum(delta**2, axis=-1))
    selected_distances = np.take_along_axis(distances, indices, axis=1)
    np.testing.assert_allclose(radii, selected_distances, rtol=1e-6, atol=1e-7)

    selected = np.zeros(distances.shape, dtype=bool)
    np.put_along_axis(selected, indices, True, axis=1)
    strictly_closer = distances < reference_radii[:, -1:]
    assert np.all(selected | ~strictly_closer), \
        "All neighbours strictly inside the k-th radius must be returned"

def test_io_order():
    # tests (and demonstrates) the different output/input ordering options
    part = ics.uniform_particles(int(1024*128))
    partz, idz, th = jz.tree.zsort_and_tree(part, jz.config.KNNConfig().tree, data=jnp.arange(len(part.pos)))

    rnn00, inn00 = jz.knn.knn.jit(part, k=16)
    rnn0z, inn0z = jz.knn.knn.jit(part, k=16, output_order="z")
    rnnzz, innzz = jz.knn.knn.jit(partz, k=16, th=th)

    print("radii only depent on output order:")
    assert jnp.all(rnn00[idz] == rnn0z)
    assert jnp.all(rnn0z == rnnzz)

    print("ids also depent on input order used to define the data")
    assert jnp.all(inn00[idz] == idz[innzz])
    assert jnp.all(inn0z == innzz)

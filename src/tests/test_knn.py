import jax
import jax.numpy as jnp
import custom_jax as cj
import pytest
import numpy as np
from scipy.spatial import cKDTree

def test_segment_sort():
    spl = jnp.insert(jnp.sort(jax.random.randint(jax.random.PRNGKey(0), (5000,), 0, 1000000)), 0, 0)
    print(f"\n max seg: {jnp.max(spl[1:] - spl[:-1])}, mean seg: {jnp.mean(spl[1:] - spl[:-1])}")

    r = jax.random.uniform(jax.random.PRNGKey(1), (spl[-1],), minval=0, maxval=1)

    ikey  = jnp.arange(len(r))
    rnew, inew = cj.knn.segment_sort(r, ikey, spl, smem_size=64)

    # for this test, we can emulate a segmented sort through a lexsort in jax
    # (it takes a factor 10 longer though)
    iseg = jnp.digitize(inew, spl) - 1
    inew2 = jnp.lexsort((r, iseg))

    print(f"Ids different {jnp.sum(inew != inew2)}/{len(inew)} (this is ok, if radii are identical):")
    assert jnp.all(rnew == r[inew2])

def get_pos(N=5555, duplicate=False, xmin=0., xmax=1.):
    pos0 = jax.random.uniform(jax.random.PRNGKey(1), (N,3), dtype=jnp.float32, minval=xmin, maxval=xmax)
    if duplicate:
        pos0 = jnp.concatenate((pos0, pos0, pos0, pos0))
    
    return pos0

# @pytest.mark.parametrize("final_size", [13, 33, 64, 77, 135, 255, 256, 299, 317, 339, 411, 415, 477])
@pytest.mark.parametrize("xmin,xmax", [(0.1, 0.4), (0.5,1.0), (-1, -0.5), (0, 1e6), (-1, 1), (-0.5, 1.)])
def test_summarize_identity(xmin, xmax):
    print("")
    posz, idz = cj.tree.pos_zorder_sort.jit(get_pos(144387, xmin=xmin, xmax=xmax))
    spl, nleaf, llvl, xleaf, numleaves = cj.tree.summarize_leaves.jit(
        posz, max_size=1)
    
    xleafz, idz2 = cj.tree.pos_zorder_sort.jit(xleaf)
    
    print("First wrong:", jnp.where(idz2 != jnp.arange(len(xleaf), dtype=jnp.int32))[0][0:10])

    assert numleaves == len(posz)
    assert jnp.all(spl[:numleaves] == jnp.arange(numleaves, dtype=jnp.int32))
    assert jnp.all(idz2 == jnp.arange(len(xleaf), dtype=jnp.int32))
    assert jnp.all(nleaf[:numleaves] == 1)
    assert jnp.all(nleaf[numleaves:] == 0)

@pytest.mark.parametrize("final_size", [4, 6, 13, 33, 63, 135, 317, 477])
def test_double_summarize(final_size):
    posz, idz = cj.tree.pos_zorder_sort.jit(get_pos(N=1024*128, xmin=0., xmax=1.0))
    spl_ref, nleaf_ref, llvl_ref, xleaf_ref, numleaves_ref = cj.tree.summarize_leaves.jit(
        posz, max_size=final_size)

    im_size = np.maximum(final_size // 7, 1)
    spl, nleaf, llvl, xleaf, numleaves = cj.tree.summarize_leaves.jit(
        posz, max_size=im_size)
    spl, nleaf, llvl, xleaf, numleaves = cj.tree.summarize_leaves.jit(
        xleaf, max_size=final_size, nleaf=nleaf, num_part=len(posz), ref_fac=final_size / im_size)
    
    assert jnp.all(llvl_ref[:numleaves_ref] < 386), "Leaf levels should be reasonable"

    print("\nfirst wrong:", jnp.where(nleaf != nleaf_ref)[0][:2])
    assert jnp.all(nleaf_ref <= final_size)
    assert jnp.all(nleaf <= final_size)
    assert jnp.all(nleaf == nleaf_ref)
    assert jnp.all(xleaf[:numleaves] == xleaf_ref[:numleaves_ref])
    assert numleaves == numleaves_ref

@pytest.mark.parametrize("rfac", [2, 8, 17, 32, 33, 40, 93])
def test_ilist_rfac(rfac):
    N = 1024*512
    xmin, xmax = 0.1, 0.4
    posz, idz = cj.tree.pos_zorder_sort.jit(get_pos(N, xmin=xmin, xmax=xmax))

    msize = 48
    spl, nleaf, llvl, xleaf, numleaves = cj.knn.summarize_leaves(posz, max_size=msize)

    rfacA = 15
    il, ir2l, ispl = cj.knn.build_ilist_recursive.jit(xleaf, llvl, nleaf, max_size=msize*rfacA, 
        refine_fac=rfacA, num_part=len(posz), k=16)
    
    il2, ir2l2, ispl2 = cj.knn.build_ilist_recursive.jit(xleaf, llvl, nleaf, max_size=msize*rfac, refine_fac=rfac,
                                                num_part=len(posz), k=16)

    print(jnp.where(ispl[1:numleaves] <= ispl[:numleaves-1])[0][0:10])
    assert jnp.all(ispl[1:numleaves] > ispl[:numleaves-1]), "should not have empty list for any leaf"
    assert jnp.all(ir2l <= 3.*(xmax-xmin)**2)
    assert jnp.all(ispl2 == ispl), f"Splits different for rfac {rfacA} and {rfac}"
    assert jnp.all(ir2l2[:ispl2[-1]] == ir2l[:ispl[-1]]), f"Radii different for rfac {rfacA} and {rfac}"

    # Note the ids can differ for identical radii:
    print(f"Fraction ids equal {jnp.mean(il2[:ispl2[-1]] == il[:ispl[-1]])}") 

@pytest.mark.parametrize("xmin,xmax", [(0.1, 0.4), (-0.3,0.3), (0.25,0.5), (-1, 0), (0, 1e6), (-1, 1), (-0.5, 1.)])
def test_domain(xmin, xmax):
    posz, idz = cj.tree.pos_zorder_sort.jit(get_pos(N=1024*512, xmin=xmin, xmax=xmax))

    rnn, inn = cj.knn.knn.jit(posz, k=16)

    tree = cKDTree(np.array(posz))
    rnn2, inn2 = tree.query(np.array(posz), k=16)

    assert jnp.allclose(rnn, rnn2), "Inferred radii differ. This should never happen for exact knn"

    degenerate_radii = jnp.sum(rnn[:,1:] == rnn[:,:-1])
    nids_diff = jnp.sum(inn2[:,:-1] != inn[:,:-1])
    print(f"Ids different: {nids_diff}. Expected up to: {degenerate_radii}")

    assert  nids_diff <= degenerate_radii, "Ids are different (not explained by degenerate radii)"
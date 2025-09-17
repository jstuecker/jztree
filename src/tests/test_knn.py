import jax
import jax.numpy as jnp
import custom_jax as cj
import pytest
import numpy as np

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

def get_pos(N=5555, duplicate=False):
    pos0 = jax.random.uniform(jax.random.PRNGKey(1), (N,3), dtype=jnp.float32, minval=0., maxval=2.)
    if duplicate:
        pos0 = jnp.concatenate((pos0, pos0, pos0, pos0))
    
    return pos0

# @pytest.mark.parametrize("final_size", [13, 33, 64, 77, 135, 255, 256, 299, 317, 339, 411, 415, 477])
def test_summarize_identity():
    print("")
    posz, idz = cj.tree.pos_zorder_sort.jit(get_pos(144387) - 0.5)
    spl, nleaf, llvl, xleaf, numleaves = cj.tree.summarize_leaves.jit(
        posz, max_size=1)
    
    xleafz, idz2 = cj.tree.pos_zorder_sort.jit(xleaf)
    
    print("First wrong:", jnp.where(idz2 != jnp.arange(len(xleaf), dtype=jnp.int32))[0][0:10])

    assert jnp.all(spl == jnp.arange(len(spl), dtype=jnp.int32))
    assert jnp.all(idz2 == jnp.arange(len(xleaf), dtype=jnp.int32))

@pytest.mark.parametrize("final_size", [13, 33, 39, 43, 63, 64, 77, 135, 255, 256, 299, 317, 339, 411, 415, 477])
def test_double_summarize(final_size):
    print("")
    pos0, mass0 = get_pos(144387)
    posz, idz = cj.tree.pos_zorder_sort.jit(pos0)
    spl_ref, nleaf_ref, llvl_ref, xleaf_ref, numleaves_ref = cj.tree.summarize_leaves.jit(
        posz, max_size=final_size)

    im_size = np.maximum(final_size // 7, 1)
    spl, nleaf, llvl, xleaf, numleaves = cj.tree.summarize_leaves.jit(
        posz, max_size=im_size)
    spl, nleaf, llvl, xleaf, numleaves = cj.tree.summarize_leaves.jit(
        xleaf, max_size=final_size, nleaf=nleaf, num_part=len(posz), ref_fac=final_size / im_size)
    
    print("first wrong:", jnp.where(nleaf != nleaf_ref)[0][:2])
    assert jnp.all(nleaf == nleaf_ref)
    assert jnp.all(xleaf == xleaf_ref)
    assert numleaves == numleaves_ref

def test_ilist_rfac():
    N = 1024*177
    posz, idz = cj.tree.pos_zorder_sort.jit(get_pos(N))

    msize = 64
    spl, nleaf, llvl, xleaf, numleaves = cj.knn.summarize_leaves(posz, max_size=msize)

    rfacA = 15
    il, ir2l, ispl = cj.knn.build_ilist_recursive.jit(xleaf, llvl, nleaf, max_size=msize*rfacA, 
        refine_fac=rfacA, num_part=len(posz), k=16)
    
    for rfacB in 2,4,8,16,31,:
        il2, ir2l2, ispl2 = cj.knn.build_ilist_recursive.jit(xleaf, llvl, nleaf, max_size=msize*rfacB, refine_fac=rfacB,
                                                    num_part=len(posz), k=16)

        assert jnp.all(ispl2 == ispl), f"Splits different for rfac {rfacA} and {rfacB}"
        assert jnp.all(ir2l2[:ispl2[-1]] == ir2l[:ispl[-1]]), f"Radii different for rfac {rfacA} and {rfacB}"

    # Note the ids can differ for identical radii:
    print(f"Fraction ids equal {jnp.mean(il2[:ispl2[-1]] == il[:ispl[-1]])}") 
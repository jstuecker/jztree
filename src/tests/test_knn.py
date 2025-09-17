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

def setup_particles(N=5555, duplicate=False):
    pos0 = jax.random.uniform(jax.random.PRNGKey(1), (N,3), dtype=jnp.float32, minval=0., maxval=2.)
    if duplicate:
        pos0 = jnp.concatenate((pos0, pos0, pos0, pos0))
    mass0 = jnp.ones(len(pos0), dtype=jnp.float32)
    
    return pos0, mass0

# @pytest.mark.parametrize("final_size", [13, 33, 64, 77, 135, 255, 256, 299, 317, 339, 411, 415, 477])
def test_summarize_identity():
    print("")
    pos0, mass0 = setup_particles(144387)
    pos0 = pos0 - 0.5
    posz, idz = cj.tree.pos_zorder_sort.jit(pos0)
    spl, nleaf, llvl, xleaf, numleaves = cj.tree.summarize_leaves.jit(
        posz, max_size=1)
    
    xleafz, idz2 = cj.tree.pos_zorder_sort.jit(xleaf)
    
    print("First wrong:", jnp.where(idz2 != jnp.arange(len(xleaf), dtype=jnp.int32))[0][0:10])

    assert jnp.all(spl == jnp.arange(len(spl), dtype=jnp.int32))
    assert jnp.all(idz2 == jnp.arange(len(xleaf), dtype=jnp.int32))

@pytest.mark.parametrize("final_size", [13, 33, 39, 43, 63, 64, 77, 135, 255, 256, 299, 317, 339, 411, 415, 477])
def test_double_summarize(final_size):
    print("")
    pos0, mass0 = setup_particles(144387)
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
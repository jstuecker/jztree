import jax
import jax.numpy as jnp
import custom_jax as cj


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
    pos0 = jax.random.uniform(jax.random.PRNGKey(0), (N,3), dtype=jnp.float32, minval=-0.5, maxval=0.5)
    if duplicate:
        pos0 = jnp.concatenate((pos0, pos0, pos0, pos0))
    mass0 = jnp.ones(len(pos0), dtype=jnp.float32)
    
    return pos0, mass0

def test_double_summarize():
    pos0, mass0 = setup_particles()
    posz, idz = cj.tree.pos_zorder_sort.jit(pos0)
    spl_ref, nleaf_ref, llvl_ref, xleaf_ref, numleaves_ref = cj.tree.summarize_leaves.jit(posz, max_size=13)

    spl, nleaf, llvl, xleaf, numleaves = cj.tree.summarize_leaves.jit(posz, max_size=6)
    spl, nleaf, llvl, xleaf, numleaves = cj.tree.summarize_leaves.jit(xleaf, max_size=13, nleaf=nleaf, num_part=len(posz))
    
    assert jnp.all(nleaf == nleaf_ref)
    assert jnp.all(xleaf == xleaf_ref)
    assert numleaves == numleaves_ref

    spl_ref, nleaf_ref, llvl_ref, xleaf_ref, numleaves_ref = cj.tree.summarize_leaves.jit(posz, max_size=64)
    spl, nleaf, llvl, xleaf, numleaves = cj.tree.summarize_leaves.jit(xleaf, max_size=64, nleaf=nleaf, num_part=len(posz))

    assert jnp.all(nleaf == nleaf_ref)
    assert jnp.all(xleaf == xleaf_ref)
    assert numleaves == numleaves_ref
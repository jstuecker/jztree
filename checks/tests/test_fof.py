import pytest
import jztree.fof
import jax
import jax.numpy as jnp
from hfof import fof
import pytest

def test_against_hfof():
    boxsize = 1.0

    pos = jax.random.uniform(jax.random.PRNGKey(0), (1000000, 3), minval=0.0, maxval=boxsize)

    rlink = 0.8 * boxsize / len(pos)**(1/3)

    igr_jz = jztree.fof.fof.jit(pos, rlink=rlink, boxsize=boxsize)
    igr_hfof = fof(pos, rlink, boxsize=boxsize)

    # uniquely map every jzfof-label to an hfof-label
    label_map = jnp.zeros(len(pos), dtype=jnp.int32).at[igr_jz].set(igr_hfof)
    label_map_rev = jnp.arange(len(pos), dtype=jnp.int32).at[igr_hfof].set(igr_jz)

    igr_hfof_jz = label_map[igr_jz]
    igr_jz_hfof = label_map_rev[igr_hfof]

    assert igr_hfof_jz == pytest.approx(igr_hfof)
    assert igr_jz_hfof == pytest.approx(igr_jz)

    group_sizes_jz = jnp.sort(jnp.bincount(igr_jz, minlength=len(pos)))[::-1]
    group_sizes_hfof = jnp.sort(jnp.bincount(igr_hfof, minlength=len(pos)))[::-1]

    # print(group_sizes_jz[0:10])

    assert group_sizes_jz == pytest.approx(group_sizes_hfof)

import pytest
from jztree.config import FofConfig
from jztree.ztree import pos_zorder_sort
from jztree.fof import fof_z
import jax
import jax.numpy as jnp
import pytest
import importlib

has_hfof = importlib.util.find_spec("hfof") is not None

# Define a setup
boxsize = 1.0
pos = jax.random.uniform(jax.random.PRNGKey(0), (int(1e6), 3), minval=0.0, maxval=boxsize)
rlink = 0.8 * boxsize / len(pos)**(1/3)
cfg = FofConfig()

# run jztree-fof
posz, idz = pos_zorder_sort.jit(pos)
igr_jz = fof_z.jit(posz, rlink, boxsize=boxsize, cfg=cfg)

# check outputs
group_sizes_jz = jnp.sort(jnp.bincount(igr_jz, minlength=len(posz)))[::-1]

print("group sizes:", group_sizes_jz[0:10])
print("(Should be [329 321 309 281 269 260 258 238 223 219])")

if has_hfof:
    print("Comparing to hfof")
    from hfof import fof
    igr_hfof = fof(posz, rlink, boxsize=boxsize)

    # uniquely map every jzfof-label to an hfof-label
    label_map = jnp.zeros(len(posz), dtype=jnp.int32).at[igr_jz].set(igr_hfof)
    label_map_rev = jnp.arange(len(posz), dtype=jnp.int32).at[igr_hfof].set(igr_jz)

    igr_hfof_jz = label_map[igr_jz]
    igr_jz_hfof = label_map_rev[igr_hfof]

    group_sizes_hfof = jnp.sort(jnp.bincount(igr_hfof, minlength=len(posz)))[::-1]

    print("Group sizes consitent:", jnp.all(group_sizes_jz == pytest.approx(group_sizes_hfof)))
    print("Group labels consistent:", jnp.all(igr_hfof_jz == igr_hfof), jnp.all(igr_jz_hfof == igr_jz))
else:
    print("Please install hfof for comparison")
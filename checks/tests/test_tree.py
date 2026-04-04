import jax.numpy as jnp
import jax
from jztree.data import PosMass, TreeHierarchy
import pytest
import numpy.testing as npt
from jztree.config import TreeConfig
from jztree.tree import get_node_geometry, search_sorted_z, zsort, build_tree_hierarchy
from jztree_utils import ics
import jztree as jz

def test_tree_hierarchy(tree_hierarchy : TreeHierarchy):
    th = tree_hierarchy

    nleaves = jnp.argmax(th.ispl_n2l.get(0))
    for lvl in range(1, th.num_planes()):
        in2l = th.ispl_n2l.get(lvl)
        in2n = th.ispl_n2n.get(lvl)

        assert jnp.all(in2l[1:] >= in2l[:-1])
        assert jnp.all(in2n[1:] >= in2n[:-1])
        
        assert jnp.max(in2l) == nleaves
        assert jnp.max(in2n) == jnp.argmax(th.ispl_n2n.get(lvl-1))
    
    npart = jnp.sum(th.npart(0))
    for i in range(th.num_planes()):
        assert jnp.sum(th.npart(i)) == npart
        lvls = th.lvl.get(i, size=th.num(i))
        assert jnp.all((lvls >= -450 ) & (lvls < 100))

def test_node_geometry(pos_mass_z: PosMass):
    pos = pos_mass_z.pos
    pos = pos[::500]
    ispl = jnp.sort(jax.random.randint(jax.random.PRNGKey(42), (50,), 0, pos.shape[0]))

    cent, ext = get_node_geometry(pos, ispl[:-1], ispl[1:], result="cent_ext")

    for i in range(len(ispl)-1):
        if ispl[i+1] - ispl[i] <= 1:
            continue
        pnode = pos[ispl[i]:ispl[i+1]]

        pmin, pmax = jnp.min(pnode, axis=0), jnp.max(pnode, axis=0)
                
        npt.assert_array_less(cent[i] - ext[i]*0.5, pmin)
        npt.assert_array_less(pmax, cent[i] + ext[i]*0.5)

        # Check that node is not too big, by seeing that half the extent would not be sufficient
        if jnp.all(jnp.isfinite(ext[i])):
            assert jnp.any(cent[i] - ext[i]*0.25 >= pmin) or jnp.any(cent[i] + ext[i]*0.25 <= pmax)

def test_multi_type_tree():
    n1, n2 = int(1e5), int(2e5)
    p1 = ics.uniform_particles(n1)
    p2 = ics.uniform_particles(n2)

    (p1z, p2z), th = jz.tree.zsort_and_tree_multi_type.jit((p1,p2), cfg_tree=jz.config.KNNConfig().tree)

    ispl_t = th.ispl_l2p_per_type
    assert jnp.all(ispl_t[0,1:] >= ispl_t[0,:-1])
    assert jnp.all(ispl_t[1,1:] >= ispl_t[1,:-1])
    assert ispl_t[0,-1] == n1
    assert ispl_t[1,-1] == n2

    nleaves = th.num(0)
    cent = th.geom_cent.get(0, nleaves)
    ext = jz.tree._lvl_to_ext(th.lvl.get(0, nleaves), th.info())

    ileaf1 = jz.tools.inverse_of_splits(ispl_t[0], n1)
    ileaf2 = jz.tools.inverse_of_splits(ispl_t[1], n2)

    assert jnp.all(p1z.pos >= cent[ileaf1] - 0.5*ext[ileaf1])
    assert jnp.all(p1z.pos <= cent[ileaf1] + 0.5*ext[ileaf1])

    assert jnp.all(p2z.pos >= cent[ileaf2] - 0.5*ext[ileaf2])
    assert jnp.all(p2z.pos <= cent[ileaf2] + 0.5*ext[ileaf2])
    
def get_pos(N=5555, xmin=0., xmax=1., seed=1):
    pos0 = jax.random.uniform(
        jax.random.PRNGKey(seed), (N,3), dtype=jnp.float32, minval=xmin, maxval=xmax
    )
    
    return pos0

def test_zsort_infinities():
    xin = jnp.array([
        [0.1, 0.1, 0.5],
        [jnp.nan, jnp.nan, jnp.nan],
        [0.5, 0.1, 0.1],
        [jnp.inf, 0.2, jnp.inf],
        [-0.7, jnp.inf, 0.],
        [-jnp.inf, -jnp.inf, -jnp.inf],
        [0.3, 0.3, 0.3]
    ])
    xout = jnp.array([
        [-jnp.inf, -jnp.inf, -jnp.inf],
        [-0.7,  jnp.inf,  0. ],
        [ 0.3,  0.3,  0.3],
        [ 0.1,  0.1,  0.5],
        [ 0.5,  0.1,  0.1],
        [ jnp.inf,  0.2,  jnp.inf],
        [ jnp.nan,  jnp.nan,  jnp.nan]
    ])

    xz = zsort.jit(xin)[0]

    assert jnp.all(xout[:-1] == xz[:-1])
    assert jnp.all(jnp.isnan(xz[-1]))  # have to split of nan comparison since nan != nan

def test_search_sorted_z():
    posz, idz = zsort.jit(get_pos(1387, xmin=0.1, xmax=0.4, seed=0))
    posz2, idz2 = zsort.jit(get_pos(2222, xmin=0.1, xmax=0.4, seed=2))

    iself = search_sorted_z.jit(posz, posz)
    assert jnp.all(iself == jnp.arange(len(posz), dtype=jnp.int32))

    i2 = search_sorted_z.jit(posz, posz2)
    pos_ins = jnp.insert(posz, i2, posz2, axis=0)
    pos_ins_ref = zsort.jit(pos_ins)[0]
    assert jnp.all(pos_ins == pos_ins_ref), "If indices were right, we should already be in z-order"

def test_leaf_search(pos_mass_z: PosMass, tree_hierarchy: TreeHierarchy):
    # Check whether we can learn the right leaf numbers just from the leaf positions
    nleaves = tree_hierarchy.lvl.num(0)
    xleaf = tree_hierarchy.geom_cent.get(0, nleaves)
    spl = tree_hierarchy.splits_leaf_to_part(size=nleaves+1)

    ileaf = search_sorted_z(xleaf, pos_mass_z.pos, leaf_search=True)
    spl2 = jnp.searchsorted(ileaf, jnp.arange(len(xleaf)+1), side="left")

    print("lvl", tree_hierarchy.lvl.get(0, 20))

    assert jnp.all(spl == spl2), "Leaf ranges should be identical"

def test_tree_nans(pos_mass_z: PosMass):
    cfg_tree = TreeConfig(mass_centered=True)
    npart = len(pos_mass_z.pos)
    nextra = 1356

    pos_mass_z2 = PosMass(
        pos = jnp.pad(pos_mass_z.pos, ((0,nextra),(0,0)), constant_values=jnp.nan),
        mass = jnp.pad(pos_mass_z.mass, (0, nextra), constant_values=jnp.nan),
        num_total=npart
    )

    th1 = build_tree_hierarchy(pos_mass_z, cfg_tree)
    th2 = build_tree_hierarchy(pos_mass_z2, cfg_tree)

    assert th1.ispl_n2l.all_equal(th2.ispl_n2l)
    assert th1.ispl_n2n.all_equal(th2.ispl_n2n)
    assert th1.geom_cent.all_equal(th2.geom_cent)
    assert th1.lvl.all_equal(th2.lvl)
    assert th1.mass.all_equal(th2.mass)
    assert th1.mass_cent.all_equal(th2.mass_cent)
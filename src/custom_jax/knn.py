import numpy as np
import jax
import jax.numpy as jnp
import custom_jax.nb_knn as nb_knn

jax.ffi.register_ffi_target("IlistKNNSearch", nb_knn.IlistKNNSearch(), platform="CUDA")
jax.ffi.register_ffi_target("ConstructIlist", nb_knn.ConstructIlist(), platform="CUDA")

def lvl_to_ext(level_binary):
    olvl, omod = level_binary//3, level_binary % 3
    levels_3d = jnp.stack((olvl, olvl + (omod >= 2).astype(jnp.int32), olvl + (omod >= 1).astype(jnp.int32)),axis=-1)
    return 2.**levels_3d

def get_node_box(x, level_binary):
    node_size = lvl_to_ext(level_binary)
    node_cent = (jnp.floor(x / node_size) + 0.5) * node_size
    return node_cent, node_size

def ilist_knn_search(xT, isplitT, xleaf, lvl_leaf, ilist, ilist_splitsB, xQ=None,  isplitQ=None, k=32, boxsize=0.):
    """Finds the k nearest neighbors of xfind in the z-sorted positions xzsort
    """
    if xQ is None: xQ = xT
    if isplitQ is None: isplitQ = isplitT

    assert xT.dtype == xQ.dtype == jnp.float32
    assert xT.shape[-1] == xQ.shape[-1] == xleaf.shape[-1] == 3
    assert isplitT.dtype == isplitQ.dtype == jnp.int32
    assert lvl_leaf.dtype == ilist.dtype == ilist_splitsB.dtype == jnp.int32
    assert k in (4,8,12,16,32,64), "Only k=4,8,12,16,32,64 supported"

    x4a = jnp.concatenate((xT, jnp.zeros(xT.shape[:-1])[...,None]), axis=-1)
    x4b = jnp.concatenate((xQ, jnp.zeros(xQ.shape[:-1])[...,None]), axis=-1)

    x4leaf = jnp.concatenate((xleaf, lvl_leaf.view(jnp.float32)[...,None]), axis=-1)

    out_type = jax.ShapeDtypeStruct((xQ.shape[0], k, 2), jnp.int32)
    knn = jax.ffi.ffi_call("IlistKNNSearch", (out_type, ))(
        x4a, x4b, isplitT, isplitQ, x4leaf, ilist, ilist_splitsB,
        boxsize=np.float32(boxsize)
    )[0]
    rknn, iknn = knn[...,0].view(jnp.float32), knn[...,1].view(jnp.int32)
 
    return rknn, iknn
ilist_knn_search.jit = jax.jit(ilist_knn_search, static_argnames=("k", "boxsize"))


def build_ilist_knn(xleaf, lvl_leaf, npart_leaf, isplit, node_ilist, node_ilist_splits, k=32, boxsize=0.):
    x4leaf = jnp.concatenate((xleaf, lvl_leaf.view(jnp.float32)[...,None]), axis=-1)

    tmp_buf = jax.ShapeDtypeStruct((2, len(xleaf)), jnp.int32)
    leaf_ilist = jax.ShapeDtypeStruct((128 * len(xleaf),), jnp.int32)
    leaf_ilist_splits = jax.ShapeDtypeStruct((len(xleaf)+1,), jnp.int32)

    res = jax.ffi.ffi_call("ConstructIlist", (tmp_buf, leaf_ilist, leaf_ilist_splits))(
        x4leaf, npart_leaf, isplit, node_ilist, node_ilist_splits,
        k=np.int32(k), boxsize=np.float32(boxsize)
    )

    return res


def box_dist2(c1, c2, s1, s2, mode="shortest"):
    """Shortest or longest distance between any points in two nodes
    mode: "shortest" or "longest"
    """
    dx = jnp.abs(c1 - c2)
    if mode == "shortest":
        dx = jnp.maximum(dx - (s1 + s2) / 2., 0.)
    elif mode == "longest":
        dx = jnp.maximum(dx + (s1 + s2) / 2., 0.)
    else:
        raise ValueError("mode must be 'shortest' or 'longest'")
    return jnp.sum(dx**2, axis=-1)

def cumsum_starting_with_zero(x):
    return jnp.concatenate((jnp.zeros((1,) + x.shape[1:], dtype=x.dtype), jnp.cumsum(x, axis=0)))

def knn_interactions(xcent, dx, npart, bins=None, k=32, batch_size=128, alloc_fac=128):
    """Get a (weakly) sorted interaction list between leaves of an octree.

    xcent : centers of leaves
    npart : number of particles in each leaf
    dx : extend of the leaves
    bins : bins for discretizing the distances to determine the necessary interaciton radius
    k : number of neighbors to include in the interaction list"""
    
    if bins is None:
        bins = jnp.logspace(-0.5, 1., 32)
    nleaves = len(xcent)

    def handle_single_leaf(ileaf):
        # We want to find the smallest distance at which we are guaranteed to find >= k neighbors
        # For this we need to include leaves whenever they exceed the distance where they are fully
        # included by every particle in the source leaf

        rbase2 = jnp.sum(dx[ileaf]**2, axis=-1)

        dist2 = box_dist2(xcent[ileaf], xcent, dx, dx, mode="longest")

        dratio2 = dist2 * (1./ rbase2)

        # bins, nkincl tells us how many neighbours are at least included at which distance:
        nkincl = jnp.cumsum(jnp.histogram(dratio2, bins=bins, weights=npart)[0])
        r2min = jnp.min(jnp.where(nkincl >= k, bins[1:], jnp.inf), axis=-1) * rbase2

        # To count the leaves we need to check at a given radius, we need to compare the closest
        # distance
        dist2min = box_dist2(xcent[ileaf], xcent, dx[ileaf], dx, mode="shortest")
        ninteractions = jnp.sum(dist2min <= r2min, axis=-1)

        return jnp.sqrt(r2min), ninteractions
    
    rneed, ninteractions = jax.lax.map(handle_single_leaf, jnp.arange(nleaves), batch_size=batch_size)

    offsets = cumsum_starting_with_zero(ninteractions)

    interactions = jnp.zeros(alloc_fac * nleaves, dtype=jnp.int32)
    
    def insert_interactions(ileaf, interactions):
        dist2 = box_dist2(xcent[ileaf], xcent, dx[ileaf], dx, mode="shortest")
        dist2b = box_dist2(xcent[ileaf], xcent, dx[ileaf], dx, mode="longest")

        isort = jnp.lexsort([dist2b, dist2])

        iarange = jnp.arange(nleaves)
        ninleaf = offsets[ileaf + 1] - offsets[ileaf]
        ioff = jnp.where(iarange < ninleaf, offsets[ileaf] + iarange, offsets[-1])

        return interactions.at[ioff].set(isort)

    interactions = jax.lax.fori_loop(0, nleaves, insert_interactions, interactions)

    return rneed, interactions, offsets
knn_interactions.jit = jax.jit(knn_interactions, static_argnames=["bins", "k", "batch_size", "alloc_fac"])
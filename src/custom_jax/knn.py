import numpy as np
import jax
import jax.numpy as jnp
import custom_jax.nb_knn as nb_knn
from .tree import summarize_leaves, lvl_to_ext, get_node_box
from .common import conditional_callback

jax.ffi.register_ffi_target("IlistKNNSearch", nb_knn.IlistKNNSearch(), platform="CUDA")
jax.ffi.register_ffi_target("ConstructIlist", nb_knn.ConstructIlist(), platform="CUDA")
jax.ffi.register_ffi_target("SegmentSort", nb_knn.SegmentSort(), platform="CUDA")

def ilist_knn_search(xT, isplitT, xleaf, lvl_leaf, ilist, ir2list, ilist_splitsB, xQ=None,  isplitQ=None, k=32, boxsize=0.):
    """Finds the k nearest neighbors of xfind in the z-sorted positions xzsort
    """
    if xQ is None: xQ = xT
    if isplitQ is None: isplitQ = isplitT

    assert ir2list.shape == ilist.shape, "rilist must have the same shape as ilist"

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
        x4a, x4b, isplitT, isplitQ, x4leaf, ilist, ir2list, ilist_splitsB,
        boxsize=np.float32(boxsize)
    )[0]
    rknn, iknn = knn[...,0].view(jnp.float32), knn[...,1].view(jnp.int32)
 
    return rknn, iknn
ilist_knn_search.jit = jax.jit(ilist_knn_search, static_argnames=("k", "boxsize"))


def build_ilist_knn(xleaf, lvl_leaf, npart_leaf, isplit, node_ilist, node_ir2list, node_ilist_splits, k=32, boxsize=0., 
                    alloc_fac=128):
    assert node_ilist_splits.shape[0] == isplit.shape[0], "Should both correspond to no. of nodes+1"

    assert node_ilist.shape == node_ir2list.shape, "node_ilist and node_ir2list must have the same shape"

    x4leaf = jnp.concatenate((xleaf, lvl_leaf.view(jnp.float32)[...,None]), axis=-1)

    rbuf = jax.ShapeDtypeStruct((len(xleaf),), jnp.float32)
    leaf_ilist = jax.ShapeDtypeStruct((int(alloc_fac * len(xleaf)),), jnp.int32)
    leaf_ilist_splits = jax.ShapeDtypeStruct((len(xleaf)+1,), jnp.int32)
    leaf_ilist_rad = jax.ShapeDtypeStruct(leaf_ilist.shape, jnp.float32)

    radii, il, ir2l, ispl = jax.ffi.ffi_call("ConstructIlist", (rbuf, leaf_ilist, leaf_ilist_rad, leaf_ilist_splits))(
        x4leaf, npart_leaf, isplit, node_ilist, node_ir2list, node_ilist_splits,
        k=np.int32(k), boxsize=np.float32(boxsize)
    )

    def myerror(n1, n2):
        raise MemoryError(f"The interaction list allocation is too small. (need: {n1} have: {n2})" +
                          f"increase alloc_fac at least by a factor of {n1/n2:.1f}")
    ispl = ispl + conditional_callback(ispl[-1] > il.size, myerror, ispl[-1], il.size)

    return il, ir2l, ispl
build_ilist_knn.jit = jax.jit(build_ilist_knn, static_argnames=["k", "boxsize", "alloc_fac"])


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

        dratio2 = dist2 / rbase2

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
knn_interactions.jit = jax.jit(knn_interactions, static_argnames=["k", "batch_size", "alloc_fac"])

def dense_ilist(num):
    ilist = jnp.array((jnp.arange(num, dtype=jnp.int32),)*num)
    isplits = jnp.arange(num+1, dtype=jnp.int32)*num
    return ilist.flatten(), isplits

def brute_force_node_ilist_prep(octree, k=16):
    """Helps preparing the inputs for build_ilist_knn"""
    npart_leaf = octree.leaf_particle_bounds[1:] - octree.leaf_particle_bounds[:-1]
    level_leaf = octree.level_binary[octree.node_of_leaf] - 1
    leaf_cent, leaf_ext = get_node_box(octree.xleaf, level_leaf)

    nleaves = int(octree.nnodes - 1 )
    nnodes = nleaves // 32 + 1
    isplit = jnp.clip(jnp.arange(0, nnodes+1, dtype=jnp.int32)*32, 0, nleaves)
    # node_ilist = jnp.array((jnp.arange(nnodes, dtype=jnp.int32),)*nnodes)
    # node_ilist_splits = jnp.arange(nnodes+1, dtype=jnp.int32)*nnodes
    node_ilist, node_ilist_splits = dense_ilist(nnodes)

    return leaf_cent, level_leaf, npart_leaf, isplit, node_ilist, node_ilist_splits

def build_ilist_recursive(xleaf, lvleaf, nleaf, max_size=64, num_part=None, 
        refine_fac=8, k=16, stop_coarsen=128, boxsize=0., alloc_fac=128.):
    """Recursively builds an interaction list for kNN search. This is done by recursively:
    (1) Coarsen the leaves
    (2) Get the interaction list for the coarsened leaves (recursively)
    (3) Use the interaction list of the coarsened leaves to build the finer interaction list
    """
    assert refine_fac < 16, "refine_fac should be < 16 to respect some blocksize assumptions in CUDA"

    if len(xleaf) <= stop_coarsen:
        il, ispl = dense_ilist(len(xleaf))
        ir2l = jnp.zeros(il.shape, dtype=jnp.float32)
        return il, ir2l, ispl
    
    spl2, nleaf2, lvleaf2, xleaf2, numleaves2 = summarize_leaves(
        xleaf, max_size=max_size, nleaf=nleaf, num_part=num_part)
    # Now build the list on the coarser levels
    # We increase the allocation factor a bit, because the total allocation will anyways be much
    # smaller on the coarser levels and we don't want it to fail on coarser levels
    il2, ir2l, ispl2 = build_ilist_recursive(
        xleaf2, lvleaf2, nleaf2, max_size=max_size*refine_fac, num_part=num_part,
        alloc_fac=alloc_fac*np.sqrt(refine_fac))
    il, ir2l, ispl = build_ilist_knn(
        xleaf, lvleaf, nleaf, spl2, il2, ir2l, ispl2, alloc_fac=alloc_fac, 
        k=k, boxsize=boxsize)
    
    return il, ir2l, ispl
build_ilist_recursive.jit = jax.jit(build_ilist_recursive, static_argnames=[
    'max_size', 'num_part', 'refine_fac', 'k', 'stop_coarsen', 'boxsize'])

def knn(posz, k=16, boxsize=0., alloc_fac=256.):
    spl, nleaf, llvl, xleaf, numleaves = summarize_leaves(posz, max_size=32)

    il, ir2l, ispl = build_ilist_recursive(xleaf, llvl, nleaf, max_size=32*15, refine_fac=15,
                                          num_part=len(posz), k=k, boxsize=boxsize, alloc_fac=alloc_fac)

    rknn, iknn = ilist_knn_search(posz, spl, xleaf, llvl, il, ir2l, ispl, k=k, boxsize=boxsize)

    return rknn, iknn
knn.jit = jax.jit(knn, static_argnames=["k", "boxsize", "alloc_fac"])

def segment_sort(key, val, isplit, smem_size=512):
    """Sorts key/val pairs within segments defined by isplit"""
    assert key.dtype == jnp.float32
    assert val.dtype == jnp.int32
    assert isplit.dtype == jnp.int32
    assert key.shape == val.shape
    assert isplit.ndim == 1
    assert smem_size >= 64

    out_type = (jax.ShapeDtypeStruct(key.shape, key.dtype), jax.ShapeDtypeStruct(val.shape, val.dtype))
    key_sorted, val_sorted = jax.ffi.ffi_call("SegmentSort", out_type)(
        key, val, isplit, smem_size=np.int32(smem_size)
    )
    return key_sorted, val_sorted
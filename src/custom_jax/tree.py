from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

import custom_jax.nb_tree as nb_tree

jax.ffi.register_ffi_target("PosZorderSort", nb_tree.PosZorderSort(), platform="CUDA")
jax.ffi.register_ffi_target("BuildZTree", nb_tree.BuildZTree(), platform="CUDA")
jax.ffi.register_ffi_target("KNNSearch", nb_tree.KNNSearch(), platform="CUDA")
jax.ffi.register_ffi_target("IlistKNNSearch", nb_tree.IlistKNNSearch(), platform="CUDA")

def pos_zorder_sort(x, block_size=64):
    assert x.dtype == jnp.float32
    assert x.shape[-1] == 3

    # To optimize memory layout, we bundle position and id together into a single array
    # We later need to reinterprete the output to extract positions and ids
    out_type = jax.ShapeDtypeStruct((x.shape[0],4), jnp.int32)
    # This is a guess, how much a temporary storage in cub::DeviceMergeSort::SortKeys requires
    # If we estimate too little, an error will be thrown from the C++ code:
    tmp_buff_type = jax.ShapeDtypeStruct((x.shape[0] + np.maximum(1024, x.shape[0]//16), 4), jnp.int32)
    isort = jax.ffi.ffi_call("PosZorderSort", (out_type, tmp_buff_type))(x, block_size=np.uint64(block_size))[0]

    pos = isort[:, :3].view(jnp.float32)
    ids = isort[:, 3].view(jnp.int32)

    return pos, ids
pos_zorder_sort.jit = jax.jit(pos_zorder_sort, static_argnames=("block_size",))

def build_ztree(x, block_size=64):
    """Builds a z-tree assume z-sorted positions
    
    returns (level, lbound, rbound, lchild, rchild)
    """
    assert x.dtype == jnp.float32
    assert x.shape[-1] == 3

    out_type = jax.ShapeDtypeStruct((5, x.shape[0]+1), jnp.int32)
    ztree = jax.ffi.ffi_call("BuildZTree", (out_type,))(x, block_size=np.uint64(block_size))[0]
 
    root_node = jnp.argmax(ztree[0][1:-1]).astype(jnp.int32) + 1 # Root node is the one with highest level
    # Marke it as a child of the first and last nodes
    ztree = ztree.at[4,0].set(root_node) # first right child
    ztree = ztree.at[3,-1].set(root_node) # last left child

    return ztree
build_ztree.jit = jax.jit(build_ztree, static_argnames=("block_size",))

def knn_search(xzsort, xfind, k=4, block_size=64, init_factor=1.0):
    """Finds the k nearest neighbors of xfind in the z-sorted positions xzsort
    """
    assert xzsort.dtype == xfind.dtype == jnp.float32
    assert xzsort.shape[-1] == xfind.shape[-1] == 3
    assert init_factor >= 0.5

    assert k in (4,8,16,32), "so far only k=4,8,16,32 are supported"

    nsearch_init = int(np.ceil(init_factor * k) + 1)

    print(f"knn_search: k={k}, block_size={block_size}, nsearch_init={nsearch_init}")

    out_type = jax.ShapeDtypeStruct((xzsort.shape[0], k), jnp.int32)
    i1,i2 = jax.ffi.ffi_call("KNNSearch", (out_type, out_type))(
        xzsort, xfind, block_size=np.uint64(block_size), nsearch_init=np.uint64(nsearch_init))
 
    return i1, i2
knn_search.jit = jax.jit(knn_search, static_argnames=("k", "block_size", "init_factor"))

def ilist_knn_search(xA, xB, isplitA, isplitB, lvlA, ilist, ilist_splitsB, k=32, interactions_per_block=1, boxsize=0.):
    """Finds the k nearest neighbors of xfind in the z-sorted positions xzsort
    """
    assert xA.dtype == xB.dtype == jnp.float32
    assert xA.shape[-1] == xB.shape[-1] == 3
    assert isplitA.dtype == isplitB.dtype == jnp.int32
    assert lvlA.dtype == ilist.dtype == ilist_splitsB.dtype == jnp.int32
    assert k in (4,8,16,32), "Only k=32 is suppported so far"
    assert interactions_per_block == 1

    x4a = jnp.concatenate((xA, jnp.zeros(xA.shape[:-1])[...,None]), axis=-1)
    x4b = jnp.concatenate((xB, jnp.zeros(xB.shape[:-1])[...,None]), axis=-1)

    out_type = jax.ShapeDtypeStruct((xB.shape[0], k, 2), jnp.int32)
    knn = jax.ffi.ffi_call("IlistKNNSearch", (out_type, ))(
        x4a, x4b, isplitA, isplitB, lvlA, ilist, ilist_splitsB,
        interactions_per_block=np.uint64(interactions_per_block), boxsize=np.float32(boxsize)
    )[0]
    rknn, iknn = knn[...,0].view(jnp.float32), knn[...,1].view(jnp.int32)
 
    return rknn, iknn
ilist_knn_search.jit = jax.jit(ilist_knn_search, static_argnames=("k", "interactions_per_block", "boxsize"))


# ================================= Deprecated functions   ======================================= #
# They will be deleted later, for now we keep them for comparison purposes

if hasattr(nb_tree, "OldArgsort"):
    jax.ffi.register_ffi_target("OldArgsort", nb_tree.OldArgsort(), platform="CUDA")
    jax.ffi.register_ffi_target("OldI3zsort", nb_tree.OldI3zsort(), platform="CUDA")
    jax.ffi.register_ffi_target("OldF3zsort", nb_tree.OldF3zsort(), platform="CUDA")
    jax.ffi.register_ffi_target("OldI3Argsort", nb_tree.OldI3Argsort(), platform="CUDA")
    jax.ffi.register_ffi_target("OldI3zMergesort", nb_tree.OldI3zMergesort(), platform="CUDA")

    def old_argsort_cubradix(key, block_size=64):
        assert key.dtype == jnp.int32

        out_type = jax.ShapeDtypeStruct(key.shape[0:1], jnp.int32)
        isort = jax.ffi.ffi_call("OldArgsort", (out_type,))(key, block_size=np.uint64(block_size))
        return isort[0]
    old_argsort_cubradix.jit = jax.jit(old_argsort_cubradix, static_argnames=("block_size",))

    def old_i3zsort_thrust(ids, block_size=64):
        assert ids.dtype == jnp.int32
        assert ids.shape[-1] == 3

        out_type = jax.ShapeDtypeStruct(ids.shape[0:1], jnp.int32)
        isort = jax.ffi.ffi_call("OldI3zsort", (out_type,))(ids, block_size=np.uint64(block_size))
        return isort[0]
    old_i3zsort_thrust.jit = jax.jit(old_i3zsort_thrust, static_argnames=("block_size",))

    def old_f3zsort_thrust(x, block_size=64):
        assert x.dtype == jnp.float32
        assert x.shape[-1] == 3

        out_type = jax.ShapeDtypeStruct(x.shape[0:1], jnp.int32)

        isort = jax.ffi.ffi_call("OldF3zsort", (out_type,))(x, block_size=np.uint64(block_size))
        return isort[0]
    old_f3zsort_thrust.jit = jax.jit(old_f3zsort_thrust, static_argnames=("block_size",))

    def old_i3argsort_cubradix(ids, block_size=64):
        assert ids.dtype == jnp.int32
        assert ids.shape[-1] == 3

        out_type = jax.ShapeDtypeStruct(ids.shape[0:1], jnp.int32)
        isort = jax.ffi.ffi_call("OldI3Argsort", (out_type,))(ids, block_size=np.uint64(block_size))
        return isort[0]
    old_i3argsort_cubradix.jit = jax.jit(old_i3argsort_cubradix, static_argnames=("block_size",))

    def old_i3zsort_cubmerge(ids, block_size=64):
        assert ids.dtype == jnp.int32
        assert ids.shape[-1] == 3

        out_type = jax.ShapeDtypeStruct((ids.shape[0],4), jnp.int32)
        isort = jax.ffi.ffi_call("OldI3zMergesort", (out_type,))(ids, block_size=np.uint64(block_size))
        return isort[0][:,3]
    old_i3zsort_cubmerge.jit = jax.jit(old_i3zsort_cubmerge, static_argnames=("block_size",))
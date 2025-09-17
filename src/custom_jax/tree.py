import numpy as np
import jax
import jax.numpy as jnp

import custom_jax.nb_tree as nb_tree

jax.ffi.register_ffi_target("PosZorderSort", nb_tree.PosZorderSort(), platform="CUDA")
jax.ffi.register_ffi_target("BuildZTree", nb_tree.BuildZTree(), platform="CUDA")
jax.ffi.register_ffi_target("SummarizeLeaves", nb_tree.SummarizeLeaves(), platform="CUDA")

def lvl_to_ext(level_binary):
    olvl, omod = level_binary//3, level_binary % 3
    levels_3d = jnp.stack((olvl, olvl + (omod >= 2).astype(jnp.int32), olvl + (omod >= 1).astype(jnp.int32)),axis=-1)
    return 2.**levels_3d

def get_node_box(x, level_binary):
    node_size = lvl_to_ext(level_binary)
    node_cent = (jnp.floor(x / node_size) + 0.5) * node_size   # might have some round-off issues
    # node_cent = x + (0.5 * node_size - jnp.mod(x, node_size))
    return node_cent, node_size

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

def div_ceil(a, b):
    return (a + b - 1) // b

def prepend_num(arr, val=0):
    return jnp.concatenate([jnp.asarray([val], dtype=arr.dtype), arr], axis=0)

def summarize_leaves(xleaf, nleaf=None, max_size=64, num_part=None, ref_fac=None):
    """Summarizes leaf nodes into parent nodes
    """
    if nleaf is None:
        nleaf = jnp.ones((xleaf.shape[0],), dtype=jnp.int32)
        num = jnp.arange(0, len(xleaf)+1, dtype=jnp.int32)
    else:
        num = prepend_num(jnp.cumsum(nleaf), 0)
    if num_part is None:
        num_part = len(xleaf)
    if ref_fac is None:
        scan_size = max_size + 1
    else:
        # Haven't perfectly understood yet, what scan_size is needed in the worst case...
        # In principle ref_fac + 2 should be enough, but I guess some rounding errors in the
        # ref_fac corrupt this sometimes (?) for now let's leave a bit slack
        scan_size = int(2.*ref_fac + 3) 
    assert scan_size <= 1024, "This probably should not happen..."

    block_size = np.clip((scan_size//64) * 64, 64, 512)

    # We may have some invalid leaves at the end
    # Let's keep track until where our leaves are valid
    nleaves_filled = jnp.count_nonzero(nleaf)
    
    max_new_leaves = div_ceil(num_part, np.maximum(max_size//2, 1))
    # print("scan size:", scan_size, max_size)

    assert xleaf.dtype == jnp.float32
    assert xleaf.shape[-1] == 3

    xnleaf = jnp.concatenate((xleaf, nleaf[:,None].view(jnp.float32)), axis=-1)

    out_splits_type = jax.ShapeDtypeStruct((xnleaf.shape[0]+1,), jnp.int32)

    flag_split = jax.ffi.ffi_call("SummarizeLeaves", (out_splits_type,))(
        xnleaf, nleaves_filled, max_size=np.uint64(max_size),
        block_size=np.uint64(block_size), scan_size=np.uint64(scan_size))[0]

    # Get the splitting points of leaves
    splits = jnp.where(flag_split > -1000, size=max_new_leaves+1, fill_value=nleaves_filled)[0]
    new_nleaf = num[splits[1:]] - num[splits[:-1]]

    new_leaf_lvl = jnp.minimum(flag_split[splits[:-1]], flag_split[splits[1:]]) - 1

    numleaves = jnp.count_nonzero(new_nleaf)

    new_leaf_cent = get_node_box(xleaf[splits[:-1]], jnp.full_like(new_leaf_lvl, new_leaf_lvl))[0]
    # new_leaf_cent = jnp.where(jnp.arange(len(new_leaf_cent))[:,None] < numleaves, new_leaf_cent, 0.)

    # assert numleaves <= max_new_leaves, "Please provide nptot if leaves are not particles"

    return splits, new_nleaf, new_leaf_lvl, new_leaf_cent, numleaves

summarize_leaves.jit = jax.jit(summarize_leaves, static_argnames=("max_size", "num_part", "ref_fac"))

# Matches CUDA's float32 behavior
def float_xor_msb(a, b):
    """
    Finds the most significant differing bit "level" between two float32s.
    Returns 128 if sign bits differ, otherwise follows the exponent/mantissa logic.
    Works with broadcasting over arrays.
    """
    a = jnp.asarray(a, jnp.float32)
    b = jnp.asarray(b, jnp.float32)

    # If sign bits differ, return 128
    sign_diff = jnp.not_equal(jnp.signbit(a), jnp.signbit(b))

    # Bitcast |a| and |b| to uint32
    a_bits = jax.lax.bitcast_convert_type(jnp.abs(a), jnp.uint32)
    b_bits = jax.lax.bitcast_convert_type(jnp.abs(b), jnp.uint32)

    # Extract unbiased exponents: (bits >> 23) - 127
    a_exp = (a_bits >> jnp.uint32(23)).astype(jnp.int32) - jnp.int32(127)
    b_exp = (b_bits >> jnp.uint32(23)).astype(jnp.int32) - jnp.int32(127)

    same_exp = a_exp == b_exp

    # If exponents equal, compare mantissas via XOR, then use leading zeros
    xor_bits = jnp.bitwise_xor(a_bits, b_bits)
    # lax.clz counts leading zeros on unsigned integers
    clz = jax.lax.clz(xor_bits).astype(jnp.int32)

    # CUDA comment: "There will always be 8 leading zeros due to the exponent"
    # (sign bit is removed by fabsf, so sign is zero as well)
    mantissa_term = a_exp + (jnp.int32(8) - clz)

    # If exponents differ, choose the larger exponent
    larger_exp = jnp.maximum(a_exp, b_exp)

    result = jnp.where(same_exp, mantissa_term, larger_exp)
    result = jnp.where(sign_diff, jnp.int32(128), result)
    return result

def ztree_diff_level(p1, p2):
    """
    p1, p2: (..., 3) float32 arrays (or anything broadcastable to that)
    Returns the level: max(3*msb_x+3, 3*msb_y+2, 3*msb_z+1)
    """
    p1 = jnp.asarray(p1, jnp.float32)
    p2 = jnp.asarray(p2, jnp.float32)

    msb_x = float_xor_msb(p1[..., 0], p2[..., 0])
    msb_y = float_xor_msb(p1[..., 1], p2[..., 1])
    msb_z = float_xor_msb(p1[..., 2], p2[..., 2])

    level = jnp.maximum(
        3 * msb_x + 3,
        jnp.maximum(3 * msb_y + 2, 3 * msb_z + 1),
    )
    return level
ztree_diff_level.jit = jax.jit(ztree_diff_level)


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
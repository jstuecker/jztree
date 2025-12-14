import numpy as np
import jax
import jax.numpy as jnp
from jztree_cuda import ffi_fof
from .tree import pos_zorder_sort, search_sorted_z
from .common import conditional_callback, masked_prefix_sum, cumsum_starting_with_zero, inverse_indices
from .data import KNNData, KNNConfig

import fmdj

jax.ffi.register_ffi_target("NodeFofAndIlist", ffi_fof.NodeFofAndIlist(), platform="CUDA")

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

def node_fof_and_ilist(
        node_igroup, node_ilist_splits, node_ilist, isplit, xleaf, lvl_leaf,
        # npart_leaf, isplit, node_ilist, node_ir2list, node_ilist_splits, 
        rlink, boxsize=0., alloc_fac=128, block_size=32
    ):
    assert node_ilist_splits.shape[0] == isplit.shape[0], "Should both correspond to no. of nodes+1"
    
    x4leaf = jnp.concatenate((xleaf, lvl_leaf.view(jnp.float32)[...,None]), axis=-1)

    leaf_igroup = jax.ShapeDtypeStruct((len(xleaf),), jnp.int32)
    leaf_icount = jax.ShapeDtypeStruct((len(xleaf),), jnp.int32)
    leaf_ilist_ispl = jax.ShapeDtypeStruct((len(xleaf)+1,), jnp.int32)
    leaf_ilist = jax.ShapeDtypeStruct((int(alloc_fac * len(xleaf)),), jnp.int32)
    
    outputs = (leaf_igroup, leaf_icount, leaf_ilist_ispl, leaf_ilist)

    res = jax.ffi.ffi_call("NodeFofAndIlist", outputs)(
        node_igroup, node_ilist_splits, node_ilist, isplit, x4leaf,
        r2link=np.float32(rlink*rlink), boxsize=np.float32(boxsize), block_size=np.int32(block_size)
    )

    leaf_igroup, leaf_icount, leaf_ilist_ispl, leaf_ilist = res

    def err(n1, n2):
        raise MemoryError(f"The interaction list allocation is too small. (need: {n1} have: {n2})" +
                          f"increase alloc_fac at least by a factor of {n1/n2:.1f}")
    n1, n2 = leaf_ilist_ispl[-1], leaf_ilist.shape[0]
    leaf_igroup = leaf_igroup + conditional_callback(n1 > n2, err, n1, n2)

    return leaf_igroup, leaf_ilist_ispl, leaf_ilist
node_fof_and_ilist.jit = jax.jit(
    node_fof_and_ilist, static_argnames=["boxsize", "rlink", "alloc_fac", "block_size"]
)
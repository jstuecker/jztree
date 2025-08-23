import numpy as np
import jax
import jax.numpy as jnp
import custom_jax.nb_knn as nb_knn

jax.ffi.register_ffi_target("IlistKNNSearch", nb_knn.IlistKNNSearch(), platform="CUDA")

def ilist_knn_search(xT, isplitT, lvlT, ilist, ilist_splitsB, xQ=None,  isplitQ=None, k=32, interactions_per_block=1, boxsize=0.):
    """Finds the k nearest neighbors of xfind in the z-sorted positions xzsort
    """
    if xQ is None: xQ = xT
    if isplitQ is None: isplitQ = isplitT

    assert xT.dtype == xQ.dtype == jnp.float32
    assert xT.shape[-1] == xQ.shape[-1] == 3
    assert isplitT.dtype == isplitQ.dtype == jnp.int32
    assert lvlT.dtype == ilist.dtype == ilist_splitsB.dtype == jnp.int32
    assert k in (4,8,16,32), "Only k=32 is suppported so far"
    assert interactions_per_block == 1

    x4a = jnp.concatenate((xT, jnp.zeros(xT.shape[:-1])[...,None]), axis=-1)
    x4b = jnp.concatenate((xQ, jnp.zeros(xQ.shape[:-1])[...,None]), axis=-1)

    out_type = jax.ShapeDtypeStruct((xQ.shape[0], k, 2), jnp.int32)
    knn = jax.ffi.ffi_call("IlistKNNSearch", (out_type, ))(
        x4a, x4b, isplitT, isplitQ, lvlT, ilist, ilist_splitsB,
        interactions_per_block=np.uint64(interactions_per_block), boxsize=np.float32(boxsize)
    )[0]
    rknn, iknn = knn[...,0].view(jnp.float32), knn[...,1].view(jnp.int32)
 
    return rknn, iknn
ilist_knn_search.jit = jax.jit(ilist_knn_search, static_argnames=("k", "interactions_per_block", "boxsize"))


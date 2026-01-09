import numpy as np
import jax
import jax.numpy as jnp
from jztree_cuda import ffi_fof
from .tree import pos_zorder_sort, grouped_dense_interaction_list
from .common import conditional_callback
from .data import  FofConfig, FofData, PosLvl, Label
from fmdj.data import InteractionList
from typing import Tuple
from fmdj.comm import get_rank_info, pytree_len, all_to_all_with_irank, all_to_all_request

import fmdj

jax.ffi.register_ffi_target("NodeFofAndIlist", ffi_fof.NodeFofAndIlist(), platform="CUDA")
jax.ffi.register_ffi_target("ParticleFof", ffi_fof.ParticleFof(), platform="CUDA")
jax.ffi.register_ffi_target("InsertLinks", ffi_fof.InsertLinks(), platform="CUDA")
jax.ffi.register_ffi_target("NodeToChildLabel", ffi_fof.NodeToChildLabel(), platform="CUDA")

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

def node_fof_and_ilist(
        node_ilist: InteractionList, isplit: jax.Array, 
        child_data: PosLvl, child_igroup: jax.Array, 
        rlink: float, boxsize: float = 0., alloc_fac:float = 128., block_size:int = 32
    ) -> Tuple[jax.Array, InteractionList]:
    assert node_ilist.ispl.shape[0] == isplit.shape[0], "Should both correspond to no. of nodes+1"
    assert len(child_data.lvl) == len(child_igroup), "Should both correspond to no. of childrne"
    assert node_ilist.iother.size < 2**31, "So far only int32 supported {ilist_alloc_size/2**31}"
    
    nchild = len(child_data.pos)

    child_igroup_out = jax.ShapeDtypeStruct((nchild,), jnp.int32)
    child_ilist_ispl = jax.ShapeDtypeStruct((nchild+1,), jnp.int32)
    child_ilist = jax.ShapeDtypeStruct((int(alloc_fac * nchild),), jnp.int32)
    
    outputs = (child_igroup_out, child_ilist_ispl, child_ilist)

    res = jax.ffi.ffi_call("NodeFofAndIlist", outputs)(
        node_ilist.ispl, node_ilist.iother, isplit, child_data.pos_lvl(), child_igroup,
        r2link=np.float32(rlink*rlink), boxsize=np.float32(boxsize), block_size=np.int32(block_size)
    )

    child_igroup_out, child_ilist_ispl, child_ilist = res
    child_ilist = InteractionList(child_ilist_ispl, child_ilist, nfilled=child_ilist_ispl[-1])

    def err(n1, n2):
        raise MemoryError(f"The interaction list allocation is too small. (need: {n1} have: {n2})" +
                          f"increase alloc_fac at least by a factor of {n1/n2:.1f}")
    n1, n2 = child_ilist.nfilled, child_ilist.iother.shape[0]
    child_igroup_out = child_igroup_out + conditional_callback(n1 > n2, err, n1, n2)

    return child_igroup_out, child_ilist
node_fof_and_ilist.jit = jax.jit(
    node_fof_and_ilist, static_argnames=["boxsize", "rlink", "alloc_fac", "block_size"]
)

def insert_links(igroup, iA, iB, num_links: jax.Array | None = None, block_size=64):
    ngroups = len(igroup)
    if num_links is None:
        num_links = jnp.array(len(iA), dtype=jnp.int32)

    igr_out = jax.ffi.ffi_call("InsertLinks", (jax.ShapeDtypeStruct((ngroups,), jnp.int32),))(
        igroup, iA, iB, num_links, block_size=np.int32(block_size)
    )[0]
    
    return igr_out

def node_to_child_label(igroup: jax.Array, spl: jax.Array, size_child: int, 
                        block_size: int = 64) -> jax.Array:
    outputs = (jax.ShapeDtypeStruct((size_child,), jnp.int32),)
    igroup_child = jax.ffi.ffi_call("NodeToChildLabel",  outputs)(
        igroup, spl, block_size=np.uint64(block_size)
    )[0]

    return igroup_child

def node_node_fof(th: fmdj.data.TreeHierarchy, rlink: float, boxsize: float=0., alloc_fac_ilist: int = 128):
    nplanes = th.num_planes()

    # initialize top-level interaction list
    spl, ilist, nsup = grouped_dense_interaction_list(
        th.lvl.num(nplanes-1), size_ilist=int(th.plane_sizes[nplanes-1]*alloc_fac_ilist), ngroup=32
    )

    # Get coarsest plane data
    igroup = jnp.arange(len(spl)-1, dtype=jnp.int32)

    for level in reversed(range(nplanes)):
        size = th.plane_sizes[level]
        child_data = PosLvl(th.geom_cent.get(level, size), th.lvl.get(level, size))
        child_igroup = node_to_child_label(igroup, spl, size)

        igroup, ilist = node_fof_and_ilist(
            ilist, spl, child_data, child_igroup,
            rlink=rlink, boxsize=boxsize, alloc_fac=alloc_fac_ilist
        )

        spl = th.ispl_n2n.get(level, size+1)
        
    return igroup, ilist, spl
node_node_fof.jit = jax.jit(node_node_fof, static_argnames=["rlink", "boxsize", "alloc_fac_ilist"])

def particle_particle_fof(node_igroup, ispl, il, spl, posz, rlink: float, boxsize: float = 0., block_size=32):
    igroup = jax.ffi.ffi_call("ParticleFof", (jax.ShapeDtypeStruct((len(posz),), node_igroup.dtype),))(
        node_igroup, ispl, il, spl, posz,
        r2link=np.float32(rlink*rlink), boxsize=np.float32(boxsize), block_size=np.int32(block_size)
    )[0]

    return igroup
particle_particle_fof.jit = jax.jit(particle_particle_fof, static_argnames=["rlink", "boxsize", "block_size"])

# ------------------------------------------------------------------------------------------------ #
#                              Functions specific to parallel version                              #
# ------------------------------------------------------------------------------------------------ #

def global_to_local_label(labels: Label) -> jax.Array:
    """Map each global to the index of the first occurence of it"""
    # Note, currently the Fof self-linking detection can be wrong for index 0!
    pairs = labels.stacked(posify = True)

    lab, indices, inv = jnp.unique(
        pairs, axis=0, size=len(pairs), return_index=True, return_inverse=True
    )
    
    return jnp.where(labels.igroup >= 0, indices[inv], labels.igroup)

def unique_labels(labels: Label, mask: jax.Array) -> Tuple[Label, jax.Array, jax.Array]:
    """Get unique labels among the masked locations

    returns labels_out, indices, num

    indices are so that labels == labels_out[indices] at locations where mask is True
    """
    pairs = labels.stacked(posify = True)

    masked_pairs, num, inv_mask = masked_to_dense(
        jnp.abs(pairs), mask, get_inverse=True, fill_value=-1
    )
    # To avoid counting invalid pairs as an extra label, we set them to the first valid label
    masked_pairs = jnp.where(masked_pairs == -1, masked_pairs[0], masked_pairs)

    lab, inv = jnp.unique(
        masked_pairs, axis=0, size=len(masked_pairs), return_inverse=True, fill_value=-1
    )

    return Label(lab[...,0], lab[...,1]), inv[inv_mask], jnp.sum(lab[...,0] != -1)

from dataclasses import dataclass
from fmdj.tools import offset_sum

@jax.tree_util.register_dataclass
@dataclass
class Link:
    a: Label
    b: Label

def masked_scatter(mask, arr, indices, values):
    indices = jnp.where(mask, indices, len(arr))
    return arr.at[indices].set(values)

def masked_min_scatter(mask, arr, indices, values):
    indices = jnp.where(mask, indices, len(arr))

    # first we erase the initial value of arr at the update locations,
    # so we will only get the min of the updates
    arr = arr.at[indices].set(jnp.iinfo(arr.dtype).max)
    
    return arr.at[indices].min(values)

def masked_to_dense(arr: jax.Array, mask, get_inverse=False, fill_value=0):
    pref, num = offset_sum(mask)
    indices = jnp.where(mask, pref, len(arr))
    new_arr = jnp.full(arr.shape, fill_value, arr.dtype).at[indices].set(arr)
    if get_inverse:
        return new_arr, num, indices
    else:
        return new_arr, num

def tree_where(condition: jax.Array, l1: Label, l2: Label) -> Label:
    return jax.tree.map(lambda x, y: jnp.where(condition, x, y), l1, l2)

def label_min_max(l1: Label, l2: Label):
    lmax = tree_where(l1 >= l2, l1, l2)
    lmin = tree_where(~(l1 >= l2), l1, l2)
    return lmin, lmax

def link_distributed_step(igroup: jax.Array, labels: Label, links: Link, nlinks: jax.Array
                          ) -> Tuple[jax.Array, jax.Array, Link, jax.Array]:
    rank, ndev, axis_name = get_rank_info()
    dtype = igroup.dtype

    def contract(lA: Label, lB: Label):
        # Contracts labels as far as possible, given the locally available information
        # We do this step every time the local information may have changed

        lA = tree_where(lA.irank == rank, labels[lA.igroup], lA)
        lB = tree_where(lB.irank == rank, labels[lB.igroup], lB)
        are_local = (lA.irank == rank).astype(dtype) + (lB.irank == rank).astype(dtype)
        lmin, lmax = label_min_max(lA, lB)
        return lmin, lmax, are_local

    # Communicate the links to the larger rank in the link
    send_rank = jnp.maximum(links.a.irank, links.b.irank)
    links, dev_spl = all_to_all_with_irank(
        send_rank, links, num=nlinks, axis_name=axis_name, copy_self=False
    )
    
    valid = jnp.arange(pytree_len(links), dtype=jnp.int32) < dev_spl[-1]

    lA, lB, are_local = contract(links.a, links.b)

    # Handle fully local links. Note that insert_links also contracts the local igroup graph
    igrA, num = masked_to_dense(lA.igroup, valid & (are_local==2))
    igrB, num = masked_to_dense(lB.igroup, valid & (are_local==2))
    igroup = insert_links(igroup, igrA, igrB, num)
    labels = labels[igroup]

    is_root = (labels.irank == rank) & (jnp.abs(labels.igroup) == jnp.arange(len(labels.igroup)))

    # Dereference again, since labels may have changed
    lmin, lmax, are_local = contract(lA, lB)
    was_resolved = lmin == lmax

    # Handle partially local links. For these we need to update the higher root node to any
    # parent of the lower node
    # Multiple links may try to update a single root node at the same time. To deal with this
    # race condition, we take the minimum suggested update and then need to retry the unresolved
    # in the next iteration where we will continue try them at the larger node

    update = (are_local == 1) & (lmax.irank == rank)  & (~was_resolved)
    update = update & (labels[lmax.igroup] >= lmin) & is_root[lmax.igroup]
    labels.irank = masked_min_scatter(update, labels.irank, lmax.igroup, lmin.irank)
    # only update labels.igroup with links that point towards the same rank
    update = update & (labels[lmax.igroup].irank == lmin.irank)
    labels.igroup = masked_min_scatter(update, labels.igroup, lmax.igroup, lmin.igroup)

    # Dereference again, since labels may have changed
    labels = labels[igroup]
    lmin, lmax, are_local = contract(lmin, lmax)

    was_resolved = lmin == lmax
    
    remaining_links = jax.tree.map(
        lambda l: masked_to_dense(l, valid & ~was_resolved)[0], Link(lmax, lmin)
    )

    return igroup, labels, remaining_links, jnp.sum(valid & ~was_resolved)
link_distributed_step.jit = jax.jit(link_distributed_step)

def contract_distributed(labels: Label, num: int | jnp.ndarray):
    rank, ndev, axis_name = get_rank_info()

    valid = jnp.arange(pytree_len(labels)) < num
    
    def contraction_step(carry):
        labels: Label = carry[0]
        nloc_lab, indices, num_uq = unique_labels(labels, (labels.irank != rank) & valid)
        
        nloc_lab_new = all_to_all_request(
            nloc_lab.irank, nloc_lab.igroup, labels, num=num_uq, axis_name=axis_name, copy_self=False
        )
        labels_new = tree_where((labels.irank != rank) & valid, nloc_lab_new[indices], labels)
        labels_new = tree_where((labels_new.irank == rank) & valid, labels_new[labels_new.igroup], labels_new)

        not_done = jax.lax.psum(~ jnp.all(labels == labels_new), axis_name) > 0
        
        return labels_new, not_done
    
    return jax.lax.while_loop(
        lambda c: c[1], contraction_step, (labels, jnp.array(True))
    )[0]


# ------------------------------------------------------------------------------------------------ #
#                                          User Interface                                          #
# ------------------------------------------------------------------------------------------------ #

def prepare_fof_z(posz: jax.Array, rlink: float, boxsize: float | None = None, 
                  cfg: FofConfig = FofConfig()) -> FofData:

    th: fmdj.data.TreeHierarchy = fmdj.ztree.build_tree_hierarchy(
        posz, cfg_tree=cfg.tree
    )

    igroup, ilist, spl = node_node_fof(
        th, rlink=rlink, boxsize=boxsize, alloc_fac_ilist=cfg.alloc_fac_ilist
    )

    return FofData(rlink, boxsize, posz, igroup, ilist, spl)
prepare_fof_z.jit = jax.jit(prepare_fof_z, static_argnames=["rlink", "boxsize", "cfg"])

def evaluate_fof_z(d: FofData):
    # Could in principle support switching out the particle data at this point.
    return particle_particle_fof(
        d.igroup, d.ilist.ispl, d.ilist.iother, d.spl, d.posz, rlink=d.rlink, boxsize=d.boxsize
    )
evaluate_fof_z.jit = jax.jit(evaluate_fof_z)

def fof_z(posz: jax.Array, rlink: float, boxsize: float = 0., cfg: FofConfig = FofConfig()) -> jax.Array:
    data = prepare_fof_z(posz, rlink, boxsize, cfg)
    return evaluate_fof_z(data)
fof_z.jit = jax.jit(fof_z, static_argnames=["rlink", "boxsize", "cfg"])

def fof(pos: jax.Array, rlink: float, boxsize: float = 0., cfg: FofConfig = FofConfig()) -> jax.Array:
    posz, idz = pos_zorder_sort(pos)
    
    igroupz = fof_z(posz, rlink, boxsize=boxsize, cfg=cfg)

    igroup = igroupz.at[idz].set(igroupz)

    return igroup
fof.jit = jax.jit(fof, static_argnames=["rlink", "boxsize", "cfg"])
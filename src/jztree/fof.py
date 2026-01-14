import numpy as np
import jax
import jax.numpy as jnp
from jztree_cuda import ffi_fof
from .tree import pos_zorder_sort, grouped_dense_interaction_list
from .common import conditional_callback
from .data import  FofConfig, FofData, PosLvl, Label, Link, FofNodeData
from fmdj.data import InteractionList
from typing import Tuple
from fmdj.comm import get_rank_info, pytree_len, all_to_all_with_irank, all_to_all_request, all_to_all_request_children
from fmdj.ztree import simplify_interaction_list
from fmdj.tools import inverse_of_splits, cumsum_starting_with_zero, offset_sum, div_ceil
from dataclasses import dataclass, replace
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

def node_to_child_label(igroup: jax.Array, lvl: jax.Array, spl: jax.Array, 
                        size_child: int, rlink: float, flag_local: jax.Array | None = None,
                        block_size: int = 64) -> jax.Array:
    if flag_local is None:
        flag_local = jnp.ones(igroup.shape, dtype=jnp.bool)

    outputs = (jax.ShapeDtypeStruct((size_child,), jnp.int32),)
    igroup_child = jax.ffi.ffi_call("NodeToChildLabel",  outputs)(
        igroup, flag_local, lvl, spl, block_size=np.uint64(block_size), r2link=np.float32(rlink*rlink)
    )[0]

    return igroup_child
node_to_child_label.jit = jax.jit(node_to_child_label, static_argnames=("size_child", "rlink", "block_size"))

# def level_to_extend(lvl, diag=True):
#     olvl, omod = lvl//3, lvl % 3

#     dx = jnp.ldexp(1., olvl)
#     dy = jnp.ldexp(1., olvl + (omod >= 2).astype(jnp.int32))
#     dz = jnp.ldexp(1., olvl + (omod >= 1).astype(jnp.int32))

#     if diag:
#         return jnp.sqrt(dx*dx + dy*dy + dz*dz)
#     else:
#         return jnp.stack((dx, dy, dz), axis=-1)

# def node_to_child_label2(node_igroup, node_lvl, spl, size, rlink):
#     inode = inverse_of_splits(spl, size)

#     node_is_linked = jnp.arange(len(node_igroup)) != node_igroup
#     node_is_linked = node_is_linked | (level_to_extend(node_lvl, diag=True) <= rlink)

#     idx = jnp.arange(size)
#     igroup = jnp.where(node_is_linked[inode], spl[node_igroup[inode]], idx)

#     return igroup

def node_node_fof(th: fmdj.data.TreeHierarchy, rlink: float, boxsize: float=0., alloc_fac_ilist: int = 128
                  ) -> Tuple[FofNodeData, InteractionList]:
    nplanes = th.num_planes()

    # initialize top-level interaction list
    spl, ilist, nsup = grouped_dense_interaction_list(
        th.lvl.num(nplanes-1), size_ilist=int(th.plane_sizes[nplanes-1]*alloc_fac_ilist), ngroup=32
    )

    # Get coarsest plane data
    igroup = jnp.arange(len(spl)-1, dtype=jnp.int32)

    node_lvl = jnp.full_like(igroup, 388)

    for level in reversed(range(nplanes)):
        size = th.plane_sizes[level]
        igroup = node_to_child_label(igroup, node_lvl, spl, size, rlink=rlink)
        child_data = PosLvl(th.geom_cent.get(level, size), th.lvl.get(level, size))

        igroup, ilist = node_fof_and_ilist(
            ilist, spl, child_data, igroup,
            rlink=rlink, boxsize=boxsize, alloc_fac=alloc_fac_ilist
        )

        spl = th.ispl_n2n.get(level, size+1)
        node_lvl = child_data.lvl
    
    node_data = FofNodeData(node_lvl, igroup, spl, th.lvl.num(0))

    return node_data, ilist
node_node_fof.jit = jax.jit(node_node_fof, static_argnames=["rlink", "boxsize", "alloc_fac_ilist"])

def particle_particle_fof(node_data: FofNodeData, ilist: InteractionList, posz: jax.Array,
                          rlink: float, boxsize: float = 0., block_size=32):
    part_igroup = node_to_child_label(
        node_data.label, node_data.lvl, node_data.spl, len(posz), rlink=rlink
    )

    igroup = jax.ffi.ffi_call("ParticleFof", (jax.ShapeDtypeStruct((len(posz),), part_igroup.dtype),))(
        ilist.ispl, ilist.iother, node_data.spl, posz, part_igroup,
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
    pairs = labels.stacked()

    lab, indices, inv = jnp.unique(
        pairs, axis=0, size=len(pairs), return_index=True, return_inverse=True
    )
    
    return jnp.where(labels.igroup >= 0, indices[inv], labels.igroup)

def masked_unique_pairs(pairs: jax.Array, mask: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Get unique pairs among the masked locations

    returns pairs_unique, indices, num

    indices are so that pars == pairs_unique[indices] at locations where mask is True
    assumes there are no negative values in tuples
    """
    masked_pairs, num, inv_mask = masked_to_dense(
        pairs, mask, get_inverse=True, fill_value=-1
    )
    # To avoid counting invalid pairs as an extra label, we set them to the first valid label
    masked_pairs = jnp.where(masked_pairs == -1, masked_pairs[0], masked_pairs)

    pair_unique, inv = jnp.unique(
        masked_pairs, axis=0, size=len(masked_pairs), return_inverse=True, fill_value=-1
    )

    return pair_unique, inv[inv_mask], jnp.sum(pair_unique[...,0] != -1)

def unique_labels(labels: Label, mask: jax.Array) -> Tuple[Label, jax.Array, jax.Array]:
    lab, inv, num = masked_unique_pairs(labels.stacked(), mask)

    return Label(lab[...,0], lab[...,1]), inv, num


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
                          ) -> Tuple[jax.Array, Label, Link, jax.Array]:
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

    is_root = (labels.irank == rank) & (labels.igroup == jnp.arange(len(labels.igroup)))

    # Dereference again, since labels may have changed
    lmin, lmax, are_local = contract(lA, lB)
    was_resolved = lmin == lmax

    # Handle partially local links. For these we need to update the higher root node to any
    # parent of the lower node
    # Multiple links may try to update a single root node at the same time. To deal with this
    # race condition, we take the minimum suggested update and then need to retry the unresolved
    # in the next iteration where we will continue try them at the larger node

    update = (are_local == 1) & (lmax.irank == rank)  & (~was_resolved)
    update = update & (labels[lmax.igroup] > lmin) & is_root[lmax.igroup]
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

def contract_distributed(labels: Label, num: int | jax.Array):
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

def link_distributed(
        igroup: jax.Array,
        labels: jax.Array,
        links: Link,
        nlabels: jax.Array,
        nlinks: jax.Array
    ):
    rank, ndev, axis_name = get_rank_info()

    def condition(carry):
        igroup, labels, links, nlinks = carry
        return jax.lax.psum(nlinks > 0, axis_name) > 0
    
    init_val = (igroup, labels, links, nlinks)
    igroup, labels, links, nlinks = jax.lax.while_loop(
        condition, lambda c: link_distributed_step(*c), init_val
    )

    return contract_distributed(labels, nlabels)

def distr_node_to_child_label(nodes: FofNodeData, size_child: int, rlink: float) -> jax.Array:
    rank, ndev, axis_name = get_rank_info()

    valid = jnp.arange(len(nodes.lvl)) < nodes.num

    local_child_labels = Label(
        jnp.full(size_child, rank),
        node_to_child_label(nodes.label.igroup, nodes.lvl, nodes.spl, size_child, rlink=rlink,
                            flag_local=(nodes.label.irank == rank) & (valid)),
    )

    inode = inverse_of_splits(nodes.spl, size_child)

    request, inv, num = unique_labels(nodes.label, (nodes.label.irank != rank) & valid)
    remote_child_labels = all_to_all_request(
        request.irank, request.igroup, local_child_labels[nodes.spl[:-1]], num=num, axis_name=axis_name
    )

    child_labels = tree_where(
        nodes.label[inode].irank == rank, local_child_labels, remote_child_labels[inv[inode]]
    )

    return child_labels

def get_min_label(valid: jax.Array, igroup: jax.Array, label: Label):
    """Finds the minimum of labels that are pointed to as the same local group"""
    irankmin = masked_min_scatter(valid, label.irank, igroup, label.irank)
    is_min_rank = valid & (label.irank == irankmin[igroup])
    igroupmin = masked_min_scatter(is_min_rank, label.igroup, igroup, label.igroup)
    return Label(irankmin[igroup], igroupmin[igroup])

def distr_local_to_global_label_change(igroup, igroup_new, label, num_labels):
    rank, ndev, axis_name = get_rank_info()
    
    valid = jnp.arange(len(igroup)) < num_labels

    # First update the global labels that our task knows about
    label_new = get_min_label(valid, igroup_new, label)

    mask = valid & (label != label_new) & ((label_new.irank != rank) | (label.irank != rank))
    pairs = jnp.stack([igroup, igroup_new], axis=-1)
    pairs, inv, num_links = masked_unique_pairs(pairs, mask)

    links = Link(label[pairs[:,0]], label_new[pairs[:,1]])

    label_new = link_distributed(igroup_new, label_new, links, num_labels, num_links)

    return label_new

def linearly_grouped(num, size, ngroup=32):
    num_sup, size_sup = div_ceil(num, ngroup), div_ceil(size, ngroup)
    return jnp.minimum(jnp.arange(size_sup+1) * ngroup, num), num_sup

def distr_fof_top_level(num_local: int, size_local: int, size: int, alloc_fac_ilist: float
                        ) -> Tuple[FofNodeData, InteractionList]:
    rank, ndev, axis_name = get_rank_info()

    # Define splits and their data
    spl, nsuper = linearly_grouped(num_local, size_local, ngroup=32)
    size_sup = len(spl)-1
    labels = Label(
        irank=jnp.full(size_sup, rank, dtype=jnp.int32),
        igroup=jnp.arange(size_sup)
    )
    node_lvl = jnp.full(size_sup, 388)
    node_data = FofNodeData(node_lvl, labels, spl, nsuper)
    
    # define interaction list with remote interactions
    nper_rank = jax.lax.all_gather(nsuper, axis_name)
    
    # due to pruning, only need data from larger tasks
    dev_spl = cumsum_starting_with_zero(nper_rank * (jnp.arange(ndev) >= rank))

    # Define a dense interaction list on top-nodes:
    ilist = fmdj.ztree.dense_interaction_list(
        dev_spl[-1], size, size*alloc_fac_ilist,
        node_range=jnp.array([dev_spl[rank], dev_spl[rank+1]])
    )
    ilist.ids = jnp.arange(size) - dev_spl[inverse_of_splits(dev_spl, size)] # !!! verify size
    ilist.dev_spl = dev_spl

    return node_data, ilist

def distr_node_node_fof(th: fmdj.data.TreeHierarchy, rlink: float, boxsize: float = 0., 
                        alloc_fac_ilist = 32) -> Tuple[FofNodeData, InteractionList]:
    rank, ndev, axis_name = get_rank_info()

    def handle_plane(level: int, node_data: FofNodeData, ilist: InteractionList):
        size = max(th.plane_sizes[level]*2, 4096)

        # Advect node to child data
        labels = distr_node_to_child_label(node_data, size_child=size, rlink=rlink)
        poslvl = PosLvl(th.geom_cent.get(level, size), th.lvl.get(level, size))
        # Request the remote node children that we need to interact with
        (poslvl, ids, labels), spl, dev_spl = all_to_all_request_children(
            ilist.dev_spl, ilist.ids, node_data.spl, (poslvl, jnp.arange(size), labels),
            axis_name=axis_name
        )

        # Do the FoF with local labels
        igroup = global_to_local_label(labels)

        igroup_new, ilist = node_fof_and_ilist(
            ilist, spl, poslvl, igroup,
            rlink=rlink, boxsize=boxsize, alloc_fac=alloc_fac_ilist
        )
        ilist = replace(ilist, ids=ids, dev_spl=dev_spl) # inform the ilist where children lie

        # Communicate the locally detected change in labels
        labels = distr_local_to_global_label_change(
            igroup, igroup_new, labels, dev_spl[-1]
        )

        # Simplify interaction list (reduces unnecessary remote requests)
        ilist = simplify_interaction_list(ilist, th.num(level))

        # Define node-splits for next level
        node_data = FofNodeData(
            th.lvl.get(level, size), labels, th.ispl_n2n.get(level, size+1), th.num(level)
        )

        return node_data, ilist
    
    # Seed with dense interactions at top-level
    node_data, ilist = distr_fof_top_level(
        th.num(th.num_planes()-1), th.plane_sizes[-1]*8, th.plane_sizes[-1]*8, alloc_fac_ilist
    )

    for level in reversed(range(th.num_planes())):
        node_data, ilist = handle_plane(level, node_data, ilist)
    
    return node_data, ilist
distr_node_node_fof.jit = jax.jit(
    distr_node_node_fof, static_argnames=("alloc_fac_ilist", "boxsize", "rlink")
)

# ------------------------------------------------------------------------------------------------ #
#                                          User Interface                                          #
# ------------------------------------------------------------------------------------------------ #

def fof_z(posz: jax.Array, rlink: float, boxsize: float = 0., cfg: FofConfig = FofConfig()) -> jax.Array:
    th = fmdj.ztree.build_tree_hierarchy(posz, cfg_tree=cfg.tree)
    node_data, ilist = node_node_fof(
        th, rlink=rlink, boxsize=boxsize, alloc_fac_ilist=cfg.alloc_fac_ilist
    )
    return particle_particle_fof(
        node_data, ilist, posz, rlink=rlink, boxsize=boxsize
    )
fof_z.jit = jax.jit(fof_z, static_argnames=["rlink", "boxsize", "cfg"])

def fof(pos: jax.Array, rlink: float, boxsize: float = 0., cfg: FofConfig = FofConfig()) -> jax.Array:
    posz, idz = pos_zorder_sort(pos)
    
    igroupz = fof_z(posz, rlink, boxsize=boxsize, cfg=cfg)

    igroup = igroupz.at[idz].set(igroupz)

    return igroup
fof.jit = jax.jit(fof, static_argnames=["rlink", "boxsize", "cfg"])
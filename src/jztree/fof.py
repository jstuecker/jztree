import numpy as np
import jax
import jax.numpy as jnp
from jztree_cuda import ffi_fof
from .tree import pos_zorder_sort, grouped_dense_interaction_list
from .common import conditional_callback
from .data import  FofConfig, FofData, PosLvl, Label, Link, FofNodeData
from fmdj.data import InteractionList, PackedArray
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

def pcast_like_vma(x, like):
    """Make x have the same VMA (varying manual axes) as `like` inside shard_map."""
    # Outside shard_map, VMA is irrelevant; just return x.
    mesh = jax.sharding.get_abstract_mesh()
    if mesh is None:
        return x

    # Keep axis order consistent with the mesh (vma is a set).
    like_vma = getattr(like, "vma", frozenset())
    axes = tuple(ax for ax in mesh.axis_names if ax in like_vma)

    # No varying axes => nothing to do (also covers single-device meshes).
    if not axes:
        return x

    # pvary is deprecated; pcast(..., to="varying") is the supported replacement.
    return jax.lax.pcast(x, axes, to="varying")

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

    child_igroup_out, child_ilist_ispl, child_ilist = pcast_like_vma(res, like=node_ilist.iother)
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
    
    return pcast_like_vma(igr_out, like=igroup)

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

def masked_to_dense(arr: jax.Array, mask, get_inverse=False, get_indices=False, fill_value=0):
    pref, num = offset_sum(mask)
    size = pytree_len(arr)
    pref = jnp.where(mask, pref, size)
    def upd(x):
        return jnp.full(x.shape, fill_value, x.dtype).at[pref].set(x)

    new_arr = jax.tree.map(upd, arr)
    res = [new_arr, num]
    if get_inverse:
        res.append(pref)
    if get_indices:
        ind = jnp.full(size, size, pref.dtype).at[pref].set(jnp.arange(size))
        res.append(ind)
    return res

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

def contract_distributed(labels: Label, igroup: jax.Array, dev_spl: int):
    rank, ndev, axis_name = get_rank_info()

    igroup = igroup[igroup]
    labels = labels[igroup]

    # mask root labels of the local graph that lie remotely
    idx = jnp.arange(len(igroup))
    is_local_root = idx == igroup
    valid = (idx >= dev_spl[rank]) & (idx < dev_spl[rank+1])
    mask = valid & is_local_root & (labels.irank != rank)

    jax.debug.log("n-to-contract: {}", jnp.sum(mask))
    
    def contraction_step(carry):
        labels, mask, _ = carry
        non_loc_lab, num_uq, ind = masked_to_dense(labels, mask, get_indices=True)
        
        nloc_lab_new: Label = all_to_all_request(
            non_loc_lab.irank, non_loc_lab.igroup, labels, num=num_uq, axis_name=axis_name, copy_self=False
        )
        labels_new = Label(
            labels.irank.at[ind].set(nloc_lab_new.irank),
            labels.igroup.at[ind].set(nloc_lab_new.igroup)
        )
        labels_new = labels_new[igroup]

        mask = mask & (labels != labels_new)
        
        any_left = jax.lax.pmax(jnp.any(mask), axis_name)
        
        return labels_new, mask, any_left
    
    return jax.lax.while_loop(
        lambda c: c[2], contraction_step, (labels, mask, jnp.array(True))
    )[0]

def link_distributed(
        igroup: jax.Array,
        labels: jax.Array,
        links: Link,
        dev_spl: jax.Array,
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

    return contract_distributed(labels, igroup, dev_spl)

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

def distr_local_to_global_label_change(igroup, igroup_new, label, dev_spl):
    rank, ndev, axis_name = get_rank_info()
    
    valid = jnp.arange(len(igroup)) < dev_spl[-1]

    # First update the global labels that our task knows about
    label_new = get_min_label(valid, igroup_new, label)

    # find globally relevant root nodes that became linked
    was_root = jnp.arange(len(igroup)) == igroup
    remote = (label_new.irank != rank) | (label.irank != rank)
    mask = (label != label_new) & was_root & remote & valid
    indices = jnp.where(mask, size=len(igroup))
    num_links = jnp.sum(mask)
    
    links = Link(label[indices], label_new[indices])

    label_new = link_distributed(igroup_new, label_new, links, dev_spl, num_links)

    return label_new

def distr_detect_new_cross_task_links(igroup, igroup_new, origin_group, dev_spl) -> Link:
    rank, ndev, axis_name = get_rank_info()
    
    valid = jnp.arange(len(igroup)) < dev_spl[-1]

    # find globally relevant root nodes that became linked
    was_root = jnp.arange(len(igroup)) == igroup
    irank = inverse_of_splits(dev_spl, igroup.size)
    remote = irank != rank
    mask = (igroup != igroup_new) & was_root & remote & valid
    indices = jnp.where(mask, size=len(igroup))
    num_links = jnp.sum(mask)
    
    links = Link(
        Label(irank[indices], origin_group[indices]),
        Label(irank[igroup_new[indices]], origin_group[igroup_new[indices]]),
    )

    return links, num_links

def linearly_grouped(num, size, ngroup=32):
    num_sup = div_ceil(num, ngroup)
    return jnp.minimum(jnp.arange(size+1) * ngroup, num), num_sup

def distr_fof_top_level(num_local: int, size: int, alloc_fac_ilist: float
                        ) -> Tuple[FofNodeData, InteractionList]:
    rank, ndev, axis_name = get_rank_info()

    # Define splits and their data
    spl, nsuper = linearly_grouped(num_local, size, ngroup=32)
    
    labels = jax.lax.pcast(jnp.arange(size), axis_name, to="varying")
    node_lvl = jax.lax.pcast(jnp.full(size, 388), axis_name, to="varying")
    node_data = FofNodeData(node_lvl, labels, spl, nsuper)
    
    # define interaction list with remote interactions
    nper_rank = jax.lax.all_gather(nsuper, axis_name)
    
    # due to pruning, only need data from larger tasks
    dev_spl = cumsum_starting_with_zero(nper_rank * (jnp.arange(ndev) >= rank))

    # Define a dense interaction list on top-nodes:
    ilist = fmdj.ztree.dense_interaction_list(
        dev_spl[-1], size, int(size*alloc_fac_ilist),
        node_range=jnp.array([dev_spl[rank], dev_spl[rank+1]])
    )
    ilist.ids = jnp.arange(size) - dev_spl[inverse_of_splits(dev_spl, size)] # !!! verify size
    ilist.dev_spl = dev_spl
    
    return node_data, ilist

def distr_node_node_fof(th: fmdj.data.TreeHierarchy, rlink: float, boxsize: float = 0., 
                        alloc_fac_ilist = 32, size_links = None) -> Tuple[FofNodeData, InteractionList]:
    rank, ndev, axis_name = get_rank_info()

    size = th.plane_sizes[0]*2
    if size_links is  None:
        size_links = size

    def handle_plane(level: int, node_data: FofNodeData, ilist: InteractionList, link_data: PackedArray):
        igroup = node_to_child_label(node_data.label, node_data.lvl, node_data.spl, size, rlink=rlink) 

        poslvl = PosLvl(th.geom_cent.get(level, size), th.lvl.get(level, size))
        l2p = th.ispl_n2n.get(0, size+1)
        leaf_id = l2p[th.ispl_n2l.get(level, size)]
        # Request the remote node children that we need to interact with
        (poslvl, ids, leaf_id), spl, dev_spl = all_to_all_request_children(
            ilist.dev_spl, ilist.ids, node_data.spl, (poslvl, jnp.arange(size), leaf_id),
            axis_name=axis_name
        )

        # Treat remote nodes as roots
        irank = inverse_of_splits(dev_spl, size)
        igroup = jnp.where(irank==rank, igroup, jnp.arange(size))

        igroup_new, ilist = node_fof_and_ilist(
            ilist, spl, poslvl, igroup,
            rlink=rlink, boxsize=boxsize, alloc_fac=alloc_fac_ilist
        )
        ilist = replace(ilist, ids=ids, dev_spl=dev_spl) # inform the ilist where children lie

        links, num_links = distr_detect_new_cross_task_links(igroup, igroup_new, leaf_id, dev_spl)
        link_data = link_data.append(links.stacked(axis=-1), num_links)

        # Simplify interaction list (reduces unnecessary remote requests)
        ilist = simplify_interaction_list(ilist, th.num(level))

        # Define node-splits for next level
        node_data = FofNodeData(
            th.lvl.get(level, size), igroup_new, th.ispl_n2n.get(level, size+1), th.num(level)
        )

        return node_data, ilist, link_data
    
    # Seed with dense interactions at top-level
    node_data, ilist = distr_fof_top_level(th.num(th.num_planes()-1), size, alloc_fac_ilist)
    
    link_data = PackedArray(pcast_like_vma(jnp.zeros((size_links,4), dtype=jnp.int32), node_data.lvl), levels=th.num_planes()+1)
    link_data.ispl = jax.lax.pcast(link_data.ispl, axis_name, to="varying")

    # for level in reversed(range(th.num_planes())):
    #     node_data, ilist = handle_plane(level, node_data, ilist)

    def loop_body(i, carry):
        return handle_plane(th.num_planes()-i-1, *carry)
    node_data, ilist, link_data = jax.lax.fori_loop(0, th.num_planes(), loop_body, (node_data, ilist, link_data))

    return node_data, ilist, link_data
distr_node_node_fof.jit = jax.jit(
    distr_node_node_fof, static_argnames=("alloc_fac_ilist", "boxsize", "rlink", "size_links")
)

def distr_particle_particle_fof(node_data: FofNodeData, ilist: InteractionList, 
                                link_data: PackedArray, posz: jax.Array,
                                rlink: float, boxsize: float = 0., block_size=32) -> Label:
    rank, ndev, axis_name = get_rank_info()
    size = len(posz)

    igroup = node_to_child_label(node_data.label, node_data.lvl, node_data.spl, size, rlink=rlink)

    numpart = node_data.spl[-1]
    (posz, pids), spl, dev_spl = all_to_all_request_children(
        ilist.dev_spl, ilist.ids, node_data.spl, (posz, jnp.arange(size)), axis_name=axis_name
    )

    # Treat remote nodes as roots
    irank = inverse_of_splits(dev_spl, size)
    igroup = jnp.where(irank==rank, igroup, jnp.arange(size))

    igroup_new = jax.ffi.ffi_call("ParticleFof", (jax.ShapeDtypeStruct((len(posz),), igroup.dtype),))(
        ilist.ispl, ilist.iother, spl, posz, igroup,
        r2link=np.float32(rlink*rlink), boxsize=np.float32(boxsize), block_size=np.int32(block_size)
    )[0]

    # insert new links
    links, num_links = distr_detect_new_cross_task_links(igroup, igroup_new, pids, dev_spl)
    link_data = link_data.append(links.stacked(axis=-1), num_links)
    links = Link.from_stacked(link_data.data)

    jax.debug.log("rank {} nlinks: {}", rank, link_data.ispl)

    # Infer global labels
    labels = Label(jnp.full(igroup.shape, rank, dtype=jnp.int32), igroup)
    labels = link_distributed(igroup_new, labels, links, dev_spl, link_data.ispl[-1])
    
    labels = tree_where(jnp.arange(len(labels.igroup)) < numpart, labels, Label(-1,-1))

    return labels
distr_particle_particle_fof.jit = jax.jit(distr_particle_particle_fof, static_argnames=["rlink", "boxsize", "block_size"])

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

def distr_fof_z_with_tree(
        posz: jax.Array, th: fmdj.data.TreeHierarchy, rlink: float, 
        boxsize: float = 0., cfg: FofConfig = FofConfig(), linearize_labels: bool = False
    ) -> Label:
    """
    linearize_labels: only for testing against single-gpu version - converts (irank,igroup) labels to 
        dense global linear ones
    """
    rank, ndev, axis_name = get_rank_info()

    node_data, ilist, link_data = distr_node_node_fof(
        th, rlink=rlink, boxsize=boxsize, alloc_fac_ilist=cfg.alloc_fac_ilist, size_links=len(posz)
    )

    labels = distr_particle_particle_fof(node_data, ilist, link_data, posz, rlink=rlink, boxsize=boxsize)

    if linearize_labels:
        num = jax.lax.all_gather(jnp.sum(~jnp.isnan(posz[...,0]), axis=0), axis_name)
        dspl = cumsum_starting_with_zero(num)
        igroup = dspl[labels.irank] + labels.igroup

        return igroup
    else:
        return labels
distr_fof_z_with_tree.jit = jax.jit(
    distr_fof_z_with_tree, static_argnames=["cfg", "rlink", "boxsize", "linearize_labels"]
)

def distr_fof(part: fmdj.data.Pos, npart_tot: int, rlink: float, boxsize: float = 0., 
              cfg: FofConfig = FofConfig):
    partz, th = fmdj.ztree.distr_zsort_and_tree(part, npart_tot, cfg.tree)

    labels = distr_fof_z_with_tree(partz.pos, th, rlink, boxsize, cfg)

    return partz, labels
import numpy as np
from typing import Tuple
import jax
import jax.numpy as jnp
from dataclasses import replace

from .config import FofConfig, FofCatalogueConfig
from .data import  FofData, PosLvl, Label, Link, FofNodeData, ParticleData, FofCatalogue
from .data import InteractionList, PackedArray, TreeHierarchy, Pos, get_num, verify_ilist
from .tools import inverse_of_splits, cumsum_starting_with_zero, offset_sum, div_ceil
from .tools import bucket_prefix_sum
from .tree import pos_zorder_sort, grouped_dense_interaction_list, build_tree_hierarchy
from .tree import simplify_interaction_list, dense_interaction_list, distr_zsort_and_tree
from .comm import pytree_len, all_to_all_with_irank, all_to_all_request
from .comm import all_to_all_request_children, all_to_all_with_splits
from .jax_ext import pcast_vma, pcast_like, get_rank_info, shard_map_constructor, tree_map_by_len
from .jax_ext import raise_if
from .stats import statistics, stats_callback, AllocStats

from jztree_cuda import ffi_fof
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

    child_igroup_out, child_ilist_ispl, child_ilist = pcast_like(res, like=node_ilist.iother)
    child_ilist = verify_ilist(InteractionList(child_ilist_ispl, child_ilist))

    n1, n2 = child_ilist.nfilled(), child_ilist.iother.shape[0]
    child_igroup_out = child_igroup_out + raise_if(n1 > n2,
        "The interaction list allocation is too small. (need: {n1} have: {n2})\n"
        "Hint: Increase alloc_fac_ilist at least by a factor of {ratio:.1f}",
        n1=n1, n2=n2, ratio=n1/n2
    )

    stats_callback("allocation", AllocStats.record_filled_interactions, n1, n2)

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
    
    return pcast_like(igr_out, like=igroup)

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

def node_node_fof(th: TreeHierarchy, rlink: float, boxsize: float=0., alloc_fac_ilist: int = 128
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
        child_data = PosLvl(pos=th.geom_cent.get(level, size), lvl=th.lvl.get(level, size))

        igroup, ilist = node_fof_and_ilist(
            ilist, spl, child_data, igroup,
            rlink=rlink, boxsize=boxsize, alloc_fac=alloc_fac_ilist
        )

        spl = th.ispl_n2n.get(level, size+1)
        node_lvl = child_data.lvl
    
    node_data = FofNodeData(node_lvl, igroup, spl)

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
#                                         Helper Functions                                         #
# ------------------------------------------------------------------------------------------------ #

def global_to_local_label(labels: Label) -> jax.Array:
    """Map each global to the index of the first occurence of it"""
    # Note, currently the Fof self-linking detection can be wrong for index 0!
    pairs = labels.stacked()

    lab, indices, inv = jnp.unique(
        pairs, axis=0, size=len(pairs), return_index=True, return_inverse=True
    )
    
    return jnp.where(labels.igroup >= 0, indices[inv], labels.igroup)

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

def get_min_label(valid: jax.Array, igroup: jax.Array, label: Label):
    """Finds the minimum of labels that are pointed to as the same local group"""
    irankmin = masked_min_scatter(valid, label.irank, igroup, label.irank)
    is_min_rank = valid & (label.irank == irankmin[igroup])
    igroupmin = masked_min_scatter(is_min_rank, label.igroup, igroup, label.igroup)
    return Label(irankmin[igroup], igroupmin[igroup])

def fof_is_superset(igroup_sup, igroup, mask = None):
    """Checks whether every FoF group in igroup_up is a superset of sets in igroup_low"""
    # For this we need to check that if we link groups together as indicated by the super-grouping
    # that they are identical to the super groups

    # indicates the super group of each label in igroup
    label_map = jnp.zeros(len(igroup), dtype=jnp.int32).at[igroup].set(igroup_sup)
    
    if mask is None:
        return jnp.all(igroup_sup == label_map[igroup])
    else:
        return jnp.all((igroup_sup == label_map[igroup]) | ~mask)

# ------------------------------------------------------------------------------------------------ #
#                                        Distributed Linking                                       #
# ------------------------------------------------------------------------------------------------ #

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
        send_rank, links, num=nlinks, axis_name=axis_name, copy_self=False,
        pack_pytree=True
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

# ------------------------------------------------------------------------------------------------ #
#                                          Distributed FoF                                         #
# ------------------------------------------------------------------------------------------------ #

def linearly_grouped(num, size, ngroup=32):
    num_sup = div_ceil(num, ngroup)
    return jnp.minimum(jnp.arange(size+1) * ngroup, num), num_sup

def distr_fof_top_level(num_local: int, size: int, alloc_fac_ilist: float
                        ) -> Tuple[FofNodeData, InteractionList]:
    rank, ndev, axis_name = get_rank_info()

    # Define splits and their data
    spl, nsuper = linearly_grouped(num_local, size, ngroup=32)
    
    labels = pcast_vma(jnp.arange(size), axis_name)
    node_lvl = pcast_vma(jnp.full(size, 388), axis_name)
    node_data = FofNodeData(node_lvl, labels, spl)
    
    # define interaction list with remote interactions
    nper_rank = jax.lax.all_gather(nsuper, axis_name)
    
    # due to pruning, only need data from larger tasks
    dev_spl = cumsum_starting_with_zero(nper_rank * (jnp.arange(ndev) >= rank))

    # Define a dense interaction list on top-nodes:
    ilist = dense_interaction_list(
        dev_spl[-1], size, int(size*alloc_fac_ilist),
        node_range=jnp.array([dev_spl[rank], dev_spl[rank+1]])
    )
    ilist.ids = jnp.arange(size) - dev_spl[inverse_of_splits(dev_spl, size)] # !!! verify size
    ilist.dev_spl = dev_spl
    
    return node_data, ilist

def distr_node_node_fof(th: TreeHierarchy, rlink: float, boxsize: float = 0., 
                        alloc_fac_ilist = 32, size_links = None
                        ) -> Tuple[FofNodeData, InteractionList, PackedArray]:
    rank, ndev, axis_name = get_rank_info()

    size = th.plane_sizes[0]*2
    if size_links is  None:
        size_links = size

    l2p = th.ispl_n2n.get(0, size+1)

    def handle_plane(level: int, node_data: FofNodeData, ilist: InteractionList, link_data: PackedArray):
        igroup = node_to_child_label(node_data.label, node_data.lvl, node_data.spl, size, rlink=rlink) 

        poslvl = PosLvl(pos=th.geom_cent.get(level, size), lvl=th.lvl.get(level, size))
        pid = l2p[th.ispl_n2l.get(level, size)] # first particle id in each node
        
        # Request the remote node children that we need to interact with
        (poslvl, ids, pid), spl, dev_spl = all_to_all_request_children(
            ilist.dev_spl, ilist.ids, node_data.spl, (poslvl, jnp.arange(size), pid),
            axis_name=axis_name
        )

        # Treat remote nodes as roots
        irank = inverse_of_splits(dev_spl, size)
        igroup = jnp.where(irank==rank, igroup, jnp.arange(size))

        igroup_new, ilist = node_fof_and_ilist(
            ilist, spl, poslvl, igroup,
            rlink=rlink, boxsize=boxsize, alloc_fac=alloc_fac_ilist
        )
        ilist = replace(ilist, ids=ids, dev_spl=dev_spl) # save the node origins in ilist

        links, num_links = distr_detect_new_cross_task_links(igroup, igroup_new, pid, dev_spl)
        link_data = link_data.append(links.stacked(axis=-1), num_links)

        # Simplify interaction list (reduces unnecessary remote requests)
        ilist = simplify_interaction_list(ilist, th.num(level))

        # Define node-splits for next level
        node_data = FofNodeData(
            th.lvl.get(level, size), igroup_new, th.ispl_n2n.get(level, size+1)
        )

        return node_data, ilist, link_data
    
    # Seed with dense interactions at top-level
    node_data, ilist = distr_fof_top_level(th.num(th.num_planes()-1), size, alloc_fac_ilist)
    
    # Set up an empty PackedArray to save link data and mark as varying per gpu
    link_data = PackedArray.create_empty(
        (size_links, 4), levels=th.num_planes()+1, dtype=jnp.int32, vma=jax.typeof(rank).vma
    )

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

    stats_callback("allocation", AllocStats.record_filled_links, link_data.nfilled(), link_data.size())

    links = Link.from_stacked(link_data.data)

    # Infer global labels
    labels = Label(jnp.full(igroup.shape, rank, dtype=jnp.int32), igroup)
    labels = link_distributed(igroup_new, labels, links, dev_spl, link_data.ispl[-1])
    
    labels = tree_where(jnp.arange(len(labels.igroup)) < numpart, labels, Label(-1,-1))

    return labels
distr_particle_particle_fof.jit = jax.jit(distr_particle_particle_fof, static_argnames=["rlink", "boxsize", "block_size"])

# ------------------------------------------------------------------------------------------------ #
#                                   Ordering particles by groups                                   #
# ------------------------------------------------------------------------------------------------ #

def fof_order(igroup: jax.Array, part: ParticleData, npart: int | None = None):
    if npart is None:
        npart = jnp.sum(~jnp.isnan(part.pos[...,0]))
    igr = jnp.where(jnp.arange(len(igroup)) < npart, igroup, len(igroup))
    counts = jnp.zeros(len(igroup), dtype=igroup.dtype).at[igr].add(1)

    isort = jnp.argsort(igr, stable=True)
    part, counts = tree_map_by_len(lambda x: x[isort], (part, counts), len(counts))

    return part, counts

def distr_fof_order(label: Label, part: ParticleData, size_out: int | None = None):
    """Rearanges particles in group-order and determines group-count at root-particles

    Group order means that each FoF-group contains group-count continguous particles, 
    starting at its root particle. Group roots are in z-order.
    Guarantees equal load balance if npart_tot % ndev == 0. Therefore, if positions were padded
    previously (to allow for load-imbalance) it is now possible to undo the padding by providing 
    size_out = npart_tot // ndev
    The last group on each device may span across one or more consecutive devices. This needs to be
    accounted for when e.g. summing group information
    """
    # Labels point towards the root of a group, which may lie on another task
    # Groups may be split over several tasks. We call disjoint parts of the group
    # that may lie on other tasks "segments". To count particles we first count
    # in segments and then send the segments to the root task
    rank, ndev, axis_name = get_rank_info()

    npart = get_num(part)
    size = len(label.igroup)

    iseg = global_to_local_label(label)

    # Count locally known roots
    is_segment_root = iseg == jnp.arange(len(iseg))
    seg_idx, num_segs = offset_sum(is_segment_root)
    seg_idx = jnp.where(jnp.arange(size) < npart, seg_idx[iseg], size) # mask invalid particles
    seg_counts = jnp.zeros(size, dtype=jnp.int32).at[seg_idx].add(1)

    # Send segments to root task
    segment_inv = jnp.where(is_segment_root, size=size, fill_value=size)[0]
    seg_label = label[segment_inv]
    
    (seg_counts, seg_igroup), dev_spl, inv = all_to_all_with_irank(
        seg_label.irank, (seg_counts, seg_label.igroup), 
        num=num_segs, get_inverse=True, axis_name=axis_name, pack_pytree=True
    )
    group_counts = jnp.zeros(size, dtype=jnp.int32).at[seg_idx[seg_igroup]].add(seg_counts)

    group_offsets = cumsum_starting_with_zero(group_counts)
    # Mark counts at root particles for later
    idx = jnp.arange(size)
    is_group_root = (label.igroup == idx) & (label.irank == rank) & (idx < npart)
    root_group_counts = jnp.where(is_group_root, group_counts[seg_idx], 0)

    dev_counts = jax.lax.all_gather(jnp.sum(group_counts), axis_name)
    seg_offsets = group_offsets[seg_idx[seg_igroup]] + bucket_prefix_sum(seg_igroup, seg_counts, num=dev_spl[-1])

    # send back offsets
    dense_seg_offsets, dev_spl = all_to_all_with_splits(
        seg_offsets, dev_spl, axis_name=axis_name
    )
    seg_offsets = dense_seg_offsets[inv]

    with jax.enable_x64():
        dev_offsets = cumsum_starting_with_zero(jnp.astype(dev_counts, jnp.int64))
        seg_offsets = jnp.astype(dense_seg_offsets[inv], jnp.int64) + dev_offsets[seg_label.irank]
        
        part_gid = seg_offsets[seg_idx] + bucket_prefix_sum(iseg, num=npart)

        nparttot = dev_offsets[-1]
        target_global_dev_spl = jnp.pad(jnp.arange(ndev) * (nparttot // ndev), (0,1), constant_values=nparttot)

        send_irank = jnp.searchsorted(target_global_dev_spl, part_gid, side="right") - 1
        (gid, part, gcnt), dev_spl = all_to_all_with_irank(
            send_irank, (part_gid, part, root_group_counts), num=npart, pack_pytree=True
        )

        valid = jnp.arange(len(gid)) < dev_spl[-1]
        itarget = jnp.where(valid, gid - target_global_dev_spl[rank], jnp.arange(len(gid)))

        isort = jnp.zeros_like(itarget).at[itarget].set(jnp.arange(size))
        
        part, gcnt = tree_map_by_len(lambda x: x[isort], (part, gcnt), len(gcnt))
    
    if size_out is not None: # Coding-note: might save some space if already done in comm. output:
        part, gcnt = tree_map_by_len(lambda x: x[:size_out], (part, gcnt), len(gcnt))
    
    part.num = dev_spl[-1]

    return part, gcnt

# ------------------------------------------------------------------------------------------------ #
#                                      Fof Catalogue Reduction                                     #
# ------------------------------------------------------------------------------------------------ #

def distr_cross_task_group_info(group_counts: jax.Array, npart: int, npart_min: int = 20):
    """The last group of each rank may span accross ranks. Here, we identify for each
    rank the rank and the local remaining count of the first group which may have started elsewhere.
    """
    rank, ndev, axis_name = get_rank_info()
    if ndev == 1:
        return 0, 0

    idx = jnp.arange(len(group_counts))
    last_group_start = jnp.max(jnp.where((group_counts > 0) & (idx < npart), idx, 0))
    last_group_count = group_counts[last_group_start]

    with jax.enable_x64():
        dev_npart = jax.lax.all_gather(npart, axis_name)
        dev_spl = cumsum_starting_with_zero(jnp.astype(dev_npart, jnp.int64))

        gl_last_group_end = dev_spl[rank] + last_group_start + last_group_count
        dev_add_count = gl_last_group_end - dev_spl[:-1]
        dev_add_count = jnp.where((last_group_count < npart_min) | (jnp.arange(ndev) <= rank), 0, dev_add_count)
        dev_add_count = jnp.minimum(dev_add_count, dev_npart)

        dev_recv_count = jax.lax.all_to_all(dev_add_count, axis_name, 0, 0, tiled=True)

    first_group_count = jnp.max(dev_recv_count.astype(jnp.int32))
    first_group_rank = jnp.argmax(dev_recv_count.astype(jnp.int32))

    return first_group_rank, first_group_count

def fof_catalogue_from_groups(
        part: ParticleData, # Particles must be in Group order! (See fof_order/distr_fof_order)
        group_counts: jax.Array,
        cfg_cata: FofCatalogueConfig = FofCatalogueConfig(),
        boxsize: float = 0.,
        size_cata: int | None = None
    ) -> FofCatalogue:
    """Reduces particle data to determine a Friends-of-Friends group-catalogue

    Some parts of the catalogue will be set to None if relevant particle attributes are missing.
    You may define your on data class that skips attributes that you are not interested in.
    Consider the "ParticleData" class to understand relevant attributes.
    """
    rank, ndev, axis_name = get_rank_info()

    npart = get_num(part, default_to_length=True)
    size_part = len(group_counts)
    npart_min = cfg_cata.npart_min

    if size_cata is None:
        size_cata = (size_part + npart_min - 1) // npart_min # Worst case estimate of catalogue size
    
    first_group_rank, first_group_count = distr_cross_task_group_info(
        group_counts, npart, npart_min=npart_min
    )
    
    # To simplify reductions we add an extra group at the beginning that carries everything
    # that belongs to the previous task
    keep_as_group = group_counts >= npart_min

    # Create dense information
    csum = jnp.cumsum(keep_as_group.astype(jnp.int32))
    part_gr_idx, ngroups = csum, csum[-1]

    gr_start = jnp.where(keep_as_group, fill_value=size_part, size=size_cata)[0]
    gr_valid = jnp.arange(size_cata) < ngroups
    gr_counts = group_counts[gr_start] * gr_valid

    # add in extra group at beginning, we'll remove it later
    gr_start = jnp.pad(gr_start, (1,0), constant_values=0)
    gr_valid = jnp.pad(gr_valid, (1,0), constant_values=True)
    gr_counts = jnp.pad(gr_counts, (1,0), constant_values=first_group_count)

    idx = jnp.arange(size_part)
    part_valid = (part_gr_idx >= 0) & (idx < npart) &  (idx < gr_start[part_gr_idx] + gr_counts[part_gr_idx])
    part_gr_idx = jnp.where(part_valid, part_gr_idx, size_cata+1)

    def sum_particles(val, invalid_val=0):
        gr_val = jnp.zeros((size_cata+1,) + val.shape[1:], val.dtype)
        gr_val = gr_val.at[part_gr_idx].add(val)
        valid = gr_valid.reshape((size_cata+1,) + (1,) * (gr_val.ndim-1))
        gr_val = jnp.where(valid, gr_val, invalid_val)
        if ndev == 1: return gr_val
        # Correct the last group by adding segments on other tasks
        dev_mask = (np.arange(ndev) == first_group_rank).reshape((-1,) + (1,) * (gr_val.ndim-1))
        send_val = jnp.where(dev_mask, gr_val[0], 0)
        add_last_val = jax.lax.all_to_all(send_val, axis_name, 0, 0, tiled=True)
        return gr_val.at[ngroups].add(jnp.sum(add_last_val, axis=0))
    def get_gr_prop(val):
        if ndev==1: return val
        last_val = jax.lax.all_gather(val[ngroups], axis_name)
        return val.at[0].set(last_val[first_group_rank])
    def wrap_dx(dx):
        if boxsize:
            return ((dx + 0.5*boxsize) % boxsize) - 0.5 * boxsize
        else:
            return dx
    def wrap_pos(x):
        return x % boxsize if boxsize else x
    
    cata = FofCatalogue(ngroups=ngroups, count=gr_counts, offset=gr_start)

    if getattr(part, "mass", None) is not None:
        part_mass = getattr(part, "mass")
        if part_mass.size == 1: # constant masses
            gr_mass = cata.count * part_mass
        else:
            gr_mass = sum_particles(part_mass)
        cata.mass = gr_mass
    else:
        part_mass = jnp.array(1., dtype=part.pos.dtype)
        gr_mass = cata.count * part_mass

    if getattr(part, "pos", None) is not None:
        pos0 = get_gr_prop(part.pos[gr_start])
        m_x_pos = sum_particles(wrap_dx(part.pos - pos0[part_gr_idx]) * part_mass[...,None])
        cata.com_pos = wrap_pos((m_x_pos / gr_mass[:,None]) + pos0)

        dx = wrap_dx(part.pos - get_gr_prop(cata.com_pos)[part_gr_idx])
        m_x_r2 = sum_particles((dx[...,0]**2 + dx[...,1]**2 + dx[...,2]**2) * part_mass)
        cata.com_inertia_radius = jnp.sqrt(m_x_r2 / gr_mass)

    if getattr(part, "vel", None) is not None:
        vel0 = get_gr_prop(part.vel[gr_start])
        m_x_vel = sum_particles((part.vel - vel0[part_gr_idx]) * part_mass[...,None])
        cata.com_vel = (m_x_vel / gr_mass[:,None]) + vel0

    # remove the first "fake" group that we inserted:
    def remove_first(x):
        return x[1:] if len(x) == size_cata+1 else x
    cata = tree_map_by_len(remove_first, cata, len(cata.count))

    return cata

# ------------------------------------------------------------------------------------------------ #
#                                          User Interface                                          #
# ------------------------------------------------------------------------------------------------ #

def fof_labels_z(posz: jax.Array, rlink: float, boxsize: float = 0., cfg: FofConfig = FofConfig()) -> jax.Array:
    th = build_tree_hierarchy(posz, cfg_tree=cfg.tree)
    node_data, ilist = node_node_fof(
        th, rlink=rlink, boxsize=boxsize, alloc_fac_ilist=cfg.alloc_fac_ilist
    )
    return particle_particle_fof(
        node_data, ilist, posz, rlink=rlink, boxsize=boxsize
    )
fof_labels_z.jit = jax.jit(fof_labels_z, static_argnames=["rlink", "boxsize", "cfg"])

def fof_labels(pos: jax.Array, rlink: float, boxsize: float = 0., cfg: FofConfig = FofConfig()) -> jax.Array:
    posz, idz = pos_zorder_sort(pos)
    
    igroupz = fof_labels_z(posz, rlink, boxsize=boxsize, cfg=cfg)

    inv_sort = jnp.zeros(len(idz)).at[idz].set(jnp.arange(len(idz)))
    igroup = igroupz.at[idz].set(idz[igroupz])

    return igroup
fof_labels.jit = jax.jit(fof_labels, static_argnames=["rlink", "boxsize", "cfg"])

def distr_fof_z_with_tree(
        posz: jax.Array, th: TreeHierarchy, rlink: float, 
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

def distr_fof(part: Pos, rlink: float, boxsize: float = 0., 
              cfg: FofConfig = FofConfig):
    partz, th = distr_zsort_and_tree(part, cfg.tree)

    labels = distr_fof_z_with_tree(partz.pos, th, rlink, boxsize, cfg)

    return partz, labels

# ------------------------------------------------------------------------------------------------ #
#                                        New User Interface                                        #
# ------------------------------------------------------------------------------------------------ #

def fof_and_catalogue(
        part: ParticleData,
        rlink: float,
        boxsize: float=0.,
        cfg: FofConfig = FofConfig(),
        input_z_ordered: bool = False
    ) -> Tuple[ParticleData, FofCatalogue]:
    """Returns particles in FoF-order and the FoFCatalogue"""
    rank, ndev, axis_name = get_rank_info()
    assert ndev == 1, "For distributed mode, please use distr_fof_and_catalogue"

    if input_z_ordered:
        partz = part
    else:
        partz = pos_zorder_sort(part)[0]
    igroup = fof_labels_z(partz.pos, rlink=rlink, boxsize=boxsize, cfg=cfg)
    partf, counts = fof_order(igroup, partz)
    catalogue = fof_catalogue_from_groups(partf, counts, cfg.catalogue, boxsize=boxsize)

    return partf, catalogue
fof_and_catalogue.jit = jax.jit(fof_and_catalogue,
    static_argnames=["rlink", "boxsize", "cfg", "input_z_ordered"]
)

from jax.sharding import PartitionSpec as P
def distr_fof_and_catalogue(
        part: ParticleData,
        rlink: float,
        boxsize: float=0.,
        cfg: FofConfig = FofConfig(),
        input_z_ordered: bool = False,
        th: TreeHierarchy | None = None
    ) -> Tuple[ParticleData, FofCatalogue]:
    """Returns particles in FoF-order and the FoFCatalogue"""
    assert len(part.pos) < 2**31, "Allocation too large... may lead to int32 overflows"

    if input_z_ordered:
        partz = part
        assert th is not None, "To skip sort, provide tree (jztree.tree.distr_zsort_and_tree)"
    else:
        partz, th = distr_zsort_and_tree(part, cfg.tree)
    labels = distr_fof_z_with_tree(partz.pos, th, rlink=rlink, boxsize=boxsize, cfg=cfg)
    partf, counts = distr_fof_order(labels, partz)
    catalogue = fof_catalogue_from_groups(partf, counts, cfg.catalogue, boxsize=boxsize)

    return partf, catalogue
distr_fof_and_catalogue.smap = shard_map_constructor(distr_fof_and_catalogue,
    in_specs=(P(-1), None, None, None, None, P(-1)),
    static_argnames=["rlink", "boxsize", "cfg", "input_z_ordered"]
)
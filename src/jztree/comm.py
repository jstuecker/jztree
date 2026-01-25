from dataclasses import dataclass
import numpy as np
import os

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from typing import Tuple, Any, TypeAlias

from .tools import cumsum_starting_with_zero, inverse_of_splits, raise_if

# Currently jax doesn't have a typehint for pytrees. We simply define one ourselves for clarity
Pytree: TypeAlias = Any

# ------------------------------------------------------------------------------------------------ #
#                                           Type helpers                                           #
# ------------------------------------------------------------------------------------------------ #

def pcast_vma(x, vma):
    if hasattr(jax.lax, "pcast"):
        return jax.lax.pcast(x, tuple(vma), to="varying")
    else:
        return jax.lax.pvary(x, tuple(vma))

def pcast_like(x, like):
    if hasattr(jax.lax, "pcast"):
        return jax.lax.pcast(x, tuple(jax.typeof(like).vma), to="varying")
    else:
        return jax.lax.pvary(x, tuple(jax.typeof(like).vma))

# ------------------------------------------------------------------------------------------------ #
#                                        General Device Info                                       #
# ------------------------------------------------------------------------------------------------ #

def get_rank_info(axis_name = None) -> Tuple[int, int, str]:
    axis_name = axis_name or jax.sharding.get_abstract_mesh().axis_names
    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    return rank, ndev, axis_name

# ------------------------------------------------------------------------------------------------ #
#                                Distributed Initialization Helpers                                #
# ------------------------------------------------------------------------------------------------ #

def _env_int(name: str, default: int = 0) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default

def should_init_jax_distributed() -> bool:
    # --- Slurm ---
    # SLURM_NTASKS is total tasks; if >1 we’re distributed.
    if _env_int("SLURM_NTASKS", 1) > 1:
        return True

    # --- Open MPI ---
    if _env_int("OMPI_COMM_WORLD_SIZE", 1) > 1:
        return True

    # --- MPICH / PMI-based launchers (incl. some Slurm/MPI setups) ---
    if _env_int("PMI_SIZE", 1) > 1 or _env_int("PMIX_SIZE", 1) > 1:
        return True

    # --- Generic “world size” used by some launchers ---
    if _env_int("WORLD_SIZE", 1) > 1:
        return True

    # --- “Explicit JAX distributed config present” heuristic ---
    # If you set these yourself in your job wrapper, treat it as distributed.
    if any(k in os.environ for k in ("JAX_PROCESS_COUNT", "JAX_PROCESS_INDEX", "JAX_COORDINATOR_ADDRESS")):
        # Only treat it as distributed if it’s actually >1 (when provided).
        pc = _env_int("JAX_PROCESS_COUNT", 1)
        return pc > 1

    return False

# ------------------------------------------------------------------------------------------------ #
#                                       Tiny Helper Functions                                      #
# ------------------------------------------------------------------------------------------------ #

def pytree_len(x):
    """Returns the leading axis of the first leaf of a pytree"""
    leaves = jax.tree_util.tree_leaves(x)

    return len(leaves[0])

def value_for_dtype(dtype, float_val=jnp.nan, int_val=0):
    if dtype.kind == "f":
        return float_val
    else:
        return int_val

def empty_like(x, float_val=jnp.nan, int_val=0):
    def empty_el(xi):
        return jnp.full_like(xi, fill_value=value_for_dtype(xi.dtype, float_val, int_val))

    return jax.tree.map(empty_el, x)

def invalidate(x, mask, invalid_float=jnp.nan, invalid_int=0):
    def inv(x):
        mask_rs = jnp.reshape(mask, mask.shape + (1,)*(x.ndim-1))
        return jnp.where(mask_rs, value_for_dtype(x.dtype, invalid_float, invalid_int), x)
    
    return jax.tree.map(inv, x)

# ------------------------------------------------------------------------------------------------ #
#                              Packing Helpers for more efficient comm                             #
# ------------------------------------------------------------------------------------------------ #

@dataclass
class PackingSpec():
    treedef: Any
    shapes: Tuple[int, ...]
    dtypes: Tuple[jnp.dtype, ...]
    offsets: np.array

def _pack_pytree(x: Pytree, ndim_keep: int = 1) -> Tuple[jax.Array, PackingSpec]:
    """Packs a pytree into a single byte-array and meta-data needed for reconstructing it

    Leaves must align in the first ndim_keep dimensions
    """
    leaves, treedef = jax.tree_util.tree_flatten(x)

    base_shape = leaves[0].shape[:ndim_keep]
    for l in leaves:
        assert l.shape[:ndim_keep] == base_shape, "leaves must align in leading dimensions"

    dtypes = [l.dtype for l in leaves]
    shapes = [l.shape for l in leaves]
    
    arrs = [l.reshape((base_shape + (-1,))).view(jnp.uint8) for l in leaves]

    num = [a.shape[-1] for a in arrs]
    offsets = np.pad(np.cumsum(num), (1,0))

    spec = PackingSpec(treedef, shapes, dtypes, offsets)

    return jnp.concatenate(arrs, axis=-1), spec

def _unpack_pytree(x: jax.Array, p: PackingSpec) -> Pytree:
    if p is None:
        return x

    leaves = []
    for i in range(0, len(p.shapes)):
        new = x[...,p.offsets[i]:p.offsets[i+1]].view(p.dtypes[i]).reshape(p.shapes[i])
        leaves.append(new)
    return jax.tree_util.tree_unflatten(p.treedef, leaves)

# ------------------------------------------------------------------------------------------------ #
#                                  Simple Communication Directives                                 #
# ------------------------------------------------------------------------------------------------ #

def global_splits(n, axis_name="gpus"):
    alln = jax.lax.all_gather(n, axis_name)
    return jnp.pad(jnp.cumsum(alln), (1,0), constant_values=0)

def send_to_right(x, axis_name, invalid_float=jnp.nan, invalid_int=0):
    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    xin = jax.lax.ppermute(x, axis_name, [(i, i+1) for i in range(0,ndev-1)])
    xin = invalidate(xin, rank == 0, invalid_float, invalid_int)

    return xin

def send_to_left(x, axis_name, invalid_float=jnp.nan, invalid_int=0):
    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    xin = jax.lax.ppermute(x, axis_name, [(i, i-1) for i in range(1,ndev)])
    xin = invalidate(xin, rank == ndev-1, invalid_float, invalid_int)

    return xin

def get_pos(x):
    if isinstance(x, jax.Array):
        return x
    else: # assume x is a pytree with .pos attribute
        return x.pos

def shift_particles_left(x, nsend, max_send, npart):
    rank, ndev, axis_name = get_rank_info()
    
    # Validate that send buffer is large enough
    npart = npart + raise_if(nsend >= max_send,
        "Cannot fit {nsend} particles into buffer of size {max_send}!",
        nsend=nsend, max_send=max_send
    )

    # Validate that particle array has enough free space
    nget = send_to_left(nsend, axis_name)
    npart = npart + raise_if(npart + nget - nsend >= pytree_len(x),
        "Cannot shift particles: have={nhave}, get={nget}, send={nsend}, max={nmax}.",
        nhave=npart, nget=nget, nsend=nsend, nmax=pytree_len(x)
    )

    # Send the particles
    x_get = send_to_left(jax.tree.map(lambda v: v[0:max_send], x), axis_name, invalid_float=jnp.nan)

    # Delete the particles that were send
    iar = jnp.arange(pytree_len(x))
    x = invalidate(x, iar < nsend)
    x = jax.tree.map(lambda v: jnp.roll(v, -nsend, axis=0), x)

    # Insert the received particles
    idx = jnp.arange(max_send)
    idx = jnp.where(idx < nget, npart - nsend + idx, pytree_len(x)) # discard indices beyond nadd
    x = jax.tree.map(lambda u,v: u.at[idx].set(v), x, x_get)

    return x, npart + nget - nsend

# ------------------------------------------------------------------------------------------------ #
#                                     All To All communication                                     #
# ------------------------------------------------------------------------------------------------ #

def all_to_all_with_splits(x, ispl, output=None, axis_name="gpus", verify=True, err_hint="", copy_self=True, pack_pytree=True):
    """all_to_all communication with data-dependent communication volume
    
    We send to rank i: x[ispl[i]:ispl[i+1]] 
    all received values will be inserted continguously into output (starting at 0).
    
    x: jax.Array or pytree. If it is a pytree the communication will be applied over the leading
       dimensions of all leaves (undefined behaviour if some leaves have different lengths)
    output: jax.Array or pytree. If x is a pytree output needs to be of identical structure.
            If not provided, we use a copy of x filled with jnp.nan (or 0 for integers)
    verify: If True, throws an error if output buffer is too small. Otherwise out-of-range values
            will simply be discarded.
    err_hint: If given, add a hint to the potential error message, indicating how to fix it
    copy_self: Extract self-send data and copy it directly (surprisingly this is faster)
    """
    if output is None:
        output = empty_like(x)

    out_size = pytree_len(output)

    input_offsets = ispl[:-1]
    send_sizes = ispl[1:] - ispl[:-1]
    recv_sizes = jax.lax.all_to_all(send_sizes, axis_name, 0, 0, tiled=True)
    dev_spl = cumsum_starting_with_zero(recv_sizes)
    output_offsets = dev_spl[:-1]

    if verify:
        need = jnp.sum(recv_sizes)
        recv_sizes = recv_sizes + raise_if(need > out_size,
            "The receiving buffer is too small, need={need}, have={have}" + err_hint,
            need=need, have=out_size
        )

    if copy_self:
        # avoid communication for self i/o
        rank = jax.lax.axis_index("gpus")
        iout = jnp.arange(out_size) #+ output_offsets[rank]
        iin = jnp.arange(out_size) + input_offsets[rank] - output_offsets[rank]
        mask = (iout >= output_offsets[rank]) & (iout < output_offsets[rank] + send_sizes[rank])
        
        send_sizes = send_sizes.at[rank].set(0)
        recv_sizes = recv_sizes.at[rank].set(0)

        def copy(xi, outi):
            mask_rs = jnp.reshape(mask, (len(mask),) + (1,)*(xi.ndim -1))
            return jnp.where(mask_rs, xi[iin], outi)

        output = jax.tree.map(copy, x, output)

    # funnily jax.lax.ragged_all_to_all wants to know the output_offsets on the
    # sending GPU rather than the receiving one... So we need to communicate the offsets
    output_offsets = jax.lax.all_to_all(output_offsets, axis_name, 0, 0, tiled=True)

    def comm(xi, outi):
        return jax.lax.ragged_all_to_all(
            xi, outi, input_offsets, send_sizes, output_offsets, recv_sizes, axis_name=axis_name
        )

    if pack_pytree:
        xp, xspec = _pack_pytree(x)
        op, ospec = _pack_pytree(output)
        op = comm(xp, op)
        return _unpack_pytree(op, ospec), dev_spl
    else:
        return jax.tree.map(comm, x, output), dev_spl

def dynamic_all_gather(x, nsend, output=None, axis_name="gpus", verify=True):
    """An all-gather where each task may send different amounts.
    
    returns output, dev_spl -- where output[dev_spl[i]:dev_spl[i+1]] contains rank i's input
    """
    if output is None:
        output = empty_like(x)
    
    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    nrecv = jax.lax.all_gather(nsend, axis_name)
    dev_spl = cumsum_starting_with_zero(nrecv)

    if verify:
        out_size = pytree_len(output)
        dev_spl = dev_spl + raise_if(dev_spl[-1] >= out_size,
            "The receiveing buffer (size: {out_size}) is to small (need: {need})",
            out_size=out_size, need=dev_spl[-1],
        )

    input_off = jnp.zeros(ndev, jnp.int32)
    nsend = jnp.full(ndev, nsend, dtype=jnp.int32)
    output_off = jnp.full(ndev, dev_spl[rank])

    def comm(xi, outi):
        return jax.lax.ragged_all_to_all(
            xi, outi, input_off, nsend, output_off, nrecv, axis_name=axis_name
        )

    return jax.tree.map(comm, x, output), dev_spl

def arange_for_comm(irank: jax.Array, x: jax.Array, 
                    num: jax.Array | int |None = None, axis_name="gpus"):
    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    if num is not None:
        irank = jnp.where(jnp.arange(len(irank), dtype=irank.dtype) < num, irank, ndev)
    isort = jnp.argsort(irank)
    dev_spl = jnp.searchsorted(irank[isort], jnp.arange(ndev+1, dtype=irank.dtype), side="left")

    xsort = jax.tree.map(lambda d: d[isort], x)

    return xsort, dev_spl, isort

def all_to_all_with_irank(
        irank: jax.Array,
        x: jax.Array | Pytree,
        output: jax.Array | Pytree | None = None,
        num: jax.Array | int | None = None,
        axis_name: str = "gpus",
        verify: bool = True,
        err_hint: str = "",
        copy_self: bool = True,
        pack_pytree: bool = True,
        get_inverse: bool = False
    ):
    """Communicate by indicating the rank of the receiving device

    To understand most arguments, see documentation of all_to_all_with_splits
    """
    xsort, dev_spl, isort = arange_for_comm(irank, x, num=num, axis_name=axis_name)
    x, dev_spl = all_to_all_with_splits(xsort, dev_spl, output, axis_name, verify=verify, err_hint=err_hint, copy_self=copy_self, pack_pytree=pack_pytree)
    if get_inverse:
        invsort = jnp.zeros_like(isort).at[isort].set(jnp.arange(len(isort), dtype=isort.dtype))
        return x, dev_spl, invsort
    else:
        return x, dev_spl

def all_to_all_request(
    irank: jax.Array,
    indices: jax.Array,
    x: jax.Array | Pytree,
    output: jax.Array | Pytree | None = None,
    num: jax.Array | int | None = None,
    axis_name: str = "gpus",
    verify: bool = True,
    err_hint: str = "",
    copy_self: bool = True,
    pack_pytree: bool = True
) -> jax.Array:
    # First inform the task with the data which indices we need
    indices_sort, dev_spl, isort = arange_for_comm(irank, indices, num=num, axis_name=axis_name)
    indices, dev_spl = all_to_all_with_splits(
        indices_sort, dev_spl, output=None, axis_name=axis_name, verify=verify, err_hint=err_hint,
        copy_self=copy_self, pack_pytree=pack_pytree
    )
    # Then send back the data at those locations
    xsort, dev_spl = all_to_all_with_splits(
        x[indices], dev_spl, output, axis_name=axis_name, verify=verify, err_hint=err_hint,
        copy_self=copy_self, pack_pytree=pack_pytree
    )
    # rearange to the original order
    invsort = jnp.zeros_like(isort).at[isort].set(jnp.arange(len(isort), dtype=isort.dtype))
    return jax.tree.map(lambda xi: xi[invsort], xsort)

def all_to_all_request_children(
    dev_spl: jax.Array,
    indices: jax.Array,
    spl: jax.Array,
    data: jax.Array | Pytree,
    output: jax.Array | Pytree | None = None,
    axis_name: str = "gpus",
    verify: bool = True,
    err_hint: str = "",
    copy_self: bool = True,
    pack_pytree: bool = True
):
    if output is None:
        output = empty_like(data)

    size = pytree_len(output)

    # First inform the task with the data which indices we need
    indices, dev_spl = all_to_all_with_splits(
        indices, dev_spl, output=None, axis_name=axis_name, verify=verify, err_hint=err_hint,
        copy_self=copy_self, pack_pytree=pack_pytree
    )
        
    # fill a continous buffer with the requested data
    node_sizes = (spl[1:] - spl[:-1])[indices]
    out_node_spl = cumsum_starting_with_zero(node_sizes)
    child_inode = inverse_of_splits(out_node_spl, size)
    child_inode_offset = jnp.arange(size) - out_node_spl[child_inode]
    child_id = spl[indices[child_inode]] + child_inode_offset
    child_data = jax.tree.map(lambda xi: xi[child_id],  data)
    child_dev_spl = out_node_spl[dev_spl]

    # send back the node_sizes so the receiver knows where each node starts
    node_sizes, node_dev_spl = all_to_all_with_splits(
        node_sizes, dev_spl, axis_name=axis_name, verify=verify, err_hint=err_hint, 
        pack_pytree=pack_pytree,
    )
    node_spl = cumsum_starting_with_zero(node_sizes)
    
    # Now send the child data
    xchild, child_dev_spl = all_to_all_with_splits(
        child_data, child_dev_spl, output, axis_name=axis_name, verify=verify, err_hint=err_hint,
        copy_self=copy_self, pack_pytree=pack_pytree
    )
    
    return xchild, node_spl, child_dev_spl

# ------------------------------------------------------------------------------------------------ #
#                                      All to all with permute                                     #
# ------------------------------------------------------------------------------------------------ #

def update_range(x, update, i1, i2):
    idx = i1 + jnp.arange(len(update))
    return x.at[jnp.where(idx < i2, idx, len(x))].set(update)

def gather_num(x, size, at):
    return x[at + jnp.arange(size)]

def permute_offset(offset, num):
    return [(i, (i + offset) % num) for i in range(num)]

def all_to_all_with_permute(x, ispl, buffer_bytes=8*1024**2, axis_name=None, verify=True):
    rank, ndev, axis_name = get_rank_info(axis_name)

    x, spec = _pack_pytree(x)
    bsize = max(buffer_bytes // (x[0].size*x.itemsize), 1)

    output = jnp.copy(x)

    nsend = ispl[1:] - ispl[:-1]
    nrecv = jax.lax.all_to_all(nsend, axis_name, split_axis=0, concat_axis=0, tiled=True)
    ispl_recv = jnp.pad(jnp.cumsum(nrecv), (1,0))

    if verify:
        out_size = pytree_len(output)
        ispl_recv = ispl_recv + raise_if(ispl_recv[-1] >= out_size,
            "The receiving buffer (size: {out_size}) is to small (need: {need})",
            out_size=out_size, need=ispl_recv[-1],
        )

    nsteps = (jnp.roll(nsend, -rank) + bsize - 1) // bsize
    nsteps = jax.lax.pmax(nsteps, axis_name)

    # jax.debug.print("s1 {}, s2 {}, nsteps {}", nsend, bsize, nsteps)

    def handle_offset(output, offset, nsteps):
        ito, ifrom = (rank + offset) % ndev, (rank - offset) % ndev
        def step(i, output):
            data_send = gather_num(x, bsize, at=ispl[ito]+bsize*i)
            data_recv = jax.lax.ppermute(data_send, axis_name, permute_offset(offset, ndev))
            return update_range(output, data_recv, ispl_recv[ifrom] + bsize*i, ispl_recv[ifrom+1])
        return jax.lax.fori_loop(0, nsteps, step, output)
    
    output = update_range(output, gather_num(x, len(x), at=ispl[rank]), ispl_recv[rank], ispl_recv[rank+1])

    for offset in range(1, ndev):
        output = handle_offset(output, offset, nsteps[offset])
    
    return _unpack_pytree(output, spec), ispl_recv
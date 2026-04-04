from dataclasses import dataclass, field, replace
from functools import partial
from typing import List
import jax
import jax.numpy as jnp

from .jax_ext import pcast_like, pcast_vma, tree_map_by_len
from .tools import set_range, inverse_of_splits, cumsum_starting_with_zero

def static_field(*args, **kwargs):
    return field(*args, metadata=dict(static=True), **kwargs)

def same_width_int(dtype: jnp.dtype) -> jnp.dtype:
    if dtype.itemsize == 4:
        return jnp.int32
    elif dtype.itemsize == 8:
        return jnp.int64
    else:
        raise ValueError(f"unsupported type {dtype}")

# ------------------------------------------------------------------------------------------------ #
#                                    User Interface data classes                                   #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass(kw_only=True, slots=True)
class Pos: # this class is mostly defined to declare an interface that particle class should follow
    """A dataclass holding positions.
    
    :paramref:`num` and :paramref:`num_total` are only required for multi-GPU setups where
    the number of filled entries may be less than the size and run-time dependent
    
    Args:
        pos: Position array of shape (size, dim)
        num: Filled count (<=size) on local device, needed for multi-GPU, may be dynamic
        num_total: Total count on all devices, static
    """
    pos: jax.Array

    num: int | jax.Array | None = None
    num_total: int | None = static_field(default=None)

@jax.tree_util.register_dataclass
@dataclass(kw_only=True, slots=True)
class PosMass:
    """Dataclass holding positions and masses, see :class:`Pos` for details"""
    pos: jax.Array
    mass: jax.Array

    num: jax.Array | None = None
    num_total: int | None = static_field(default=None)

@jax.jax.tree_util.register_dataclass
@dataclass(kw_only=True, slots=True)
class ParticleData:
    """Dataclass holding positions and optional data, see :class:`Pos` for details"""
    pos: jax.Array # (Nparticles, 3)
    mass: jax.Array | None = None # (Nparticles,) or scalar
    vel: jax.Array | None = None # (Nparticles, 3)
    id: jax.Array | None = None

    num: jax.Array | None = None
    num_total: int | None = static_field(default=None)

# ------------------------------------------------------------------------------------------------ #
#                     Helpful methods for accessing and manipulating particles                     #
# ------------------------------------------------------------------------------------------------ #

def get_pos(part: Pos):
    if isinstance(part, jax.Array):
        assert (part.shape[-1] <= 10) and (part.ndim == 2)
        return part
    elif hasattr(part, "pos"):
        assert part.pos.ndim == 2
        return part.pos
    else:
        raise ValueError("Invalid input particles")
    
def get_pos_mass(part: PosMass):
    mass = jnp.broadcast_to(part.mass, part.pos.shape[:-1])
    return jnp.concatenate([part.pos, mass[..., None]], axis=-1)

def get_num(part: Pos, default_to_length=False, default_to_nancount=True):
    if getattr(part, "num", None) is None:
        if default_to_length:
            return len(get_pos(part))
        if default_to_nancount:
            return jnp.sum(~jnp.isnan(get_pos(part)[:,0]))
        raise ValueError("Need .num attribute in particle structure for distributed mode")
    return jnp.sum(part.num) # Summing here, to be correct if sharding changed

def get_num_total(part: Pos, default_to_length=False):
    num = getattr(part, "num_total", None)
    if num is not None:
        return num
    elif default_to_length:
        return len(get_pos(part))
    else:
        raise ValueError("Need num_total attribute on particle data structure (for distributed mode)")

def flatten_particles(part: Pos):
    """Flattens particles from shape (Ndev,N) -> (Ndev*N)."""
    assert part.pos.ndim == 3, "Can only flatten from (Ndev,N) to (Ndev*N)"

    part_flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), part)
    part_flat.num = jnp.sum(part_flat.num)
    if jnp.shape(part.mass) != jnp.shape(part.pos)[:-1]: # Have scalar mass, but shardmap reshaped it
        part_flat.mass = jnp.reshape(part_flat.mass, (-1,))[0]

    return part_flat

def expand_particles(part: Pos, ndev: int):
    """Expands particles from shape (Ndev*N) -> (Ndev,N)"""
    assert part.pos.ndim == 2, "Can only expand from (Ndev*N) to (Ndev,N)"
    assert len(part.pos) % ndev == 0
    assert part.num_total == len(part.pos), "So far this function cannot handle padded particles"

    Ntot = len(part.pos)
    N = len(part.pos) // ndev

    def expand(x):
        if jnp.ndim(x) > 0 and len(x) == Ntot:
            return jnp.reshape(x, (ndev, N) + jnp.shape(x)[1:])
        else:
            return jnp.broadcast_to(x, (ndev,) + jnp.shape(x)) # replicate dynamic meta-data per device

    part_exp = jax.tree.map(expand, part)
    part_exp.num = jnp.full(ndev, N)

    return part_exp

def pad_particles(part: Pos, num: int,  float_val:float = jnp.nan, int_val: int = 0):
    if num == 0: return part
    
    assert part.pos.ndim == 2, "Positions should have shape (N,3)"

    def pad(xi):
        if xi.dtype.kind == "f":
            val = float_val
        else:
            val = int_val
        
        return jnp.pad(xi, [(0, num)] + [(0,0)]*(xi.ndim - 1), constant_values=val)

    return tree_map_by_len(pad, part, len(part.pos))

def squeeze_any(data, size, num, ntot):
    with jax.enable_x64():
        spl = cumsum_starting_with_zero(num)
        idx = jnp.arange(ntot, dtype=jnp.int64)
        irank = jnp.cumsum(jnp.zeros_like(idx).at[spl].set(1)) - 1
        ipart = idx - spl[irank]

        data_sq = tree_map_by_len(lambda x: x[irank, ipart], data, size, axis=1)
        
    return data_sq

def squeeze_particles(part: Pos):
    """Squeezes multi-gpu output particles (Ndev,N) into a dense form (Ntot)"""
    if part.pos.ndim == 2:
        # We are already flat or inside of a shard_map
        # in this case the only use case is to remove a possible padding
        return tree_map_by_len(lambda x: x[:part.num], part, len(part.pos))

    assert part.pos.ndim == 3, "Require input particles of form (Ndev,N,...)"
    assert len(part.num) == len(part.pos), "Should have particle number from each device"

    ntot = get_num_total(part)

    size_part = part.pos.shape[1]
    
    with jax.enable_x64():
        spl = cumsum_starting_with_zero(part.num)
        idx = jnp.arange(ntot, dtype=jnp.int64)
        irank = jnp.cumsum(jnp.zeros_like(idx).at[spl].set(1)) - 1
        ipart = idx - spl[irank]

        part_sq = tree_map_by_len(lambda x: x[irank, ipart], part, size_part, axis=1)
        part_sq.num = ntot

        if getattr(part, "mass", None) is not None:
            ndev = len(spl) - 1
            if (part.mass.ndim == 1) and (len(part.mass) == ndev): # scalar mass per device
                part_sq.mass = part.mass[0]
        
    return part_sq

def all_particles_equal(p1: ParticleData, p2: ParticleData):
    flag_tree = jax.tree.map(lambda a, b: jnp.all(a==b), p1, p2)
    all_leaves_equal = jax.tree.reduce(lambda a, b: a and b, flag_tree)

    return all_leaves_equal and (p1.num_total == p2.num_total)

# ------------------------------------------------------------------------------------------------ #
#                                  Internal Particle Data classes                                  #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass(slots=True, kw_only=True)
class PosId():
    pos: jax.Array
    id: jax.Array

@jax.tree_util.register_dataclass
@dataclass(slots=True, kw_only=True)
class PosLvl():
    pos: jax.Array
    lvl: jax.Array

    def pos_lvl(self):
        if self.pos.dtype.itemsize == 4:
            lvl_t = jnp.astype(self.lvl, jnp.int32).view(self.pos.dtype)
        elif self.pos.dtype.itemsize == 8:
            lvl_t = jnp.astype(self.lvl, jnp.int64).view(self.pos.dtype)
        else:
            raise ValueError("only 32 and 64 bit types handled here")
        
        return jnp.concatenate((self.pos, lvl_t[...,None]), axis=-1)

@jax.tree_util.register_dataclass
@dataclass(slots=True, kw_only=True)
class PosLvlNum():
    pos: jax.Array
    lvl: jax.Array
    npart: jax.Array

    def pos_lvl(self):
        if self.pos.dtype.itemsize == 4:
            lvl_t = jnp.astype(self.lvl, jnp.int32).view(self.pos.dtype)
        elif self.pos.dtype.itemsize == 8:
            lvl_t = jnp.astype(self.lvl, jnp.int64).view(self.pos.dtype)
        else:
            raise ValueError("only 32 and 64 bit types handled here")
        
        return jnp.concatenate((self.pos, lvl_t[...,None]), axis=-1)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PosLvlId(PosLvl):
    id: jax.Array

# ------------------------------------------------------------------------------------------------ #
#                                       Generic Data Holders                                       #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass
class PackedArray:
    """A dataclass allowing to stack several dynamic arrays into a single buffer"""
    data: jax.Array
    ispl: jax.Array
    fill_values: jax.Array | None = None
    # keep levels_filled as an array with shape (1,) for compatibility with shard map:
    levels_filled: jax.Array = field(default_factory=lambda: jnp.zeros((1,), dtype=jnp.int32))

    # Alternate constructors
    @classmethod
    def create_empty(cls, size, levels: int, dtype=jnp.float32, *, fill_values=None, vma=None):
        data = jnp.zeros(size, dtype=dtype)
        ispl = jnp.zeros(levels + 1, dtype=jnp.int32)
        levels_filled = jnp.zeros((1,), dtype=jnp.int32)

        if vma is not None:
            data = pcast_vma(data, vma)
            ispl = pcast_vma(ispl, vma)

        if jnp.isscalar(fill_values):
            fill_values = jnp.full(levels, fill_values, dtype=dtype)

        return cls(data, ispl, fill_values, levels_filled)
    
    @classmethod
    def from_data(cls, data, ispl, fill_values=None, levels_filled=None):
        if levels_filled is None:
            levels_filled = jnp.full((1,), len(ispl)-1, dtype=jnp.int32)
        else:
            levels_filled = jnp.asarray(levels_filled).reshape(1,)

        if jnp.isscalar(fill_values):
            fill_values = jnp.full(len(ispl)-1, fill_values, dtype=data.dtype)

        return cls(data, ispl, fill_values, levels_filled)

    def get(self, level, size=None, fill_value=None):
        if size is None:
            size = self.size()
        indices = jnp.arange(size, dtype=jnp.int32) + self.ispl[level]
        valid = indices < self.ispl[level + 1]
        valid = valid.reshape((-1,) + (1,) * (self.data.ndim - 1))
        if fill_value is None:
            fill_value = jnp.astype(self.fill_values[level], self.data.dtype)
        return jnp.where(valid, self.data[indices], fill_value)
    
    def set(self, level, values, num=None, fill_value=None):
        if num is None:
            num = values.shape[0]
        new_spl = jnp.where(jnp.arange(len(self.ispl)) <= level, self.ispl, self.ispl[level] + num)
        new_data = set_range(self.data, values, self.ispl[level], self.ispl[level] + num)
        if fill_value is not None:
            new_fill_vals = self.fill_values.at[level].set(jnp.astype(fill_value, self.data.dtype))
        else:
            new_fill_vals = self.fill_values
        return PackedArray(new_data, new_spl, new_fill_vals, jnp.reshape(level+1, (1,)))
    
    def append(self, values, num=None, fill_value=None, resize=False):
        if resize:
            arr =  self.resize_levels(self.nlevels()+1)
        else:
            arr = self
        
        return arr.set(self.levels_filled[0], values, num, fill_value)
    
    def resize_levels(self, levels):
        ispl_new = jnp.full(levels+1, fill_value=self.ispl[-1], dtype=self.ispl.dtype)
        ispl_new = ispl_new.at[:len(self.ispl)].set(self.ispl)
        return PackedArray(
            self.data, ispl_new, jnp.resize(self.fill_values, levels), self.levels_filled
        )
    
    def size(self):
        return len(self.data)
    
    def num(self, level):
        return self.ispl[level + 1] - self.ispl[level]
    
    def nfilled(self):
        return self.ispl[-1]
    
    def nlevels(self):
        return len(self.ispl) - 1
    
    def all_equal(self, other: "PackedArray"):
        def eq_nan(x, y):
            return (x == y) | (jnp.isnan(x) & jnp.isnan(y))

        equal  = jnp.all(eq_nan(self.data, other.data))
        equal &= jnp.all(self.ispl == other.ispl)
        equal &= jnp.all(eq_nan(self.fill_values, other.fill_values))
        equal &= jnp.all(self.levels_filled == other.levels_filled)
        return equal

# ------------------------------------------------------------------------------------------------ #
#                                          Tree Structure                                          #
# ------------------------------------------------------------------------------------------------ #


def _min_max_msb_diff(dtype):
    if dtype == jnp.float32:
        return -150, 128
    elif dtype == jnp.float64:
        return -1075, 1024
    elif dtype == jnp.int32:
        return -1, 32
    elif dtype == jnp.int64:
        return -1, 64

def _min_tree_level(dim, dtype):
    min_per_dim = _min_max_msb_diff(dtype)[0]
    return dim*min_per_dim

def _max_tree_level(dim, dtype):
    max_per_dim = _min_max_msb_diff(dtype)[1]
    return dim*(max_per_dim + 1)

@dataclass(frozen=True)
class LevelInfo():
    """Data class holding info on minimum and maximum level of a tree"""
    dim: int
    dtype: jnp.dtype

    def min_lvl(self) -> int:
        return _min_tree_level(self.dim, self.dtype)
    def max_lvl(self) -> int:
        return _max_tree_level(self.dim, self.dtype)

@jax.tree_util.register_dataclass
@dataclass
class TreeHierarchy():
    """Dataclass holding the tree-plane hierarchy

    Most properties are stored in :class:`PackedArray` classes that stack them
    contingously in a single buffer for all tree-planes
    
    Args:
        size_leaves: Size of the leaf-node level allocation
        ispl_n2n: The node-to-node splitting points. Node i on plane p includes all plane
            p-1 nodes in the range ispl_n2n.get(p)[i] ... ispl_n2n.get(p)[i+1]
        ispl_n2l: Splitting points that go directly from nodes to leaves.
        ispl_l2p_per_type: Leaf-to-particle splitting points per particle type.
        lvl: Morton-level of nodes. Can be used to define extent of nodes.
        geom_cent: Geometric centers
        mass: Mass of nodes (only available for mass_centered tree)
        mass_cent: Mass centers of nodes (only available for mass_centered tree)
    """
    size_leaves: int = static_field()

    # Packed Arrays:
    ispl_n2n: PackedArray
    ispl_n2l: PackedArray

    ispl_l2p_per_type: List[jax.Array]

    # tree plane data:
    lvl: PackedArray
    geom_cent: PackedArray
    mass: PackedArray | None = None
    mass_cent: PackedArray | None = None

    def splits_leaf_to_part(self, ptype: int = 0, size: int | None = None) -> jax.Array:
        """Convenience method to get the leaf-to-particle splits for a specific type"""
        if size is None:
            size = self.size()+1
        return self.ispl_l2p_per_type[ptype][:size]
    # self.ispl_n2n.get(0, size)
    
    def npart(self, level: int, ptype: int = 0, size=None) -> jax.Array:
        """Number of particles in an node"""
        if size is None:
            size = self.size()
        ispl_n2p = self.splits_leaf_to_part(ptype)[self.ispl_n2l.get(level, size+1)]
        return ispl_n2p[1:] - ispl_n2p[:-1]

    def center(self) -> PackedArray:
        """Returns mass_cent or geom_cent, depending on how tree was constructed"""
        if self.mass_cent is not None:
            return self.mass_cent
        else:
            return self.geom_cent
           
    def num_planes(self) -> int:
        """Numer of planes in the tree"""
        return len(self.ispl_n2l.ispl) - 1

    def num(self, level) -> int:
        """Number of nodes in a given plane"""
        return self.lvl.num(level)
    
    def size(self) -> int:
        """The recommended allocation size at the leaf level"""
        # Could choose something smaller here later...
        return self.ispl_n2n.size() - 1
    
    def info(self) -> LevelInfo:
        """Info about the minimum and maximum Morton levels of the tree"""
        dim = self.geom_cent.data.shape[-1]
        dtype = self.geom_cent.data.dtype
        return LevelInfo(dim, dtype)

# ------------------------------------------------------------------------------------------------ #
#                                         Interaction Data                                         #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class InteractionList:
    """A dataclass that holds interaction information for dual tree-walks

    The interaction list is defined so that a receiving node i needs to interact with
    all source nodes in the range isrc[ispl[i]:ispl[i+1]]
    
    Args:
        ispl: splitting points of interaction list segments
        isrc: interaction source indices
        rad2: interaction radii squared. Optional -- so far only used in kNN algorithm
        ids: used in multi-GPU scenarios to define for each source index the origin index
        dev_spl: used in multi-GPU scenarios to define for each unique source index the
            origin rank. (Sources in range dev_spl[r]:dev_spl[r+1] belong to rank r.)
    """
    ispl: jax.Array
    isrc: jax.Array

    # Optional: interaction radii (can be useful for pruning)
    rad2: jax.Array | None = None

    # Multi-GPU specific: origin ids and device offets
    ids: jax.Array | None = None
    dev_spl: jax.Array | None = None

    def without_remote_query_points(self, rank: int) -> 'InteractionList':
        """
        By default the interaction list carries remote and local points both on ispl so that
        query and source indices are consistent.
        However, this function redefines the interaction list so that ispl is only defined for 
        local query points, but interaction indices may still point to remote points
        """
        idx = jnp.arange(len(self.ispl))
        mask = (idx >= self.dev_spl[rank]) & (idx <= self.dev_spl[rank+1])
        ispl_new = jnp.compress(
            mask, self.ispl, size=len(self.ispl), fill_value=self.ispl[self.dev_spl[rank+1]]
        )
        return replace(self, ispl=ispl_new)

    def nfilled(self):
        """Number of filled entries"""
        return self.ispl[-1]

    def size(self):
        """Size of the interaction indices array"""
        return self.isrc.size
    
    def dtype(self):
        """Index datatype (so far always int32)"""
        return self.isrc.dtype

def verify_ilist(ilist: InteractionList):
    if len(ilist.isrc) >= 2**31:
        raise ValueError("Intercation list allocation too large and may produce integer overflows",
                         "Hint: Use more GPUs or decrease alloc_fac_ilist (if possible)")
    return ilist

# ------------------------------------------------------------------------------------------------ #
#                                      Fof Specific Data Class                                     #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Label:
    # global labels are pointing to a particle that may lie on another task
    irank: jax.Array
    igroup: jax.Array

    # Optional, points towards a local particle that has the same global label
    # Useful for saving communication time
    ilocal_segment: jax.Array | None = None

    def stacked(self) -> jax.Array:
        return jnp.stack([self.irank, self.igroup], axis=-1)
    
    def __getitem__(self, key) -> "Label":
        return jax.tree.map(lambda x: x[key], self)
    
    def __eq__(self, other: "Label") -> jax.Array:
        return (self.irank == other.irank) & (self.igroup == other.igroup)
    
    def __ne__(self, other: "Label") -> jax.Array:
        return ~self.__eq__(other)

    def __ge__(self, other: "Label") -> jax.Array:
        rank_gtr = self.irank > other.irank
        return rank_gtr | ((self.irank == other.irank) & (self.igroup >= other.igroup))

    def __gt__(self, other: "Label") -> jax.Array:
        rank_gtr = self.irank > other.irank
        return rank_gtr | ((self.irank == other.irank) & (self.igroup > other.igroup))

@jax.tree_util.register_dataclass
@dataclass
class Link:
    a: Label
    b: Label

    @classmethod
    def from_stacked(cls, ld, axis=-1):
        assert axis == -1
        return cls(Label(ld[...,0], ld[...,1]), Label(ld[...,2], ld[...,3]))

    def stacked(self, axis=0):
        return jnp.stack([self.a.irank, self.a.igroup, self.b.irank, self.b.igroup], axis=axis)

@jax.tree_util.register_dataclass
@dataclass
class FofNodeData():
    lvl: jax.Array
    igroup: jax.Array
    spl: jax.Array

@jax.jax.tree_util.register_dataclass
@dataclass(slots=True)
class FofCatalogue:
    """Friends-of-friends catalogue data
    
    Many properties are optional and will be set to None if calculation was lacking data.
    To allow **jax.jit** compatibility the arrays are generally allocated larger than
    what is actually needed, with :paramref:`ngroups` indicating the filled
    count. You may use :func:`squeeze_catalogue` (outside of **jax.jit**) to obtain 
    a squeezed version.

    Args:
        ngroups: Actual number of groups. (On multi-GPU the local count.)
        mass: Group masses
        count: Number of particles
        offset: Starting point of each group in the particle array
        com_pos: Center of mass position
        com_vel: Center of mass velocity
        com_inertia_radius: Inertia radius 
            :math:`\\sqrt{\\langle (\\mathbf{x} - \\mathbf{x}_0)^2 \\rangle}`.
        scale_factor: For light-cone data: scale factor of light-cone crossing.
        v_rad: For light-cone data: radial (line of sight) velocity
        offset_rank: Provided for squeezed catalogues to indicate origin rank.
    """
    ngroups: jax.Array
    mass: jax.Array | None = None
    count: jax.Array | None = None
    offset: jax.Array | None = None
    com_pos: jax.Array | None = None
    com_vel: jax.Array | None = None
    com_inertia_radius: jax.Array | None = None
    scale_factor: jax.Array | None = None
    v_rad: jax.Array | None = None
    offset_rank: jax.Array | None = None # Only provided for squeezed catalogues

    def flatten(self):
        flat_cata = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), self)
        flat_cata.ngroups = jnp.sum(flat_cata.ngroups)
        return flat_cata

def squeeze_catalogue(
        cata: FofCatalogue,
        size_out: int | None = None,
        offset_mode: str = "rank",
        nparts: jax.Array | None = None
    ) -> FofCatalogue:
    """Squeezes multi-GPU FoF catalogue that was returned from a shard_map (Ndev,size_group)
    into a dense form (Ntot) and replicates it on every device

    Args:
        cata: FoF catalogue
        size_out: Can be provided to allow **jit** compatibility
        offset_mode: Can be "rank" or "flat". Before squeezing offsets indicate locations in the
            particle array of the same device. Since squeezing looses the device info, we need
            to indicate either the rank or we need to convert offsets to global offsets
            that index a squeezed particle array (converts to int64).
            "global" needs "nparts" as an input, i.e. how many particles were on each device
        nparts: particles that are on each device, needed for "global" offset_mode
    """
    assert offset_mode in ("rank", "global")

    if size_out is None:
        size_out = jnp.sum(cata.ngroups)

    if cata.count.ndim == 1:
        # for ndim==1 the only point of calling this function is to remove
        # trailing invalid groups
        return tree_map_by_len(lambda x: x[:size_out], cata, cata.count.shape[0])

    size_groups = cata.count.shape[1]
    
    with jax.enable_x64():
        spl = cumsum_starting_with_zero(cata.ngroups)
        idx = jnp.arange(size_out, dtype=jnp.int64)
        irank = jnp.cumsum(jnp.zeros_like(idx).at[spl].add(1)) - 1
        igroup = idx - spl[irank]

        cata_sq = tree_map_by_len(lambda x: x[irank, igroup], cata, size_groups, axis=1)
        cata_sq.ngroups = jnp.sum(cata.ngroups)

        if offset_mode == "rank":
            cata_sq.offset_rank = irank
        elif offset_mode == "global":
            with jax.enable_x64():
                assert nparts is not None, "Require particle numbers for 'global' offset mode"
                assert jnp.size(nparts) == cata.count.shape[0], "please provide npart for each device"
                spl = jnp.astype(cumsum_starting_with_zero(nparts), jnp.int64)
                cata_sq.offset = jnp.astype(cata_sq.offset, jnp.int64) + spl[irank]
        
        return cata_sq
squeeze_catalogue.jit = jax.jit(squeeze_catalogue, static_argnames=("size_out", "offset_mode"))

def sort_catalogue(cata: FofCatalogue, by: str = "count", descending: bool = True):
    key = getattr(cata, by, None)
    assert key is not None, f"catalogue doesn't have {by}"
    assert cata.count.ndim == 1, "Please squeeze catalogue before sorting"
        
    isort = jnp.argsort(key, descending=descending)

    return tree_map_by_len(lambda x: x[isort], cata, len(isort))
sort_catalogue.jit = jax.jit(sort_catalogue, static_argnames=("descending", "by"))

def catalogues_equal(c1: FofCatalogue, c2: FofCatalogue):
    flag_tree = jax.tree.map(lambda a, b: jnp.all(a==b), c1, c2)
    all_leaves_equal = jax.tree.reduce(lambda a, b: a and b, flag_tree)

    return all_leaves_equal and (c1.ngroups == c2.ngroups)

@partial(jax.tree_util.register_dataclass)
@dataclass(slots=True)
class RankIdx:
    """Holds a rank and an index -- often used to indicate the origin of data."""
    rank: jax.Array
    idx: jax.Array
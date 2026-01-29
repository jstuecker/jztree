from dataclasses import dataclass, field
from functools import partial
from typing import List, Iterator
import jax
import jax.numpy as jnp

from .config import TreeConfig
from .comm import pcast_like, pcast_vma
from .tools import set_range, inverse_of_splits, cumsum_starting_with_zero

def static_field(*args, **kwargs):
    return field(*args, metadata=dict(static=True), **kwargs)

# ------------------------------------------------------------------------------------------------ #
#                                    User Interface data classes                                   #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class Pos: # this class is mostly defined to declare an interface that particle class should follow
    pos: jax.Array

    num: jax.Array | None = None
    num_total: int | None = static_field(default=None)

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class PosMass:
    pos: jax.Array  # (Nparticles, 3)
    mass: jax.Array  # (Nparticles,) or (1,)

    num: jax.Array | None = None
    num_total: int | None = static_field(default=None)

    def __post_init__(self):
        if self.num is not None:
            assert jnp.ndim(self.num) == 1, "Please reshape num to shape (1,) (to allow shardmap return)"

@jax.jax.tree_util.register_dataclass
@dataclass
class ParticleData:
    pos: jax.Array # (Nparticles, 3)
    mass: jax.Array | None = None # (Nparticles,) or (1,)
    vel: jax.Array | None = None # (Nparticles, 3)

    num: jax.Array | None = None
    num_total: int | None = static_field(default=None)

    def __post_init__(self):
        assert self.pos.shape[1] == 3, "provide positions as an array of shape (N, 3)"
        if self.vel != None:
            assert self.vel.shape[1] == 3, "provide velocities as an array of shape (N, 3)"
            assert self.vel.shape[0] == self.pos.shape[0], "provide as many velocities as positions"

# ------------------------------------------------------------------------------------------------ #
#                   Methods for accessing class data that allow some flexibility                   #
# ------------------------------------------------------------------------------------------------ #

def get_pos(part: Pos):
    if isinstance(part, jax.Array):
        assert (part.shape[-1] == 3) and (part.ndim == 2)
        return part
    elif hasattr(part, "pos"):
        assert (part.pos.shape[-1] == 3) and (part.pos.ndim == 2)
        return part.pos
    else:
        raise ValueError("Invalid input particles")
    
def get_pos_mass(part: PosMass):
    mass = jnp.broadcast_to(part.mass, part.pos.shape[:-1])
    return jnp.concatenate([part.pos, mass[..., None]], axis=-1)

def get_num(part: Pos, default_to_length=False):
    if getattr(part, "num", None) is None:
        if default_to_length:
            return len(part.pos)
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

# ------------------------------------------------------------------------------------------------ #
#                                  Internal Particle Data classes                                  #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class PosLvl():
    pos: jax.Array
    lvl: jax.Array

    def pos_lvl(self):
        return jnp.concatenate((self.pos, self.lvl.view(jnp.float32)[...,None]), axis=-1)
    
@jax.tree_util.register_dataclass
@dataclass
class PosLvlId(PosLvl):
    id: jax.Array


# ------------------------------------------------------------------------------------------------ #
#                                       Generic Data Holders                                       #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass
class PackedArray:
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
        indices = jnp.arange(size) + self.ispl[level]
        valid = indices < self.ispl[level + 1]
        valid = valid.reshape((-1,) + (1,) * (self.data.ndim - 1))
        if fill_value is None:
            fill_value = self.fill_values[level]
        return jnp.where(valid, self.data[indices], fill_value)
    
    def set(self, level, values, num=None, fill_value=None):
        if num is None:
            num = values.shape[0]
        new_spl = jnp.where(jnp.arange(len(self.ispl)) <= level, self.ispl, self.ispl[level] + num)
        new_data = set_range(self.data, values, self.ispl[level], self.ispl[level] + num)
        if fill_value is not None:
            new_fill_vals = self.fill_values.at[level].set(fill_value)
        else:
            new_fill_vals = self.fill_values
        return PackedArray(new_data, new_spl, new_fill_vals, jnp.reshape(level+1, (1,)))
    
    def append(self, values, num=None, fill_value=None):
        return self.set(self.levels_filled[0], values, num, fill_value)
    
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

@jax.tree_util.register_dataclass
@dataclass
class TreePlane():
    # Defined per node:
    ispl: jax.Array # relation to children

    npart: jax.Array
    lvl: jax.Array
    geom_cent: jax.Array

    # Scalars (data dependent)
    nnodes: jax.Array

    around_com: bool = static_field()

    # Optional data:
    mass_cent: PosMass | None = None      # Optionally needed

    def size(self) -> int: # needed
        return self.lvl.shape[0]
    def center(self) -> jax.Array:
        if self.around_com:
            assert self.mass_cent is not None, "Mass center not available"
            return self.mass_cent.pos
        else:
            return self.geom_cent
    def node_extent(self, diag2=False) -> jax.Array: # only jax
        # return jnp.ldexp(1., self.lvl)
        olvl, omod = self.lvl//3, self.lvl % 3

        dx = jnp.ldexp(1., olvl)
        dy = jnp.ldexp(1., olvl + (omod >= 2).astype(jnp.int32))
        dz = jnp.ldexp(1., olvl + (omod >= 1).astype(jnp.int32))

        if diag2:
            return dx*dx + dy*dy + dz*dz
        else:
            return jnp.stack((dx, dy, dz), axis=-1)

@jax.tree_util.register_dataclass
@dataclass
class TreeHierarchy():
    # Packed Arrays:
    ispl_n2n: PackedArray
    ispl_n2l: PackedArray

    # tree plane data:
    lvl: PackedArray
    geom_cent: PackedArray
    mass: PackedArray | None = None
    mass_cent: PackedArray | None = None

    plane_sizes: List[int] = static_field(default_factory=list)

    def npart(self, level: int, size=None) -> jax.Array:
        if size is None:
            size = self.plane_sizes[level]
        ispl_n2p = self.ispl_n2n.get(0)[self.ispl_n2l.get(level, size+1)]
        return ispl_n2p[1:] - ispl_n2p[:-1]

    def center(self) -> PackedArray:
        if self.mass_cent is not None:
            return self.mass_cent
        else:
            return self.geom_cent
        
    def get_tree_plane(self, level: int, size=None) -> TreePlane:
        if size is None:
            size = self.plane_sizes[level]

        ispl_n2p = self.ispl_n2n.get(0)[self.ispl_n2l.get(level, size)]
        if self.mass_cent is not None:
            mass_cent = PosMass(pos=self.mass_cent.get(level, size), mass=self.mass.get(level, size))
        else:
            mass_cent = None
        return TreePlane(
            ispl = self.ispl_n2n.get(level, size+1),
            npart = ispl_n2p[1:] - ispl_n2p[:-1],
            lvl = self.lvl.get(level, size),
            geom_cent = self.geom_cent.get(level, size),
            nnodes = self.lvl.num(level),
            around_com = self.mass_cent is not None,
            mass_cent = mass_cent
        )
    
    def num_planes(self) -> int:
        return len(self.ispl_n2l.ispl) - 1
    
    def planes(self) -> Iterator[TreePlane]:
        for level in range(self.num_planes()):
            yield self.get_tree_plane(level)

    def num(self, level) -> int:
        return self.lvl.num(level)

# ------------------------------------------------------------------------------------------------ #
#                                         Interaction Data                                         #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass
class InteractionList:
    """Node i0 will interact with all indices iother[ispl[i0]:ispl[i0+1]]"""
    ispl: jax.Array
    iother: jax.Array

    # Multi-GPU specific: origin ids and device offets
    ids: jax.Array | None = None
    dev_spl: jax.Array | None = None

    def get_interactions(self, get_valid=False):
        """Returns (i0, i1, valid) indicating two interaction nodes and validity"""
        iint = jnp.arange(self.size(), dtype=self.dtype())
        i0 = inverse_of_splits(self.ispl, self.size())
        i1 = self.iother#[iint]
        if get_valid:
            valid = iint < self.ispl[-1]
            return i0, i1, valid
        else:
            return i0, i1
    
    def filter(self, mask: jax.Array, size: int | None = None) -> 'InteractionList':
        """Returns a filtered interaction list according to the boolean mask"""
        if size is None:
            size = mask.size
        ioff = cumsum_starting_with_zero(mask)
        iupdate = jnp.where(mask, ioff, size)
        iother_new = jnp.zeros(size, dtype=self.iother.dtype).at[iupdate].set(self.iother)
        ispl_new = ioff[self.ispl]

        return InteractionList(ispl=ispl_new, iother=iother_new)
    
    def nfilled(self):
        return self.ispl[-1]

    def size(self):
        return self.iother.size
    
    def dtype(self):
        return self.iother.dtype

# ------------------------------------------------------------------------------------------------ #
#                                      Fof Specific Data Class                                     #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass
class Label:
    irank: jax.Array
    igroup: jax.Array

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
    label: Label | jax.Array # Local index pointer for single-gpu, but (rank,id) pointer for multi
    spl: jax.Array

@partial(jax.tree_util.register_dataclass, 
         meta_fields=["rlink", "boxsize"],
         data_fields=["posz", "igroup", "node_lvl", "ilist", "spl"])
@dataclass
class FofData:
    rlink : float
    boxsize : float

    posz: jax.Array       # z-sorted positions
    # idz: jax.Array        # ids so that posz = pos0[idz]
    igroup: jax.Array     # group labels
    node_lvl: jax.Array   # node-levels
    ilist: InteractionList  # interaction list (on leaf indices)
    spl: jax.Array        # leaf splits so that posz[spl[i]:spl[i+1]] are in leaf i

@jax.jax.tree_util.register_dataclass
@dataclass
class FofCatalogue:
    ngroups: jax.Array
    mass: jax.Array | None = None
    count: jax.Array | None = None
    offsets: jax.Array | None = None
    com_pos: jax.Array | None = None
    com_vel: jax.Array | None = None
    com_inertia_radius: jax.Array | None = None

# ------------------------------------------------------------------------------------------------ #
#                                     KNN Specific Data Classes                                    #
# ------------------------------------------------------------------------------------------------ #


@partial(jax.tree_util.register_dataclass, 
         meta_fields=["k", "boxsize"],
         data_fields=["posz", "idz", "spl", "ilist", "ir2list", "ilist_spl"])
@dataclass
class KNNData:
    k : int
    boxsize : float

    posz: jax.Array       # z-sorted positions
    idz: jax.Array        # ids so that posz = pos0[idz]
    spl: jax.Array        # leaf splits so that posz[spl[i]:spl[i+1]] are in leaf i
    ilist: jax.Array      # interaction list (leaf indices)
    ir2list: jax.Array    # interaction r2 list (lower bound leaf-leaf distances squared)
    ilist_spl: jax.Array  # leaf i interacts with leaves ilist[ilist_spl[i]:ilist_spl[i+1]]

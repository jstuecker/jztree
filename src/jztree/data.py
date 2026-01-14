import jax
import jax.numpy as jnp
from dataclasses import dataclass
from functools import partial
from fmdj.config import TreeConfig
from fmdj.data import InteractionList

@jax.tree_util.register_dataclass
@dataclass
class PosLvl():
    pos: jax.Array
    lvl: jax.Array

    def pos_lvl(self):
        return jnp.concatenate((self.pos, self.lvl.view(jnp.float32)[...,None]), axis=-1)
    
@jax.tree_util.register_dataclass
@dataclass
class PosLvlId(PosLvl):
    id: jax.Array

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

    def stacked(self):
        return jnp.stack([self.a.irank, self.a.igroup, self.b.irank, self.b.igroup])

@jax.tree_util.register_dataclass
@dataclass
class FofNodeData():
    lvl: jax.Array
    label: Label | jax.Array # Local index pointer for single-gpu, but (rank,id) pointer for multi
    spl: jax.Array
    num: jax.Array

@dataclass(frozen=True)
class KNNConfig:
    alloc_fac_ilist: float = 256.

    tree: TreeConfig = TreeConfig(
        max_leaf_size = 48,
        coarse_fac = 8.,       
        alloc_fac_nodes = 1.,
        stop_coarsen = 2048,
        mass_centered = False
    )

@dataclass(frozen=True)
class FofConfig:
    alloc_fac_ilist: float = 64.   

    tree: TreeConfig = TreeConfig(
        max_leaf_size = 48,
        coarse_fac = 8.,       
        alloc_fac_nodes = 1.,
        stop_coarsen = 2048,
        mass_centered = False
    )

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
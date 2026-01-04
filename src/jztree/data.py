import jax
import jax.numpy as jnp
from dataclasses import dataclass
from functools import partial
from fmdj.config import TreeConfig
from fmdj.data import InteractionList

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

    posz: jnp.ndarray       # z-sorted positions
    idz: jnp.ndarray        # ids so that posz = pos0[idz]
    spl: jnp.ndarray        # leaf splits so that posz[spl[i]:spl[i+1]] are in leaf i
    ilist: jnp.ndarray      # interaction list (leaf indices)
    ir2list: jnp.ndarray    # interaction r2 list (lower bound leaf-leaf distances squared)
    ilist_spl: jnp.ndarray  # leaf i interacts with leaves ilist[ilist_spl[i]:ilist_spl[i+1]]

@partial(jax.tree_util.register_dataclass, 
         meta_fields=["rlink", "boxsize"],
         data_fields=["posz", "igroup", "ilist", "spl"])
@dataclass
class FofData:
    rlink : float
    boxsize : float

    posz: jnp.ndarray       # z-sorted positions
    # idz: jnp.ndarray        # ids so that posz = pos0[idz]
    igroup: jnp.ndarray     # group labels
    ilist: InteractionList  # interaction list (on leaf indices)
    spl: jnp.ndarray        # leaf splits so that posz[spl[i]:spl[i+1]] are in leaf i
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from functools import partial

@dataclass(frozen=True)
class KNNConfig:
    max_leaf_size: int = 48
    rfac : float = 8.
    alloc_fac_ilist: float = 256.
    alloc_fac_nodes: float = 1.
    stop_coarsen: int = 2048

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
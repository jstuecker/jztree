from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, List, Callable, Any
import numpy as np
import threading

import jax
import jax.numpy as jnp
from jax.experimental.multihost_utils import process_allgather

from dataclasses import dataclass, replace, field
from .config import FofConfig

def max_allow_None(a, b):
    if a is None: return b
    if b is None: return a
    return max(a,b)

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class AllocStats:
    max_part_frac_sort: float | None = None
    max_part_frac_domain: float | None = None
    max_part_frac_interaction: float | None = None
    max_node_frac: float | None = None
    max_reg_frac: float | None = None
    max_reg_leaf_lvl: int | None = None
    max_ilist_frac_fof: float | None = None
    max_links_frac: float | None = None

    def record_filled_sort(self, npart, size):
        self.max_part_frac_sort = max_allow_None(self.max_part_frac_sort, float(npart/size))

    def record_filled_domain(self, npart, size):
        self.max_part_frac_interaction = max_allow_None(self.max_part_frac_interaction, float(npart/size))

    def record_filled_part_interactions(self, npart, size):
        self.max_part_frac_domain = max_allow_None(self.max_part_frac_domain, float(npart/size))

    def record_filled_nodes(self, num, size):
        self.max_node_frac = max_allow_None(self.max_node_frac, float(num/size))

    def record_regularization(self, lvl, nleaves_pre, nleaves_post):
        # print(f"reg: {lvl} {nleaves_pre}, {nleaves_post}")
        self.max_reg_frac = max_allow_None(self.max_reg_frac, float(nleaves_post/nleaves_pre))
        self.max_reg_leaf_lvl = max_allow_None(self.max_reg_leaf_lvl, lvl)

    def record_filled_interactions(self, nfilled, size):
        self.max_ilist_frac_fof = max_allow_None(self.max_ilist_frac_fof, float(nfilled/size))

    def record_filled_links(self, nfilled, size):
        self.max_links_frac = max_allow_None(self.max_links_frac, float(nfilled/size))

    def print_suggestions(self, cfg: FofConfig):
        print("--- Allocation Info ---")
        fill_frac = max_allow_None(self.max_part_frac_sort, self.max_part_frac_domain)
        fill_frac = max_allow_None(fill_frac, self.max_part_frac_interaction)
        if fill_frac is not None:
            print(f"Filled at most {fill_frac:.1%} of particles. (Affected by padding.)")
        if self.max_node_frac is not None:
            print(f"Filled at most {self.max_node_frac:.1%} of nodes. Could decrease alloc_fac_nodes "
                  f"at most from {cfg.tree.alloc_fac_nodes} "
                  f"to {cfg.tree.alloc_fac_nodes * self.max_node_frac:.2f}")
        if self.max_ilist_frac_fof is not None:
            print(f"Filled at most {self.max_ilist_frac_fof:.1%} of interaction list. Could decrease "
                  f"alloc_fac_ilist at most from {cfg.alloc_fac_ilist} to "
                  f"{cfg.alloc_fac_ilist * self.max_ilist_frac_fof:.2f}")
        if self.max_links_frac is not None:
            print(f"Filled at most {self.max_links_frac*100.:.2g}% of cross-task link data. Could "
                  f"decrease alloc_fac_distr_links at most from {cfg.alloc_fac_distr_links} to"
                  f"{cfg.alloc_fac_distr_links*self.max_links_frac:.2g}")
        if self.max_reg_frac is not None:
            print(f"Regularization increased number of leaves by {self.max_reg_frac- 1.:.2%}")
        print("-------")

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class InteractionStats:
    node2node: int | None = None
    leaf2leaf: int | None = None
    num_open: int | None = None

    largest_interaction_count: int | None = None

    def record_largest_interaction(self, num):
        self.largest_interaction_count = max_allow_None(self.largest_interaction_count, num)

    def print_info(self):
        if self.largest_interaction_count is not None:
            print(f"Largest interaction of any node: {self.largest_interaction_count}")

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Statistics:
    allocation: AllocStats = field(default_factory=AllocStats)
    interaction: InteractionStats = field(default_factory=InteractionStats)

    # Use to block data-races, e.g. in a one host - multiple devices setup:
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, metadata=dict(static=True)
    )

    def print_suggestions(self, cfg):
        self.allocation.print_suggestions(cfg)

        self.print_info()

    def print_info(self):
        self.interaction.print_info()

# ContextVar doesn't work well with callbacks and shardmap...
# That's why we simply set a single global variable
# _current_statistics = ContextVar("_current_ctx", default=None)
_current_statistics = None

def get_stats_context(key: str | None = None) -> Statistics | Any:
    stats = _current_statistics
    if stats is None:
        return None
    
    if key is not None:
        return getattr(stats, key, None)
    else:
        return stats

def stats_callback(key, f: Callable, *args, **kwargs):
    if get_stats_context() is None: return

    def f_with_lock(*args, **kwargs):
        stats = get_stats_context()
        sub_stats = get_stats_context(key)
        if stats is None or sub_stats is None:
            return
        with stats._lock:
            f(sub_stats, *args, **kwargs)
    jax.debug.callback(f_with_lock, *args, **kwargs)

def gather_stats_multihost(stats: Statistics):
    return process_allgather(stats)

def reduce_stats_multihost(stats: Statistics):
    return jax.tree.map(lambda x: np.max(x), stats)

@contextmanager
def statistics(ctx: Statistics | None = None) -> Iterator[Statistics]:
    global _current_statistics
    assert _current_statistics is None, "Can't nest statistics contextmanager"
    if ctx is None:
        ctx = Statistics()
    token = _current_statistics
    _current_statistics = ctx
    try:
        yield ctx
    finally:
        _current_statistics = None
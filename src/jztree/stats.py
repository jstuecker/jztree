from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, List, Callable, Any
import numpy as np
import threading

import jax
import jax.numpy as jnp
from jax.experimental.multihost_utils import process_allgather

from dataclasses import dataclass, replace, field

def max_allow_None(a, b):
    return b if a is None else max(a,b)

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class AllocStats:
    max_part_frac_sort: float | None = None
    max_part_frac_domain: float | None = None
    max_ilist_frac_fof: float | None = None

    def record_filled_sort(self, npart, size):
        self.max_part_frac_sort = max_allow_None(self.max_part_frac_sort, float(npart/size))

    def record_filled_domain(self, npart, size):
        self.max_part_frac_domain = max_allow_None(self.max_part_frac_domain, float(npart/size))

    def record_filled_interactions(self, nfilled, size):
        self.max_ilist_frac_fof = max_allow_None(self.max_ilist_frac_fof, float(nfilled/size))

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class InteractionStats:
    node2node: int | None = None
    leaf2leaf: int | None = None
    num_open: int | None = None

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Statistics:
    allocation: AllocStats = field(default_factory=AllocStats)
    interaction: InteractionStats = field(default_factory=InteractionStats)

    # Use to block data-races, e.g. in a one host - multiple devices setup:
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, metadata=dict(static=True)
    )

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
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, List, Callable, Any
import numpy as np
import threading

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace, field

@dataclass(slots=True)
class AllocStats:
    max_ilist_fac_need: float = 0.

    def insert_nfilled_and_size(self, nfilled, size):
        self.max_ilist_fac_need = max(self.max_ilist_fac_need, nfilled/size)

@dataclass(slots=True)
class InteractionStats:
    node2node: int = 0
    leaf2leaf: int = 0
    num_open: int = 0

@dataclass(slots=True)
class Statistics:
    allocation: AllocStats = field(default_factory=AllocStats)
    interaction: InteractionStats = field(default_factory=InteractionStats)

    # Use to block data-races, e.g. in a one host - multiple devices setup:
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

_current_statistics: ContextVar[Statistics | None] = ContextVar("_current_ctx", default=None)

def get_stats_context(key: str | None = None) -> Statistics | Any:
    stats = _current_statistics.get()
    if stats is None:
        return None
    
    if key is not None:
        return getattr(stats, key, None)
    else:
        return stats

def stats_callback(f: Callable, *args, key: str | None = None, **kwargs):
    stats = get_stats_context()
    sub_stats = get_stats_context(key)

    if stats is None or sub_stats is None: return

    def f_with_lock(*args, **kwargs):
        with stats._lock:
            f(sub_stats, *args, **kwargs)
    jax.debug.callback(f_with_lock, *args, **kwargs)

@contextmanager
def statistics(ctx: Statistics | None = None) -> Iterator[Statistics]:
    if ctx is None:
        ctx = Statistics()
    token = _current_statistics.set(ctx)
    try:
        yield ctx
    finally:
        _current_statistics.reset(token)
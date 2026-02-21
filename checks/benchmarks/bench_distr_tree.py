import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from jztree.data import Pos, PosMass
from jztree.tree import distr_zsort, distr_zsort_and_tree
from jztree.config import TreeConfig
from jztree.jax_ext import shard_map_constructor
from jztree_utils import ics

def pow2_upto(n: int) -> list[int]:
    out = []
    p = 1
    while p <= n:
        out.append(p)
        p *= 2
    return out

_MAX = jax.device_count()          # devices visible to this process
NDEVS = pow2_upto(_MAX)[::-1]

def mk_pos(Ntot, alloc_fac=1.2, ndev=4):
    N = Ntot // ndev
    @jax.shard_map(out_specs=P("gpus"), in_specs=P("gpus"), mesh=get_mesh(ndev))
    def f():
        rank = jax.lax.axis_index(axis_name="gpus")
        Nextra = int(N*(alloc_fac-1.))
        pos = jax.random.normal(jax.random.key(rank), (int(N+Nextra),3))
        pos = pos.at[-Nextra:].set(jnp.nan)

        return pos

    return f

def get_mesh(ndev=-1):
    # if ndev <= 4:
    return jax.make_mesh((ndev,), ('gpus',), axis_types=(AxisType.Auto))
    # else:
    #     return jax.make_mesh((ndev//4, 4), ('nodes', 'gpus'), axis_types=(AxisType.Auto,AxisType.Auto))

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("ndev", NDEVS)
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_multi_zsort(jax_bench, ndev):
    part = ics.gaussian_particles.smap(get_mesh(ndev), jit=True)(512**3, npad=int(512**3*0.2))
    fzs = distr_zsort.smap(get_mesh(ndev), jit=True)

    jb = jax_bench(jit_rounds=5, jit_warmup=1, eager_rounds=0, eager_warmup=0)
    
    partz = jb.measure(fn_jit=fzs, part=part, tag="random")[1][0]
    jb.measure(fn_jit=fzs, part=partz, tag="sorted")
    partz.pos = partz.pos + 1e-2 * part.pos
    jb.measure(fn_jit=fzs, part=partz, tag="displaced")

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("ndev", NDEVS)
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_multi_tree(jax_bench, ndev):
    part = ics.gaussian_particles.smap(get_mesh(ndev), jit=True)(512**3, npad=int(512**3*0.2))

    jb = jax_bench(jit_rounds=5, jit_warmup=1)

    get_tree = distr_zsort_and_tree.smap(get_mesh(ndev), jit=True)
    jb.measure(None, get_tree, part, cfg_tree=TreeConfig(), tag="sort_and_tree")
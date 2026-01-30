import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from jztree.data import Pos, PosMass
from jztree.tree import distributed_zsort, distr_zsort_and_tree
from jztree.config import TreeConfig

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

def mksort(ndev=4):
    @jax.shard_map(out_specs=P("gpus"), in_specs=P("gpus"), mesh=get_mesh(ndev))
    def f(pos):
        posz = distributed_zsort(pos)
        return posz

    return lambda pos=1: f(pos) # Hack to make function accept kw arguments

def get_mesh(ndev=-1):
    return jax.sharding.Mesh(jax.devices()[:ndev], ('gpus',), axis_types=(AxisType.Auto))

@pytest.mark.parametrize("ndev", NDEVS)
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_multi_zsort(jax_bench, ndev):
    Ntot = 256**3*ndev
    
    pos = jax.jit(mk_pos(Ntot, alloc_fac=1.2, ndev=ndev))()
    part = Pos(pos=pos, num=Ntot//ndev, num_total=Ntot)
    fzs = jax.jit(mksort(ndev=ndev))

    jb = jax_bench(jit_rounds=20, jit_warmup=2, eager_rounds=0, eager_warmup=0)

    partz = jb.measure(fn_jit=fzs, pos=part, tag="random")[1]
    posz1 = jb.measure(fn_jit=fzs, pos=partz, tag="sorted")[1]
    partz.pos = partz.pos + 1e-2 * jax.random.uniform(jax.random.key(0), partz.pos.shape)
    posz2 = jb.measure(fn_jit=fzs, pos=partz.pos, tag="displaced")[1]

@pytest.mark.parametrize("ndev", NDEVS)
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_multi_tree(jax_bench, ndev):
    Ntot = 256**3*ndev

    pos = jax.jit(mk_pos(Ntot, alloc_fac=1.2, ndev=ndev))()

    @jax.shard_map(out_specs=P("gpus"), in_specs=P("gpus"), mesh=get_mesh(ndev))
    def get_tree(pos):
        cfg_tree = TreeConfig()
        part = PosMass(pos=pos, mass=jnp.ones_like(pos[...,0]), num=Ntot//ndev, num_total=Ntot)
        partz, th = distr_zsort_and_tree(part, cfg_tree)
        return partz, th
    get_tree.jit = jax.jit(get_tree)

    jb = jax_bench(jit_rounds=5, jit_warmup=1, eager_rounds=0, eager_warmup=0)

    posz = jb.measure(get_tree, get_tree.jit, pos, tag="sort_and_tree")[1]
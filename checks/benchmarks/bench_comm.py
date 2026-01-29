import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from jztree.comm import all_to_all_with_irank, all_to_all_with_permute, arange_for_comm

from fmdj_utils.ics import gaussian_blob

def pow2_upto(n: int) -> list[int]:
    out = []
    p = 1
    while p <= n:
        out.append(p)
        p *= 2
    return out

_MAX = jax.device_count()          # devices visible to this process
NDEVS = pow2_upto(_MAX)[::-1]

def get_mesh(ndev=-1):
    return jax.sharding.Mesh(jax.devices()[:ndev], ('gpus',), axis_types=(AxisType.Auto))

def def_run(ndev, self_prob=0., copy_self=True, Nperdev=256**3, pytree = False, pack_pytree=True, permute=False):
    @jax.shard_map(in_specs=P(), out_specs=P("gpus"), mesh=get_mesh(ndev))
    def run():
        rank = jax.lax.axis_index(axis_name="gpus")
        x = jax.random.uniform(jax.random.key(rank), Nperdev)
        xpad = jnp.pad(x, (0, int(Nperdev*0.2)))
        irank = jax.random.randint(jax.random.key(rank+113), xpad.shape, 0, ndev)
        irank = jnp.where(xpad <= self_prob, rank, irank)
        if pytree:
            xpad = [xpad, xpad]
        if permute:
            xsort, dev_spl, isort = arange_for_comm(irank, xpad, num=Nperdev, axis_name="gpus")
            x, dev_spl = all_to_all_with_permute(xsort, dev_spl, buffer_bytes=1024*1024*8)
        else:
            pack_pytree_len = int(Nperdev*1.2) if pack_pytree else None
            x, dev_spl = all_to_all_with_irank(irank, xpad, num=Nperdev, axis_name="gpus", copy_self=copy_self, pack_pytree_len=pack_pytree_len)
        return x
    return jax.jit(run)

@pytest.mark.parametrize("ndev", NDEVS)
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_all_to_all(jax_bench, ndev):
    jb = jax_bench(jit_rounds=10, jit_loops=1, jit_warmup=2)

    for copy_self in True, False:
        prefix = 'copy_' if copy_self else ''
        jb.measure(fn_jit=def_run(ndev, 0., copy_self), tag=f"{prefix}p0.0")
        jb.measure(fn_jit=def_run(ndev, 0.5, copy_self), tag=f"{prefix}p0.5")
        jb.measure(fn_jit=def_run(ndev, 0.9, copy_self), tag=f"{prefix}p0.9")
        jb.measure(fn_jit=def_run(ndev, 0.99, copy_self), tag=f"{prefix}p0.99")
        jb.measure(fn_jit=def_run(ndev, 1.0, copy_self), tag=f"{prefix}p1.0")

@pytest.mark.parametrize("ndev", NDEVS)
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_permute_all_to_all(jax_bench, ndev):
    jb = jax_bench(jit_rounds=10, jit_loops=1, jit_warmup=2)
    prefix="permute"
    jb.measure(fn_jit=def_run(ndev, 0., permute=True), tag=f"{prefix}p0.0")
    jb.measure(fn_jit=def_run(ndev, 0.5, permute=True), tag=f"{prefix}p0.5")
    jb.measure(fn_jit=def_run(ndev, 0.9, permute=True), tag=f"{prefix}p0.9")
    jb.measure(fn_jit=def_run(ndev, 0.99, permute=True), tag=f"{prefix}p0.99")
    jb.measure(fn_jit=def_run(ndev, 1.0, permute=True), tag=f"{prefix}p1.0")

@pytest.mark.parametrize("ndev", NDEVS)
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_pytree(jax_bench, ndev):
    jb = jax_bench(jit_rounds=10, jit_loops=1, jit_warmup=2)

    for pack_pytree in True, False:
        prefix = 'packed_' if pack_pytree else ''
        kwargs = dict(copy_self=True, pack_pytree=pack_pytree, pytree=True)
        jb.measure(fn_jit=def_run(ndev, 0., **kwargs), tag=f"{prefix}p0.0")
        jb.measure(fn_jit=def_run(ndev, 0.5, **kwargs), tag=f"{prefix}p0.5")
        jb.measure(fn_jit=def_run(ndev, 0.9, **kwargs), tag=f"{prefix}p0.9")
        jb.measure(fn_jit=def_run(ndev, 0.99, **kwargs), tag=f"{prefix}p0.99")
        jb.measure(fn_jit=def_run(ndev, 1.0, **kwargs), tag=f"{prefix}p1.0")

def smap_jit(f, ndev):
    fsm = jax.shard_map(f, in_specs=P(), out_specs=P("gpus"), mesh=get_mesh(ndev))
    return jax.jit(lambda x: fsm(x)) # the lambda helps with passing x as keyword argument

@pytest.mark.parametrize("ndev", NDEVS)
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_meta_comm(jax_bench, ndev):
    Nperdev = 256**3
    axis_name = "gpus"

    def rng():
        rank = jax.lax.axis_index(axis_name)
        return jax.random.uniform(jax.random.key(rank), Nperdev)
    def get_sum(x):
        return jnp.sum((rng()+x)**2)
    
    jb = jax_bench(jit_rounds=10, jit_loops=5, jit_warmup=4)

    def nocomm(x):
        return get_sum(x).reshape(1)
    jb.measure(fn_jit=smap_jit(nocomm, ndev), x=0., tag="nocomm")

    def psum(x):
        return jax.lax.psum(get_sum(x), axis_name).reshape(-1)
    jb.measure(fn_jit=smap_jit(psum, ndev), x=0., tag="psum")

    def allgather(x):
        return jax.lax.all_gather(get_sum(x), axis_name).reshape(-1)
    jb.measure(fn_jit=smap_jit(allgather, ndev), x=0., tag="allgather")

    def all_to_all(x):
        val = get_sum(x) * jnp.arange(ndev)
        return jax.lax.all_to_all(val, axis_name, 0, 0, tiled=True).reshape(-1)
    jb.measure(fn_jit=smap_jit(all_to_all, ndev), x=0., tag="all_to_all")

    def ragged(x):
        val = get_sum(x) * jnp.arange(ndev)
        ones = jnp.ones(ndev, dtype=jnp.int32)
        devices = jnp.arange(ndev)
        out = jax.lax.ragged_all_to_all(val, val, devices, ones, devices, ones, axis_name=axis_name)
        return out
    jb.measure(fn_jit=smap_jit(ragged, ndev), x=0., tag="ragged")
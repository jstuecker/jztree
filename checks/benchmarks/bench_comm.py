import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from jztree.comm import all_to_all_with_irank, all_to_all_with_permute, arange_for_comm
from jztree.comm import all_to_all_with_splits, nested_all_to_all_with_splits
from jztree.jax_ext import shard_map_constructor

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
            x, dev_spl = all_to_all_with_irank(irank, xpad, num=Nperdev, axis_name="gpus", copy_self=copy_self, pack_pytree=pack_pytree)
        return x
    return jax.jit(run)

@pytest.mark.shrink_in_quick(keep_index=0)
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

@pytest.mark.shrink_in_quick(keep_index=0)
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

@pytest.mark.shrink_in_quick(keep_index=0)
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

@pytest.mark.shrink_in_quick(keep_index=0)
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

    def allgather_samp(x):
        rank = jax.lax.axis_index("gpus")
        data = jax.random.uniform(jax.random.key(rank), (1024,1024,16))**x + x
        return jax.lax.all_gather(data, axis_name).reshape(-1)
    jb.measure(fn_jit=smap_jit(allgather_samp, ndev), x=0., tag="allgather_samp")

    def all_to_all(x):
        val = get_sum(x) * jnp.arange(ndev)
        return jax.lax.all_to_all(val, axis_name, 0, 0, tiled=True).reshape(-1)
    jb.measure(fn_jit=smap_jit(all_to_all, ndev), x=0., tag="all_to_all")

    def all_to_all_samp(x):
        rank = jax.lax.axis_index("gpus")
        ndev = jax.lax.axis_size("gpus")
        data = jax.random.uniform(jax.random.key(rank), (ndev,1024,1024,256//ndev))**x + x
        return jax.lax.all_to_all(data, axis_name, 0, 0, tiled=True).reshape(-1)
    jb.measure(fn_jit=smap_jit(all_to_all_samp, ndev), x=0., tag="all_to_all_samp")

    def ragged(x):
        val = get_sum(x) * jnp.arange(ndev)
        ones = jnp.ones(ndev, dtype=jnp.int32)
        devices = jnp.arange(ndev)
        out = jax.lax.ragged_all_to_all(val, val, devices, ones, devices, ones, axis_name=axis_name)
        return out
    jb.measure(fn_jit=smap_jit(ragged, ndev), x=0., tag="ragged")

def n_ax_comm(MB: int, direct: bool = False, normal_all2all=False):
    axes = jax.sharding.get_abstract_mesh().axis_names
    rank = jax.lax.axis_index(axes)
    N = int(1024**2 * MB / 4)
    data = jax.random.randint(jax.random.key(rank), int(N*1.1), 0, 100, dtype=jnp.int32)

    def splits(ndev):
        dev_spl = jnp.arange(ndev+1)*(N//ndev) 
        return dev_spl

    if normal_all2all:
        res = jax.lax.all_to_all(data[0:N], axes, split_axis=0, concat_axis=0, tiled=True)
    elif direct:
        res, dev_spl = all_to_all_with_splits(data, splits(jax.lax.axis_size(axes)), axis_name=axes, copy_self=True)
    else:
        res, dev_spl = nested_all_to_all_with_splits(data, splits(jax.lax.axis_size(axes)), copy_self=True)
    
    return jnp.mean(res[:N]**2)
n_ax_comm.smap = shard_map_constructor(n_ax_comm, in_specs=(None, None, None), static_argnums=(0,1,2))

@pytest.mark.shrink_in_quick(keep_index=2)
@pytest.mark.parametrize("MB", (1,32,128,512,2048))
@pytest.mark.skipif(jax.device_count() < 4, reason="Requires >= 4 devices")
def bench_two_axes(jax_bench, MB):
    jb = jax_bench(jit_rounds=5, jit_loops=4, jit_warmup=2)

    for ndev in NDEVS[:-2]:
        mesh = jax.make_mesh((ndev//4,4), ("nodes", "gpus"), axis_types=(jax.sharding.AxisType.Explicit,)*2)
        
        # In parallel setups very first compilation seems to take longer... do it once
        # to get reliable timing
        n_ax_comm.smap(mesh, jit=True)(1, True)
    
        res1 = jb.measure(None, n_ax_comm.smap(mesh, jit=True), MB, True, tag=f"direct_N{ndev}")[1]
        res2 = jb.measure(None, n_ax_comm.smap(mesh, jit=True), MB, False, tag=f"indirect_N{ndev}")[1]

        # assert res1 == pytest.approx(res2, rel=1e-4)

@pytest.mark.shrink_in_quick(keep_index=2)
@pytest.mark.parametrize("MB", (1,32,128,512,2048))
@pytest.mark.skipif(jax.device_count() < 64, reason="Requires 64 devices")
def bench_indirect_mesh(jax_bench, MB):
    jb = jax_bench(jit_rounds=20, jit_loops=1, jit_warmup=2)

    expl = jax.sharding.AxisType.Explicit

    mesh1 = jax.make_mesh((64,), ("gpus"), axis_types=(expl,))
    mesh2 = jax.make_mesh((16,4), ("nodes", "gpus"), axis_types=(expl,)*2)
    mesh3 = jax.make_mesh((4,4,4), ("snodes", "nodes", "gpus"), axis_types=(expl,)*3)
    mesh4 = jax.make_mesh((2,2,2,2,4), ("i", "j", "k", "l", "gpus"), axis_types=(expl,)*5)
    
    f1 = n_ax_comm.smap(mesh1, jit=True)
    f2 = n_ax_comm.smap(mesh2, jit=True)
    f3 = n_ax_comm.smap(mesh3, jit=True)
    f4 = n_ax_comm.smap(mesh4, jit=True)

    r1 = jb.measure(None, f1, MB, False, tag=f"mesh_64")[1]
    r2 = jb.measure(None, f2, MB, False, tag=f"mesh_16x4")[1]
    r3 = jb.measure(None, f3, MB, False, tag=f"mesh_4x4x4")[1]
    r4 = jb.measure(None, f4, MB, False, tag=f"mesh_2x2x2x2x4")[1]
    
    jb.measure(None, f1, MB, False, True, tag=f"fixed_all2all")

    def p0(x):
        return jax.device_put(x, jax.sharding.NamedSharding(mesh1, P()))

    assert p0(r2) == pytest.approx(p0(r1), rel=1e-4)
    assert p0(r3) == pytest.approx(p0(r1), rel=1e-4)
    assert p0(r4) == pytest.approx(p0(r1), rel=1e-4)

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("ndev", NDEVS)
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple")
def bench_a2a_ndev(jax_bench, ndev):
    jb = jax_bench(jit_rounds=20, jit_loops=1, jit_warmup=2)

    expl = jax.sharding.AxisType.Explicit

    mesh1 = jax.make_mesh((ndev,), ("gpus"), axis_types=(expl,))
    
    f1 = n_ax_comm.smap(mesh1, jit=True)

    jb.measure(None, f1, 1, False, tag=f"ragged_all2all_1MB")[1]
    jb.measure(None, f1, 1, False, True, tag=f"all2all_1MB")[1]

    jb.measure(None, f1, 2048, False, tag=f"ragged_all2all_2GB")[1]
    jb.measure(None, f1, 2048, False, True, tag=f"all2all_2GB")[1]
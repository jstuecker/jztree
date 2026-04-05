# Advanced Guide: Multi-GPU
This is a guide on how to use the multi-GPU functionality in **jz-tree**. All functionality works both in
cases with a single host and multiple GPUs (e.g. if you allocate 4 GPUs on a single node with 1 task)
or in multi-host multi-GPU setups (e.g. 4 nodes with 4 GPUs each with 16 tasks).

However, to start with a simple interactive scenario, this guide assumes you are executing this
interactively with a single host on a multi-GPU system. For a multi-host case, the main difference
is that you'd need to execute `jax.distributed_initialize()` at the beginning, as we will explain later.

Let us start by importing the relevant modules and by declaring a mesh that includes all available
GPUs:


```python
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P 
import jztree as jz
from jztree_utils import ics
import numpy as np

mesh = jax.make_mesh((jax.device_count(),), axis_names=("gpus",), axis_types=jax.sharding.AxisType.Auto)
```

## shard_map in JAX
All the multi-GPU code in **jz-tree** is assumed to run from inside a [shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html) in JAX. We will briefly reiterate the most important concepts here.

Normally, JAX takes a global perspective on arrays that are distributed across several devices,
and it permits calling global operations that may internally rearrange the data between
devices (i.e. it may change the 'sharding'). However, this model is not very useful for scenarios
where data may be padded and imbalanced between devices.

A `shard_map` instead allows writing a program locally from the perspective of a single device
(i.e. we only see locally available data), and it allows us
to declare explicit communication directives between devices. The context inside a `shard_map`
is very similar to that of an MPI program. The `shard_map` is an interface that switches
between the perspectives.

Consider the following example:


```python
@jax.jit
@jax.shard_map(in_specs=P(), out_specs=(P("gpus"), P("gpus"), P()), mesh=mesh)
def f(seed):
    # Here, we write code from the perspective of one GPU
    # that may communicate with others

    rank = jax.lax.axis_index("gpus")
    pos = jax.random.uniform(jax.random.key(seed + rank), (10000, 3))

    loc = jnp.sum(pos**2)
    total = jax.lax.psum(loc, axis_name="gpus")

    print("inside", jax.typeof(pos), jax.typeof(loc), jax.typeof(total))

    return pos, loc.reshape(1,), total

pos, loc, total = f(0)

print("outside", jax.typeof(pos), jax.typeof(loc), jax.typeof(total))
```

    inside float32[10000,3]{V:gpus} float32[]{V:gpus} float32[]
    outside float32[40000,3] float32[4] float32[]


For each input and output of the function, we declare a partition spec that tells JAX how the
data is supposed to be split (in the input) or joined across the GPUs. `P()` means replication
across all GPUs (therefore, not changing the shape), `P("gpus")` means that the data should be
split in inputs and tiled in outputs along the "gpus" axis of the provided mesh. `None` can be used
to indicate a static parameter.

Inside the function, you can see that we create random numbers with different seeds on each device
and then we calculate on each GPU locally the sum of squares and later do `jax.lax.psum` (involving
communication) that gets us the global sum.

With `jax.typeof` we can check the types, revealing that `pos` and `loc` vary across GPUs, but
`total` is the same number on every device. **JAX**'s `shard_map` always tiles the varying outputs; this is why
we had to reshape `loc` to be 1D, otherwise we get an error here.

This behavior is actually very inconvenient when working with particle data that may be padded and
have imbalances and some scalar info (like the particle number or masses). This is why in **jz-tree** we have defined a modified version `expanding_shard_map`:


```python
def f(seed):
    rank, ndev, axis_name = jz.comm.get_rank_info()
    print("rank info:", rank, ndev, axis_name)

    pos = jax.random.uniform(jax.random.key(seed + rank), (10000, 3))

    loc = jnp.sum(pos**2)
    total = jax.lax.psum(loc, axis_name=axis_name)

    print("inside", jax.typeof(pos), jax.typeof(loc), jax.typeof(total))

    return pos, loc, total

f_smapped = jz.jax_ext.expanding_shard_map(f, out_specs=(P(-1), P(-1), P()), in_specs=P(), mesh=mesh)
f_smapped_jitted = jax.jit(f_smapped)

pos, loc, total = f_smapped_jitted(0)

print("outside", pos.shape, loc.shape, total.shape)
```

    rank info: JitTracer(int32[]{V:gpus}) 4 ('gpus',)
    inside float32[10000,3]{V:gpus} float32[]{V:gpus} float32[]
    outside (4, 10000, 3) (4,) ()


Note the following main differences:
* Varying outputs get an extra leading dimension with length **ndev**
* The function allows us to put `P(-1)` as a partition spec (this is not a standard **JAX** feature, [but I requested it](https://github.com/jax-ml/jax/issues/34752)). This simply shards across all dimensions that are in the provided mesh. This allows code to stay agnostic of how mesh axes are named and of their shapes. (E.g. if you provided a mesh with two axes, e.g. ("nodes", "gpus") to the code above, it would still work.)
* We call a function `jz.comm.get_rank_info` that gets information about the mesh, allowing us not to hard-code axis names.

The code above still has the inconvenience that we need to know the mesh when defining `f_smapped` and also the jitted version `f_smapped_jitted`. As a workaround, we have chosen in **jz-tree** to add a `shard_map_constructor`, saved on a function attribute `.smap`, to every mappable function. It allows us to create the shard-mapped and jitted versions of the function conveniently:


```python
def f(seed):
    rank, ndev, axis_name = jz.comm.get_rank_info()
    print("rank info:", rank, ndev, axis_name)

    pos = jax.random.uniform(jax.random.key(seed + rank), (10000, 3))

    loc = jnp.sum(pos**2)
    total = jax.lax.psum(loc, axis_name=axis_name)

    print("inside", jax.typeof(pos), jax.typeof(loc), jax.typeof(total))

    return pos, loc, total
f.smap = jz.jax_ext.shard_map_constructor(f, out_specs=(P(-1), P(-1), P()), in_specs=P())

pos, loc, total = f.smap(mesh, jit=True)(seed=0)

print("outside", pos.shape, loc.shape, total.shape)
```

    rank info: JitTracer(int32[]{V:gpus}) 4 ('gpus',)
    inside float32[10000,3]{V:gpus} float32[]{V:gpus} float32[]
    outside (4, 10000, 3) (4,) ()


This allows us to pass data easily between global and shard-mapped contexts, for example:


```python
part = ics.uniform_particles.smap(mesh, jit=True)(N=10000, npad=1000)
print(part.pos.shape, part.num)
partz = jz.tree.distr_zsort.smap(mesh, jit=True)(part, equalize=False)[0]
print(partz.pos.shape, partz.num)
```

    (4, 11000, 3) [10000 10000 10000 10000]
    (4, 11000, 3) [ 9532 10078 10448  9942]


Here, we have created a uniform random distribution and sorted it in z-order. The sort creates
a temporary imbalance (balance would be restored afterwards if `equalize=True` was passed).

However, our recommended approach is to keep as much as possible inside the shard-mapped
context and to write full programs from an "MPI-style" perspective as much as possible.

## Particle-data and padding
As you may have noticed in the example above, for multi-GPU scenarios our particle data needs to carry
some extra information: not every GPU will have the same number of particles at the same time, and
we need to pad the arrays to allow some extra space for imbalance. Consider the following dataclass
from `jztree.data.Pos`:

```python
@jax.tree_util.register_dataclass
@dataclass(kw_only=True, slots=True)
class Pos:
    pos: jax.Array
    num: int | jax.Array | None = None
    num_total: int | None = static_field(default=None)
```

Beyond the position array, it carries a dynamical value `num` which indicates the currently filled number of particles
(which will be less than or equal to `pos.shape[0]`) and `num_total`, which statically defines the total number of
particles.

You may define your own particle data class that follows the same interface. Whenever particles
get communicated or rearranged, we use a `jax.tree.map` approach to adapt all fields that have the
same leading dimension as `pos`. To pad particles, you may use the function `jztree.data.pad_particles`
that adds the indicated number of particles along each pytree leaf that has the correct shape. For example,
this is the implementation of `jztree_utils.ics.uniform_particles`:

```python
def uniform_particles(N, total_mass=1., seed=0, npad=0):
    rank, ndev, axis_name = get_rank_info()

    pos = jax.random.uniform(jax.random.PRNGKey(seed + rank), (N,3), dtype=jnp.float32)
    posmass = PosMass(pos=pos, mass=total_mass/(N*ndev), num=N, num_total=ndev*N)

    return pad_particles(posmass, npad)
uniform_particles.smap = shard_map_constructor(uniform_particles,
    in_specs=(None, None, None, None), out_specs=P(-1), static_argnums=(0,3)
)
```

You can see that it uses the `jztree.data.PosMass` dataclass which additionally has masses, for example,
as would be required to calculate masses in FoF catalogues. The type hints in **jz-tree**, e.g. `Pos`
in `jztree.tree.distr_zsort`, only indicate a minimal interface, but don't require that the
provided data is a subclass of the indicated class.

## Distributed kNN

If a padded instance of `Pos` is provided, the distributed kNN can be used more or less identically to the single-GPU version. However, bringing the full neighbour list back into input order is
very communication-heavy. To avoid this, it is recommended to define a reduction function that
directly extracts the property that you need while all source particles are present in z-order.

Here is an example:


```python
def get_mean_neighbour_rad(N, npad):
    part = ics.uniform_particles(N, npad=npad)

    def get_rmean(rnn, **kwargs):
        return jnp.mean(rnn[:,1:], axis=-1)

    cfg = jz.config.KNNConfig()
    cfg.tree.alloc_fac_nodes = 2.0
    rmean = jz.knn.knn(part, k=9, result="reduce", reduce_func=get_rmean)

    return part, rmean
get_mean_neighbour_rad.smap = jz.jax_ext.shard_map_constructor(
    get_mean_neighbour_rad, in_specs=(None, None), out_specs=(P(-1), P(-1)), 
    static_argnames=["N", "npad"]
)

part, rmean = get_mean_neighbour_rad.smap(mesh, jit=True)(int(1e6), npad=int(5e5))
print(part.pos.shape, rmean.shape)
print(part.pos[0,0], rmean[0,0])
```

    (4, 1500000, 3) (4, 1500000)
    [0.947667   0.9785799  0.33229148] 0.0069931187


This calculates the average distance to the 8 nearest neighbours (excluding the particle itself). For example, you could easily use this to get a local estimate of the density.
> The `reduce_func` needs to have `**kwargs`, because the code passes several other optional keyword arguments to this function, namely `part`, `rnn`, `inn`, and `origin`, so you could easily define more general reductions.

## Distributed friends-of-friends

Distributed friends-of-friends works similarly. For example:


```python
from dataclasses import asdict

def write_callback(rank, cata: jz.data.FofCatalogue):
    cata = jz.data.squeeze_catalogue(cata)
    print(f"rank: {rank} ngroups {cata.ngroups}")
    np.savez(f"fof_catalogue_rank_{rank}.npz", **asdict(cata))

def write_fof_catalogue(N, npad):
    rank, ndev, axis_name = jz.comm.get_rank_info()

    part = ics.uniform_particles(N, npad=npad)
    
    rlink = 0.7 * np.cbrt(1./part.num_total)

    partf, cata = jz.fof.fof_and_catalogue(part, rlink=rlink)

    jax.debug.callback(write_callback, rank, cata)
write_fof_catalogue.smap = jz.jax_ext.shard_map_constructor(
    write_fof_catalogue, in_specs=(None, None), out_specs=None, 
    static_argnames=["N", "npad"]
)

write_fof_catalogue.smap(mesh, jit=True)(250000, 100000)
```

    rank: 3 ngroups 310
    rank: 1 ngroups 322
    rank: 0 ngroups 337
    rank: 2 ngroups 320


```{note} 
You may notice that JIT compilation takes significantly longer than for the single-GPU version. This is because **JAX's** communication routines compile relatively slowly and FoF requires many of them.
```

Here we have written the catalogues to disk in a host callback. Inside the host callback, the data is given as NumPy arrays and we can remove the padding in the catalogue with `jztree.data.squeeze_catalogue`. Every rank writes its own NumPy file.
Here you can verify that the output makes sense:


```python
for rank in range(0,4):
    cata = np.load(f"fof_catalogue_rank_{rank}.npz")
    print("rank", rank, "ngr", cata["ngroups"], "largest ten:", np.sort(cata["count"])[::-1][0:10])
```

    rank 0 ngr 337 largest ten: [61 54 46 45 43 43 43 43 42 41]
    rank 1 ngr 322 largest ten: [64 61 51 47 45 45 43 41 40 40]
    rank 2 ngr 320 largest ten: [64 62 55 46 46 44 41 41 40 39]
    rank 3 ngr 310 largest ten: [60 52 50 47 46 44 43 43 42 41]


Of course, for an HPC code it is advisable to use a more advanced file format like HDF5 to save the data.

## Multi-host execution and performance

Generally, to execute the code with multiple hosts, all you need to add before calling any other JAX function is:
```python
import jax
jax.distributed_initialize()
```
Of course, you should use a `.py` script (rather than an `.ipynb` notebook) for your code, and with Slurm you may execute it as `srun python myscript.py`. To get good performance, it is very important that you provide a sufficient number of CPUs per task. These extra CPUs are heavily involved in communication, e.g. see [this GitHub issue](https://github.com/jax-ml/jax/discussions/34883). For the same reason, the single-host multiple-GPU case is significantly slower than the case where you provide one task per GPU. So it is recommended to only use it to interactively develop and test code.

Here is an example Slurm script that I use for some performance measurements on 4 nodes with 4 GPUs each:
```slurm
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
# ...

conda activate cu12.2
cd repos/jz-tree/checks/profiling

srun python multi_gpu.py
```

Of course, the setup may vary notably, depending on your cluster.

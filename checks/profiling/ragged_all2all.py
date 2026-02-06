import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
import time
import matplotlib.pyplot as plt
import os

save_reference = False
load_reference = True

def balanced_a2a(MB: int, ragged=True):
    axes = jax.sharding.get_abstract_mesh().axis_names
    rank = jax.lax.axis_index(axes)
    ndev = jax.lax.axis_size(axes)
    N = int(1024**2 * MB / 4)
    data = jax.random.randint(jax.random.key(rank), N, 0, 100, dtype=jnp.int32)

    if ragged:
        nsend_per_dev = N // ndev
        offsets = jnp.arange(ndev) * nsend_per_dev
        sizes = jnp.ones_like(offsets) * nsend_per_dev

        res = jax.lax.ragged_all_to_all(
            data, jnp.zeros_like(data), offsets, sizes, offsets, sizes, axis_name=axes
        )
    else:
        res = jax.lax.all_to_all(data, axes, 0, 0, tiled=True)
    
    return jnp.mean(res**2).reshape(1) # do some reduction to avoid jit pruning anything

def smap_balanced_a2a(mesh, jit=True):
    smapped = jax.shard_map(balanced_a2a, in_specs=(None, None), out_specs=P(mesh.axis_names), mesh=mesh)
    if jit:
        return jax.jit(smapped, static_argnums=(0,1))
    else:
        return smapped

def profile(mesh, MB, ragged=True, nruns=10):
    f = smap_balanced_a2a(mesh, jit=True)
    jax.block_until_ready(f(MB, ragged)) # do once in advance to compile and warmup
    times = []
    for i in range(nruns):
        t0 = time.perf_counter()
        jax.block_until_ready(f(MB, ragged))
        times.append(time.perf_counter() - t0)
    
    return 1e3*np.mean(times), 1e3*np.std(times)

def main():
    jax.distributed.initialize()

    mesh = jax.make_mesh((jax.device_count(),), ("gpus",), (AxisType.Auto,))

    mbs = (1,32,128,512,2048,8096)
    means, stds, means_rag, stds_rag = np.zeros(((4, len(mbs))), dtype=np.float32)
    
    for i, mb in enumerate(mbs):
        means[i], stds[i] = profile(mesh, mb, ragged=False)
        means_rag[i], stds_rag[i] = profile(mesh, mb, ragged=True)
    
    if jax.process_index() == 0:
        plt.plot(mbs, means_rag, marker="o", label="ragged_all2all", alpha=0.8)
        plt.fill_between(mbs, means_rag-stds_rag, means_rag+stds_rag, alpha=0.3)
        plt.plot(mbs, means, marker="o", label="all2all", alpha=0.8)
        plt.fill_between(mbs, means-stds, means+stds, alpha=0.3)
        if load_reference and os.path.exists("reference.npy"):
            rmean, rstd, rmean_rag, rstd_rag = np.load("reference.npy")
            plt.plot(mbs, rmean_rag, marker="o", label="ref_ragged", alpha=0.6, color="black")
            plt.fill_between(mbs, rmean_rag-rstd_rag, rmean_rag+rstd_rag, alpha=0.3, color="black")
        
        plt.legend()
        plt.title(f"Performance with {jax.device_count()} GPUs")
        plt.ylabel("Time (ms)")
        plt.xlabel("Data per GPU (MB)")
        plt.yscale("log")
        plt.xscale("log")
        plt.savefig(f"ragged_performance_{jax.device_count()}.png", bbox_inches="tight")

        if save_reference:
            np.save("reference.npy", np.stack([means, stds, means_rag, stds_rag], axis=0))

    jax.distributed.shutdown()

if __name__ == "__main__":
    main()
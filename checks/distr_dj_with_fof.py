import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide INFO(0) + WARNING(1), keep ERROR/FATAL
import jax
import jax.numpy as jnp
import importlib
import numpy as np
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from jax.experimental.multihost_utils import sync_global_devices

import jztree as jz
from jztree_utils import ics, io

if jz.comm.should_init_jax_distributed():
    jax.distributed.initialize()

mesh = jax.make_mesh((jax.device_count(),), ("gpus",), axis_types=jax.sharding.AxisType.Explicit)

has_discodj = importlib.util.find_spec("discodj") is not None
if not has_discodj:
    raise ImportError("This example requires DISCO-DJ with multi-GPU support installed!")

def sim_and_fof(cfg: jz.config.FofConfig):
    ndev = jax.device_count()
    boxsize = 400.

    part = ics.multi_gpu_dj_sim(boxsize, num_per_device=512**3)
    ngrid = int(np.round(np.cbrt(part.num_total)))
    part = jz.data.expand_particles(part, ndev)

    rlink = 0.2 * boxsize / np.cbrt(part.num_total)

    def distr_fof(part: jz.data.ParticleData):
        part = jz.data.pad_particles(part, int(part.num_total // ndev * 0.2))
        part_fof, cata = jz.fof.distr_fof_and_catalogue(part, rlink=rlink, boxsize=boxsize, cfg=cfg)
        
        base_name = f"data/dj_sim_{boxsize}_{ngrid}_{ndev}/snap_and_fof"
        # token = io.distr_write_hdf5(base_name, fof=cata, particles=part_fof)
        # return token
        return cata
    distr_fof = jz.jax_ext.expanding_shard_map(distr_fof, mesh=mesh)
    
    return distr_fof(part)
sim_and_fof = jax.jit(sim_and_fof, static_argnums=0)


def myprint(*args, on_all = False, **kwargs):
    if jax.process_index() == 0 or on_all:
        print(*args, **kwargs)

def main():
    with jz.stats.statistics() as stats:
        cfg = jz.config.FofConfig()
        cfg.tree.alloc_fac_nodes = 1.25

        t0 = time.time()
        myprint("Compiling...")
        sim_and_fof.lower(cfg).compile()
        t1 = time.time()
        myprint(f"Done compiling ({t1-t0:.1f}s).\nRunning...")
        jax.block_until_ready(sim_and_fof(cfg))
        myprint(f"Done running ({time.time()-t1:.1f}s).")
        if jax.process_index() == 0:
            # print("stats:", jz.stats.reduce_stats_multihost(stats))
            stats.print_suggestions(cfg)    

    sync_global_devices("pre-shutdown")
    jax.distributed.shutdown()

if __name__ == "__main__":
    main()
import jax
import jax.numpy as jnp
import importlib
import numpy as np
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType

import jztree as jz
from jztree_utils import ics, io

if jz.comm.should_init_jax_distributed():
    jax.distributed.initialize()

has_discodj = importlib.util.find_spec("discodj") is not None
if not has_discodj:
    raise ImportError("This example requires DISCO-DJ with multi-GPU support installed!")

mesh = jax.make_mesh((jax.device_count(),), ("gpus",))

@jax.jit
def sim_and_fof():
    ndev = jax.device_count()
    boxsize = 1000.

    part = ics.multi_gpu_dj_sim(boxsize, num_per_device=128**3)
    ngrid = int(np.round(np.cbrt(part.num_total)))
    part = jz.data.expand_particles(part, ndev)

    rlink = 0.2 * boxsize / np.cbrt(part.num_total)

    def distr_fof(part: jz.data.ParticleData):
        part = jz.data.pad_particles(part, int(part.num_total // ndev * 0.5))
        part_fof, cata = jz.fof.distr_fof_and_catalogue(part, rlink=rlink, boxsize=boxsize)
        
        base_name = f"data/dj_sim_{ngrid}_{ndev}/snap_and_fof"
        token = io.distr_write_hdf5(base_name, fof=cata, particles=part_fof)
        return token
    distr_fof = jz.jax_ext.expanding_shard_map(distr_fof, mesh=mesh)
    
    return distr_fof(part)

def main():
    sim_and_fof()

if __name__ == "__main__":
    main()
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import custom_jax_cuda.nb_multipoles as nb_multipoles

jax.ffi.register_ffi_target("multipoles_from_particles", nb_multipoles.multipoles_from_particles(), platform="CUDA")
# jax.ffi.register_ffi_target("coarsen_multipoles", nb_multipoles.coarsen_multipoles(), platform="CUDA")

from fmdj.new_tree import TreePlane, Multipoles, Particles

def multipoles_from_particles(tp : TreePlane, part : Particles, p : int = 2, around_com : bool = True) -> Multipoles:
    assert around_com

    posm = part.posm()

    assert posm.dtype == jnp.float32
    assert tp.ispl.dtype == jnp.int32

    ncomb = np.array([1, 4, 10, 20, 35, 56, 84, 120, 165])
    
    out_mp = jax.ShapeDtypeStruct((tp.size(), ncomb[p]), posm.dtype)
    out_xcent = jax.ShapeDtypeStruct((tp.size(), 3), posm.dtype)

    mp, xcent = jax.ffi.ffi_call("multipoles_from_particles", (out_mp, out_xcent))(
        tp.ispl, posm, p=np.uint64(p), block_size=np.uint64(32)
    )
    
    return Multipoles(xcent=xcent, values=mp, p=p, around_com=True)
multipoles_from_particles.jit = jax.jit(multipoles_from_particles, static_argnames=['p', 'around_com'])

def coarsen_multipoles(mp : Multipoles, tp : TreePlane) -> Multipoles:
    """Determines the multipoles at the next coarser tree plane"""
    assert mp.around_com

    dtype = mp.values.dtype

    assert mp.values.dtype == jnp.float32
    assert tp.ispl.dtype == jnp.int32

    ncomb = np.array([1, 4, 10, 20, 35, 56, 84, 120, 165])

    out_mp = jax.ShapeDtypeStruct((tp.size(), ncomb[mp.p]), dtype)
    out_xcent = jax.ShapeDtypeStruct((tp.size(), 3), dtype)

    mp, xcent = jax.ffi.ffi_call("coarsen_multipoles", (out_mp, out_xcent))(
        tp.ispl, mp.center, mp.values, p=np.uint64(mp.p), block_size=np.uint64(32)
    )

    # # Compute the center of mass
    # mnode = jax.ops.segment_sum(mp.get(0), **kwargs)

    # dx = mp.center() - tp.geom_center()[parent]
    # mxnode = [jax.ops.segment_sum(dx[...,d]*mp.get(0), **kwargs) for d in range(3)]
    
    # if mp.around_com:
    #     xcent = jnp.stack([mxnode[d]/mnode for d in range(3)], axis=-1)
    # else:
    #     xcent = tp.geom_center()
    
    # dx = mp.center() - xcent[parent]
    
    # mpshift = shift_multipoles(mp.values, dx, p=mp.p)

    # mp_coarse = [jax.ops.segment_sum(mpshift[...,k], **kwargs) for k in range(mpshift.shape[-1])]

    # return Multipoles(
    #     xcent=xcent,
    #     values=jnp.stack(mp_coarse, axis=-1),
    #     p=mp.p,
    #     around_com=mp.around_com
    # )
coarsen_multipoles.jit = jax.jit(coarsen_multipoles)
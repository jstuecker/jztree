from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import custom_jax.nb_multipoles as nb_multipoles

jax.ffi.register_ffi_target("multipoles_from_particles", nb_multipoles.multipoles_from_particles(), platform="CUDA")

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
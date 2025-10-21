from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import custom_jax_cuda.ffi_multipoles as ffi_multipoles

jax.ffi.register_ffi_target("multipoles_from_particles", ffi_multipoles.multipoles_from_particles(), platform="CUDA")
jax.ffi.register_ffi_target("coarsen_multipoles", ffi_multipoles.coarsen_multipoles(), platform="CUDA")

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

def cj_coarsen_multipoles(mp : Multipoles, tp : TreePlane) -> Multipoles:
    """Determines the multipoles at the next coarser tree plane"""
    assert mp.around_com

    dtype = mp.values.dtype

    assert mp.values.dtype == jnp.float32
    assert tp.ispl.dtype == jnp.int32

    ncomb = np.array([1, 4, 10, 20, 35, 56, 84, 120, 165])

    out_mp = jax.ShapeDtypeStruct((tp.size(), ncomb[mp.p]), dtype)
    out_xcent = jax.ShapeDtypeStruct((tp.size(), 3), dtype)

    mpnew, xcent = jax.ffi.ffi_call("coarsen_multipoles", (out_mp, out_xcent))(
        tp.ispl, mp.values, mp.center(), p=np.uint64(mp.p), block_size=np.uint64(32)
    )
    return Multipoles(xcent=xcent, values=mpnew, p=mp.p, around_com=mp.around_com)
cj_coarsen_multipoles.jit = jax.jit(cj_coarsen_multipoles)
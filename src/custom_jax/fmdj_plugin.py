from fmdj.variants import Variant, V, VariantManager, vm, has_gpu
from fmdj import config
import jax.numpy as jnp

TAG = "cuda"

variants = VariantManager()

import custom_jax as cj
import custom_jax.cj_new_tree as cnt

# This function has a different signature, so we need to connect it manually.
def _ilist_leaf_to_leaf_cj(xpart, mpart, leaf_bounds, interactions, irange, cfg : config.Config):
    xm = jnp.concatenate([xpart, mpart[:, None]], axis=-1)
    return cj.forces.ilist_fphi(xm, leaf_bounds, jnp.abs(interactions), irange, softening=cfg.softening)[...,3]

def direct_summation_force_cj(xpart, mpart, cfg : config.Config):
    ispl, ilist = cj.forces.dense_ilist(len(xpart), bls=64)
    xm = jnp.concatenate([xpart, mpart[:,None]], axis=1)
    fphi = cj.forces.ilist_fphi.jit(xm, ispl, ilist, softening=cfg.softening, block_size=64, interactions_per_block=16)
    fphi = cfg.G() * fphi

    return fphi[..., :3]

def register():
    if has_gpu():
        variants[V.ilist_node_to_node][TAG] = Variant(cj.multipoles.ilist_node_to_node)
        variants[V.ilist_leaf_to_node][TAG] = Variant(cj.multipoles.ilist_leaf_to_node)
        variants[V.ilist_leaf_to_leaf][TAG] = Variant(_ilist_leaf_to_leaf_cj)
        variants[V.direct_summation_force][TAG] = Variant(direct_summation_force_cj)
        variants[V.multipoles_from_particles][TAG] = Variant(cnt.multipoles_from_particles)
        variants[V.coarsen_multipoles][TAG] = Variant(cnt.coarsen_multipoles)
        variants[V.evaluate_plane_interactions][TAG] = Variant(cnt.cj_evaluate_tree_plane)

        vm.register_variants(variants)
        print("Custom Jax Plugin registered!")
    else:
        print("Could not find GPU, Custom Jax Plugin not registered!")
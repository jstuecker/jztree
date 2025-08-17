from fmdj.variants import Variant, VariantManager, vm, has_gpu
from fmdj import config
import custom_jax as cj
import jax.numpy as jnp

TAG = "cuda"

variants = VariantManager()

def _ilist_node_to_node_cj(xnodes, multipoles, interactions, irange, cfg : config.Config):
    return cj.multipoles.ilist_node_to_node(xnodes, multipoles, interactions, irange=irange, p=cfg.p, eps=cfg.softening)

def _ilist_leaf_to_node_cj(xnodes, xpart, mpart, leaf_bounds, interactions, irange, cfg : config.Config):
    return cj.multipoles.ilist_leaf_to_node(xnodes, xpart, mpart, leaf_bounds, jnp.abs(interactions), irange, p=cfg.p, eps=cfg.softening)

def _ilist_leaf_to_leaf_cj(xpart, mpart, leaf_bounds, interactions, irange, cfg : config.Config):
    xm = jnp.concatenate([xpart, mpart[:, None]], axis=-1)
    return cj.forces.ilist_fphi(xm, leaf_bounds, jnp.abs(interactions), irange, eps=cfg.softening)[...,3]

def register():
    if has_gpu():
        variants.ilist_node_to_node[TAG] = Variant(_ilist_node_to_node_cj)
        variants.ilist_leaf_to_node[TAG] = Variant(_ilist_leaf_to_node_cj)
        variants.ilist_leaf_to_leaf[TAG] = Variant(_ilist_leaf_to_leaf_cj)

        vm.register_variants(variants)
        print("Custom Jax Plugin registered!")
    else:
        print("Could not find GPU, Custom Jax Plugin not registered!")
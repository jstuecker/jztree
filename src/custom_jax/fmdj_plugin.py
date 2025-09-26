from fmdj.variants import Variant, V, VariantManager, vm, has_gpu
from fmdj import config
import custom_jax as cj
import jax.numpy as jnp

TAG = "cuda"

variants = VariantManager()

# This decorator helps to convert config parameters to keyword arguments for the functions,
# reducing boilerplate code when connecting the variants as below.
def cfg_to_kwargs(func, kwargnames=()):
    def wrapper(*args, cfg=None, **kwargs):
        if cfg is not None:
            for name in kwargnames:
                if name not in kwargs:
                    kwargs[name] = getattr(cfg, name)
        return func(*args, **kwargs)
    return wrapper

# This function has a different signature, so we need to connect it manually.
def _ilist_leaf_to_leaf_cj(xpart, mpart, leaf_bounds, interactions, irange, cfg : config.Config):
    xm = jnp.concatenate([xpart, mpart[:, None]], axis=-1)
    return cj.forces.ilist_fphi(xm, leaf_bounds, jnp.abs(interactions), irange, softening=cfg.softening)[...,3]

def direct_summation_force_cj(xpart, mpart, cfg : config.Config):
    ispl, ilist = cj.forces.dense_ilist(len(xpart), bls=64)
    xm = jnp.concatenate([xpart, mpart[:,None]], axis=1)
    fphi = cj.forces.ilist_fphi.jit(xm, ispl, ilist, softening=cfg.softening, block_size=64, interactions_per_block=16)

    if cfg.get_potential:
        return fphi[..., :3], fphi[..., 3]
    else:
        return fphi[..., :3]

def register():
    if has_gpu():
        variants[V.ilist_node_to_node][TAG] = Variant(cfg_to_kwargs(cj.multipoles.ilist_node_to_node, ("p", "softening")))
        variants[V.ilist_leaf_to_node][TAG] = Variant(cfg_to_kwargs(cj.multipoles.ilist_leaf_to_node, ("p", "softening")))
        variants[V.ilist_leaf_to_leaf][TAG] = Variant(_ilist_leaf_to_leaf_cj)
        variants[V.direct_summation_force][TAG] = Variant(direct_summation_force_cj)

        vm.register_variants(variants)
        print("Custom Jax Plugin registered!")
    else:
        print("Could not find GPU, Custom Jax Plugin not registered!")
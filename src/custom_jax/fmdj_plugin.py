from fmdj.variants import Variant, VariantManager, vm, has_gpu
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

def register():
    if has_gpu():
        variants.ilist_node_to_node[TAG] = Variant(cfg_to_kwargs(cj.multipoles.ilist_node_to_node, ("p", "softening")))
        variants.ilist_leaf_to_node[TAG] = Variant(cfg_to_kwargs(cj.multipoles.ilist_leaf_to_node, ("p", "softening")))
        variants.ilist_leaf_to_leaf[TAG] = Variant(_ilist_leaf_to_leaf_cj)

        vm.register_variants(variants)
        print("Custom Jax Plugin registered!")
    else:
        print("Could not find GPU, Custom Jax Plugin not registered!")
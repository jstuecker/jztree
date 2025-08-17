# custom_jax/fmdj_plugin.py
from fmdj.variants import Variant, VariantManager, vm, has_gpu
import custom_jax as cj
import jax.numpy as jnp

def test_func(x, cfg):
    return x

tag = "cj"
priority = 1
def only_register_if(cfg):
    return has_gpu()
# def reg(op, fn):
#     vm.reg
    # register_variant(op, name=name, fn=fn, priority=priority, only_if=only_register_if)

variants = VariantManager()

def _ilist_leaf_to_leaf_cj(xpart, mpart, leaf_bounds, interactions, irange, max_leaf_size=64, eps=0.):
    print("cj ilist_leaf_to_leaf called with eps=%f" % eps)
    xm = jnp.concatenate([xpart, mpart[:, None]], axis=-1)
    return cj.forces.ilist_fphi(xm, leaf_bounds, jnp.abs(interactions), irange, eps=eps)[...,3]*2

def register():
    if has_gpu():
        variants.testfunc[tag] = Variant(test_func)
        variants.ilist_leaf_to_leaf[tag] = Variant(_ilist_leaf_to_leaf_cj)

        vm.register_variants(variants)
        print("Custom Jax Plugin registered!")
    else:
        print("Could not find GPU, Custom Jax Plugin not registered.")
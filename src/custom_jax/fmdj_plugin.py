# custom_jax/fmdj_plugin.py
from fmdj.variants import variant, has_gpu, has_pkg, register_variant
import custom_jax as cj
import jax.numpy as jnp


def test_func(x, p=3.):
    return x - p

name = "cj"
priority = 1
def only_register_if():
    return has_gpu() and has_pkg("custom_jax")
def reg(op, fn):
    register_variant(op, name=name, fn=fn, priority=priority, only_if=only_register_if)

def _ilist_leaf_to_leaf_cj(xpart, mpart, leaf_bounds, interactions, irange, max_leaf_size=64, eps=0.):
    print("cj ilist_leaf_to_leaf called with eps=%f" % eps)
    xm = jnp.concatenate([xpart, mpart[:, None]], axis=-1)
    return cj.forces.ilist_fphi(xm, leaf_bounds, jnp.abs(interactions), irange, eps=eps)[...,3]*2

def register():
    reg("testfunc", fn=test_func)
    reg("ilist_leaf_to_leaf", fn=_ilist_leaf_to_leaf_cj)

    print("Custom Jax Plugin registered!")
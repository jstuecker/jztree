# custom_jax/fmdj_plugin.py
from fmdj.variants import variant, has_gpu, has_pkg, register_variant

# @variant("testfunc", name="cj", priority=10, only_if=lambda: has_gpu() and has_pkg("custom_jax"))
# def _analytic_gpu(x, *, theta=0.6):
#     from .runtime import analytic_potential_gpu
#     return analytic_potential_gpu(x)

def test_func(x, p=3.):
    return x - p

def register():
    register_variant("testfunc", name="cj", fn=test_func, priority=1, 
                    only_if=lambda: has_gpu() and has_pkg("custom_jax"))
    print("Custom Jax Plugin registered!")
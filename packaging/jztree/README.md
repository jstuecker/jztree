# jz-tree
jz-tree offers a framwork for GPU-friendly implementations of tree algorithms in jax (with a CUDA backend). Currently nearest neighbour search and friends-of-friends are supported and deliver top performance.

Install either with
```bash
pip install jztree[cuda12]
```
or 
```bash
pip install jztree[cuda13]
```
(depending on your jax/cuda setup).

For detailled installation instructions, please check [the documentation](https://jstuecker.github.io/jztree/installation.html) or the [code repository](https://github.com/jstuecker/jztree).
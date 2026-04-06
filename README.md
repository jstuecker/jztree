# jz-tree
jz-tree offers a framwork for GPU-friendly implementations of tree algorithms in jax (with a CUDA backend). Currently nearest neighbour search and friends-of-friends are implemented and deliver top performance.

For installation instructions, please check [the documentation](https://jstuecker.github.io/jztree/installation.html)!


## Third-party components
Some CUDA submodules use NVIDIA CUB (BSD-3-Clause licensed).
See src/jztree_cuda/THIRD_PARTY_NOTICES for details.
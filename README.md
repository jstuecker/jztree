# jz-tree

**jz-tree** provides GPU-native Morton (z-order) trees for fast spatial algorithms in JAX, with a performance-critical CUDA backend.

It includes:

- Fast nearest-neighbour searches and friends-of-friends group finding.
- Single- and multi-GPU execution.
- Reusable tree construction and traversal routines for building other algorithms, such as [jz-fmm](https://jstuecker.github.io/jzfmm/).

**[Documentation](https://jstuecker.github.io/jztree/)** · [Installation](https://jstuecker.github.io/jztree/installation.html) · [Getting started](https://jstuecker.github.io/jztree/quickstart.html) · [Attribution](https://jstuecker.github.io/jztree/attribution.html)

## Performance

![Nearest-neighbour search performance compared with other libraries](docs/_static/knn_libraries_comparison.png)

Comparison of 30-nearest-neighbour searches in three dimensions on a single NVIDIA A100 GPU. See the [documentation](https://jstuecker.github.io/jztree/) for more information.

## Attribution

If you publish work using **jz-tree**, please cite the paper: [arXiv:2604.05885](https://arxiv.org/abs/2604.05885). See the [attribution page](https://jstuecker.github.io/jztree/attribution.html) for author, funding, and license information.

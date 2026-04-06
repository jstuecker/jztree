# JZ-Tree
Jz-Tree offers a framwork for GPU-friendly implementations of tree algorithms in jax (with a CUDA backend).
Currently nearest neighbour search and friends-of-friends are implemented and deliver top
performance.

For more details check [the documentation](https://jstuecker.github.io/jztree/installation.html)!

# Installation

## Via pip
Coming soon!

## Build from sources
First of all, clone the repository
```bash
git clone https://github.com/jstuecker/jz-tree
```
Then you need to check whether your GPU supports CUDA13 or CUDA12 (older CUDA versions are not supported
by jax). To install with CUDA13 you need
* A [GPU with compute capability](https://developer.nvidia.com/cuda-gpus?utm_source=chatgpt.com) >=7.5.
* A sufficiently new graphics driver [>= 580](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). 
You can check your GPU and your driver with `nvidia-smi`

Take note of your compute capability. You can significantly speed up the build time 
by setting the `CUDAARCHS` environment variable, e.g.
```bash
export CUDAARCHS=87
```
for compute capability 8.7. By default we use `CUDAARCHS="all"` to build for all architectures. This
may taking a very long time. You may also provide `CUDAARCHS="native"` if you are building on the
same system where you want to run the code.

### CUDA13 installation

The simplest way to install with CUDA13 is via `pip`. First, install the build dependencies:


```
pip install jax[cuda13] scikit-build-core nanobind cmake>=3.24 setuptools_scm
```
Finally, install **jz-tree** with `--no-build-isolation`
```
pip install -e . --no-build-isolation
```
```{note}
If you do an editable installation without `--no-build-isolation`, you python may have problems to
locate the CUDA modules.
```
(Note: Installation speed my be significantly higher with [uv pip](https://docs.astral.sh/uv/) ) -->
### CUDA12 installation
To build with CUDA12 independently of system installations, we require a conda distribution, since
the `nvidia-cuda-nvcc-cu12` pip package does not ship the `nvcc` compiler binary. However,
we can install it with a conda package.

Install miniforge (or any other conda distribution) / setup an environment / activate it (skip steps as appropriate)
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-Linux-x86_64.sh -b
rm Miniforge3-Linux-x86_64.sh
eval "$(~/miniforge3/bin/conda shell.bash hook)" # possibly fill in the correct directory
conda init
conda create --name jzenv -y
conda activate jzenv
```
Install prequisites via conda and pip:
```bash
export CONDA_OVERRIDE_CUDA="12.9"  # Use a version that fits your needs
conda install pip
# conda install -c conda-forge cuda-nvcc cuda-version=12 cudnn nccl conda-forge libcufft cuda-cupti libcublas libcusparse
conda install -c conda-forge pip cuda-nvcc cuda-version=12 cudnn nccl libcufft cuda-cupti libcublas libcusparse
#nvidia-cuda-crt
pip install scikit-build-core nanobind cmake>=3.24 setuptools_scm
pip install --upgrade "jax[cuda12-local]"
```
Finally, install the code with
```
pip install -e . --no-build-isolation
```
or with the [dev] optional dependencies if you'd like to use unit tests and some optional features.

### System CUDA setup:
I assume that you have already installed or loaded some CUDA version. E.g. check with
```bash
nvcc --version
```
Install:
```bash
pip install --upgrade "jax[cuda13-local]"   # or jax[cuda12-local]
pip install scikit-build-core nanobind cmake>=3.24 setuptools_scm
pip install -e . --no-build-isolation
```
Note that you may run into troubles if you have installed a second CUDA version in your python 
environment.


### Speeding up build-time

As mentioned earlier, the primary way to speed up build is to provide the CUDAARCHS environment
variable. For example,
```
CUDAARCHS=87 pip install -e . --no-build-isolation
```

A more advanced way of reducing the build time is to reduce the number of template variants that are 
instanced, by modifying the code generation script `src/_generate_ffi.py` (and executing it again).
This is explained in more detail in {ref}`CUDA kernels and automatic FFI generation <cuda-kernels-and-automatic-ffi-generation>`.

## Hello World

You can verify that the installation was succesful by running 
```bash
python checks/hello_world.py
```

This should give you something like this:

```bash
rnn: [[0.         0.00327169 0.00362817 0.00418469 0.00620413 0.00629311
  0.00657683 0.0065819 ]
 [0.         0.0018539  0.00218362 0.00325193 0.00418783 0.00457177
  0.00464585 0.00483315]
 [0.         0.00392929 0.00410999 0.00522736 0.00623543 0.00679859
  0.006818   0.00706907]]
Should be:
[[0.         0.00327169 0.00362817 0.00418469 0.00620413 0.00629311
  0.00657683 0.0065819 ]
 [0.         0.0018539  0.00218362 0.00325193 0.00418783 0.00457177
  0.00464585 0.00483315]
 [0.         0.00392929 0.00410999 0.00522736 0.00623543 0.00679859
  0.006818   0.00706907]]
```

If you get some warning like this in the beginning
```bash
E0406 01:02:01.239070 1581469 cuda_executor.cc:1743] Could not get kernel mode driver version: [INVALID_ARGUMENT: Version does not match the format X.Y.Z]
```
don't worry, it's a new *feature* in jax and can be savely ignored or silenced by defining the environment variable 
`TF_CPP_MIN_LOG_LEVEL=3`.

## Third-party components
Some CUDA submodules use NVIDIA CUB (BSD-3-Clause licensed).
See src/jztree_cuda/THIRD_PARTY_NOTICES for details.


# Useful commands
#uv python install 3.12
#uv python pin 3.12
#uv add --dev ipykernel
uv pip install -e . --no-build-isolation

# CustomJax
Some c++ cuda kernels that expose new functionality to jax via the foreign function interface (FFI)

# Installation
Installation from source can be a bit tricky, because the CUDA compiler nvcc and a couple of libraries are required. These may be accessible in three different ways (1) installed with pip (only available for CUDA>=13), (2) installed in a conda environment or (3) system installed

If option (1) is available, it is the easiest and the most reproducible option. Since CUDA13 is very new, it may require updating the GPU drivers and it is only supported on [Graphics cards with compute capability](https://developer.nvidia.com/cuda-gpus?utm_source=chatgpt.com)  >=7.5

You may consider these options in the following order:
* Your graphics card and driver support CUDA13 -> Go with option (1)
* A system installed CUDA is available (or can easily be loaded as a module) -> Go with option (3)
* If you like conda or if neither of the two options above are available, go with option (2)

How to set up for each of these cases is explained below:
### CUDA13 + pip setup:
Check that the NVIDIA "Driver Version" is new enough [>= 580](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). Note that CUDA 13 is not supported on older GPUs
```bash
nvidia-smi
```
If it is, go ahead and:
```bash
pip install .
```
To check whether the installation was successful, run
```
python src/benchmarks/hello_world.py
```
(Note: Installation speed my be significantly higher with [uv pip](https://docs.astral.sh/uv/) )
### CONDA + CUDA setup:
Install miniforge (or any other conda distribution) / setup an environment / activate it (skip steps as appropriate)
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-Linux-x86_64.sh -b
rm Miniforge3-Linux-x86_64.sh
eval "$(/root/miniforge3/bin/conda shell.bash hook)"
conda init
conda create --name cjtest -y
conda activate cjtest
```
Install prequisites via conda
```bash
conda install "jaxlib=0.7=*cuda12*" jax cuda-nvcc -c conda-forge   # replace jax/CUDA version as needed
conda install pip
```
Install the package
```bash
export CUDA_LOCAL=1   # This environment variable removes pip installed nvcc from build requirements
pip install .
```
### SYSTEM CUDA setup:
I assume that you have already installed or loaded some CUDA version. E.g. check with
```bash
nvcc --version
```
Install:
```bash
pip install --upgrade "jax[cuda13-local]"   # replace CUDA version as appropriate
export CUDA_LOCAL=1   # This environment variable removes pip installed nvcc from build requirements
pip install .
```
### Editable installation
If you want to edit source files, it is recommended that you install like this:
```bash
pip install -e . --no-build-isolation
```
Note that the editable installation will be rebuild automatically on module load, so that it is not necessary to invoke install again whenever a .cu file changes (it will take a couple of seconds though). However, for this to work properly I found it necessary to use --no-build-isolation. For the "--no-build-isolation" flag to work, it is necessary that the build dependencies are installed separately. You should be able to achieve this by installing these in advance, e.g.
```bash
pip install nvidia-cuda-cccl>=13.0.0 scikit-build-core nanobind jax[cuda13]
```
or similar, depending on your setup.

### uv
If you want to reconstruct the exact virtual environment that the code was developed in, you can use
```
uv sync
source .venv/bin/activate   # optionally activate the created environment
```

# Useufl links
* z-order with floats: http://compgeom.com/~piyush/papers/tvcg_stann.pdf

# Other notes:
* Probably I can remove dependence on thrust when deleting deprecated code


# Useful commands
#uv python install 3.12
#uv python pin 3.12
#uv add --dev ipykernel
uv pip install -e . --no-build-isolation
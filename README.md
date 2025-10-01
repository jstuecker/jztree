# CustomJax
Some c++ cuda kernels that expose new functionality to jax via the foreign function interface (FFI)

# Installation
For now we require installation with the jax[cuda13] pacakge. This is necessary, because NVIDIA
only packages the nvcc compiler with its python packages since cuda13. In principle an installation
with older CUDA version would be possible if the CUDA compiler is installed elswhere, independently
of the pip eco-system. (For example with a system installed CUDA or with an anaconda managed CUDA. We may support this at a later point.) 
Since, we can only easily guarantee a correct installation with the python managed CUDA13, we will
stick with this for now. Since CUDA13 is very new, it may require updating the GPU drivers and it
is only supported on [Graphics cards with compute capability](https://developer.nvidia.com/cuda-gpus?utm_source=chatgpt.com)  >=7.5

### Prequisites: 
Check that the NVIDIA "Driver Version" is new enough [>= 580](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). Note that CUDA 13 is not supported on older GPUs
```bash
nvidia-smi
```
### pip installation
```bash
pip install .
```
To check whether the installation was successful, run
```
python src/benchmarks/hello_world.py
```
(Note: Installation speed my be significantly higher with [uv pip](https://docs.astral.sh/uv/) )
### Editable installation
If you want to edit source files, it is recommended that you install like this:
```bash
pip install -e . --no-build-isolation
```
(--no-build-isolation helps with compile speed and it also helps with proper code highlighting
e.g. in VSCode, since it creates "build/compile_commands.json".)
Additionally uncomment this line in pyproject.toml
```python
#editable.rebuild = true
```
It will ensure that the .cu files will be recompiled whenever they are imported after modifications.

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
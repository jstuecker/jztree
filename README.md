# CustomJax
Some c++ cuda kernels that expose new functionality to jax via the foreign function interface (FFI)

# Installation
```bash
pip install
```

Editable installation with 
```bash
pip install -e .
```
will only work if hard-coded paths to .so files are adapted. So far I didn't figure out how to make the build system put the libraries in a folder that can be found by the .py files...
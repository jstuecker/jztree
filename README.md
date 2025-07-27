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
will be in auto-rebuild mode. This means that libraries will be recompiled (if necessary) on import. Therfore, you don't need to do 'pip install' again when changing .cu files.

Additionally to properly make VSCode find your paths (and also save build time) you can use
```bash
pip install -e . --no-build-isolation
```
which will properly write the used paths into 'build/compile_commands.json' so that VSCode understands them. (With build isolation the include paths may be wrong, pointing to temporary directories.) Only use this during development!
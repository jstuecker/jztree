# Changelog

## Version 1.1.0 (unreleased)

* Added `TreeHierarchy.poslvl()` for accessing node positions and levels.
* Added separate query/source indexing for interaction lists, including updated
  CUDA bindings. The Python interface and CUDA backend must be upgraded together.
* Improved shard-map context detection, including single-device mapped contexts,
  and fixed multi-GPU edge cases.
* Fixed integer dtype consistency with JAX x64 enabled and added validation and tests.
* Explicitly blocked gradients through tree construction.
* Fixed mass-centering for dimensions other than three.
* Extended initial-condition helpers, statistics, and array utilities.
* Configurations now use slots: assigning an unknown parameter raises an error
  instead of silently adding an unused attribute. Existing parameters remain mutable.

## Version 1.0.3
* Updated .vma access to support jax 0.10
* Fixed issue with CUDA_MAJOR version detection for source builds
* Added source links to documentation

## Version 1.0.2
Initial stable release

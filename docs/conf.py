import sys
import os
import types
import contextlib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path('..', 'src').resolve()))

# Build docs without requiring JAX/CUDA runtime packages.
os.environ.setdefault("JZTREE_SKIP_JAX_CUDA_CHECK", "1")


def _identity_decorator(obj=None, **_kwargs):
	if obj is None:
		return lambda x: x
	return obj


def _install_jax_stub():
	"""Install a tiny jax shim so autodoc can import modules without jax/cuda."""
	if "jax" in sys.modules:
		return

	class _Dummy:
		def __getattr__(self, _name):
			return self

		def __call__(self, *args, **kwargs):
			return self

		def __iter__(self):
			return iter(())

		def __bool__(self):
			return False

		def __getitem__(self, _key):
			return self

	jax_mod = types.ModuleType("jax")
	jax_mod.Array = object
	jax_mod.jax = jax_mod
	jax_mod.numpy = np

	tree_util = types.ModuleType("jax.tree_util")
	tree_util.register_dataclass = _identity_decorator
	tree_util.tree_map = lambda fn, tree: tree
	tree_util.tree_flatten = lambda tree: ([], None)
	tree_util.tree_unflatten = lambda _def, leaves: leaves
	tree_util.tree_leaves = lambda _tree: []

	sharding = types.ModuleType("jax.sharding")

	class PartitionSpec(tuple):
		def __new__(cls, *parts):
			return super().__new__(cls, parts)

	sharding.PartitionSpec = PartitionSpec

	experimental = types.ModuleType("jax.experimental")
	experimental.io_callback = lambda f, *_args, **_kwargs: f()

	multihost_utils = types.ModuleType("jax.experimental.multihost_utils")
	multihost_utils.process_allgather = lambda x, *args, **kwargs: x
	experimental.multihost_utils = multihost_utils

	ffi = types.SimpleNamespace(
		register_ffi_target=lambda *args, **kwargs: None,
		ffi_call=lambda *args, **kwargs: (lambda *a, **k: ()),
	)

	jax_mod.tree_util = tree_util
	jax_mod.tree = types.SimpleNamespace(map=lambda fn, tree: tree)
	jax_mod.sharding = sharding
	jax_mod.experimental = experimental
	jax_mod.ffi = ffi
	jax_mod.jit = _identity_decorator
	jax_mod.devices = lambda *_args, **_kwargs: []
	jax_mod.enable_x64 = contextlib.nullcontext
	jax_mod.debug = types.SimpleNamespace(print=lambda *args, **kwargs: None)
	jax_mod.lax = _Dummy()

	sys.modules["jax"] = jax_mod
	sys.modules["jax.numpy"] = np
	sys.modules["jax.tree_util"] = tree_util
	sys.modules["jax.sharding"] = sharding
	sys.modules["jax.experimental"] = experimental
	sys.modules["jax.experimental.multihost_utils"] = multihost_utils


_install_jax_stub()

# requires:
# pip install sphinx-paramlinks

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'jz-tree'
copyright = '2026, Jens Stücker'
author = 'Jens Stücker'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'sphinx.ext.autodoc', 'sphinx_paramlinks', 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_preserve_defaults = True
# python_maximum_signature_line_length = 60
autodoc_typehints = "description"
# autodoc_member_order = "bysource"

# Mock heavy optional dependencies so API docs can be built in a minimal env.
autodoc_mock_imports = [
	"jztree_cuda",
	"jztree_cuda.ffi_fof",
	"jztree_cuda.ffi_knn",
	"jztree_cuda.ffi_sort",
	"jztree_cuda.ffi_tools",
	"jztree_cuda.ffi_tree",
	"jztree.stats",
	"nvidia",
	"nvidia.cuda_nvcc",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

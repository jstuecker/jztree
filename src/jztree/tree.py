import numpy as np
import jax
import jax.numpy as jnp

from jztree_cuda import ffi_tree

from fmdj.ztree import pos_zorder_sort, search_sorted_z
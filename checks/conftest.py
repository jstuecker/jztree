import jax
import jax.numpy as jnp
import pytest
from jztree.tree import pos_zorder_sort

@pytest.fixture
def npart(request):    
    return getattr(request, "param", 1024*1024)

@pytest.fixture
def pos(npart):
    # pos = jax.random.normal(jax.random.PRNGKey(0), (npart,3), dtype=jnp.float32)
    pos = jax.random.uniform(jax.random.PRNGKey(0), (npart,3), dtype=jnp.float32)
    return jax.block_until_ready(pos)

@pytest.fixture
def pos_z(pos):
    posz, isort = pos_zorder_sort(pos)
    return posz
import jax
import jax.numpy as jnp
import pytest
from jztree.tree import pos_zorder_sort
import sys
import os

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

def _silence_process_output() -> None:
    """
    Redirect stdout/stderr to /dev/null at the OS FD level.
    Also rebind sys.stdout/sys.stderr to avoid some Python-level oddities.
    """
    # Make Python less likely to buffer weirdly
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    # Redirect low-level file descriptors (covers most output sources)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)  # stdout
    os.dup2(devnull_fd, 2)  # stderr
    os.close(devnull_fd)

    # Rebind Python-level streams (some libs write to these objects directly)
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


def pytest_configure(config):
    try:
        jax.distributed.initialize()
    except (ValueError, RuntimeError) as err:
        print(f"Distributed mode not available ({err})")

    if jax.process_index() != 0:
        _silence_process_output()
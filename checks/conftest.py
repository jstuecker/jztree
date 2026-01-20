import jax
import jax.numpy as jnp
import pytest
from jztree.tree import pos_zorder_sort
from fmdj.comm import should_init_jax_distributed
import sys
import os

# ------------------------------------------------------------------------------------------------ #
#                                         Configure pytest                                         #
# ------------------------------------------------------------------------------------------------ #

def pytest_addoption(parser):
    parser.addoption( "--quick", action="store_true", default=False,
        help="Quick mode: deselect slow tests and reduce parametrized tests to first case.",
    )

def pytest_runtest_setup(item):
    if item.config.getoption("--quick"):
        if item.get_closest_marker("skip_in_quick") or item.get_closest_marker("slow"):
            pytest.skip("Skipped in --quick mode")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: skip the whole test in --quick mode")
    config.addinivalue_line("markers", "skip_in_quick: skip the whole test in --quick mode")
    config.addinivalue_line( "markers",
        "shrink_in_quick(keep_index=0): in --quick mode, keep only the parametrized at index",
    )
    # Your existing setup:
    if should_init_jax_distributed():
        jax.distributed.initialize()
    else:
        print("Using single-GPU mode")

    if jax.process_index() != 0:
        _silence_process_output()

def _keep_index_from_marker(item) -> int | None:
    """
    Returns the per-test keep index from @pytest.mark.shrink_in_quick(index=...)
    (or @pytest.mark.shrink_in_quick(<int>)), or None if marker not present.
    """
    m = item.get_closest_marker("shrink_in_quick")
    if m is None:
        return None

    if "keep_index" in m.kwargs:
        return int(m.kwargs["keep_index"])
    if m.args:
        return int(m.args[0])
    return 0

def pytest_runtest_setup(item):
    if not item.config.getoption("--quick"):
        return

    # Skip whole tests marked slow
    if item.get_closest_marker("slow") or item.get_closest_marker("skip_in_quick"):
        pytest.skip("Skipped in --quick mode (slow / skip_in_quick)")

    # Shrink parametrized tests that opt in
    keep = _keep_index_from_marker(item)
    if keep is None:
        return

    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return  # not parametrized

    indices = getattr(callspec, "indices", {}) or {}
    if indices and any(i != keep for i in indices.values()):
        pytest.skip(f"Skipped in --quick mode (keep={keep})")

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

# ------------------------------------------------------------------------------------------------ #
#                                             Fixtures                                             #
# ------------------------------------------------------------------------------------------------ #

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

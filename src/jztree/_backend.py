"""Backend loader and CUDA compatibility checks for jztree."""

from __future__ import annotations

import importlib.metadata
import os
import re
from typing import Optional


def _infer_jax_cuda_major() -> Optional[int]:
    """Best-effort inference of JAX CUDA major version.

    Returns None if JAX is missing or no reliable CUDA major can be determined.
    """
    try:
        # Prefer explicit CUDA plugin package markers when available.
        importlib.metadata.version("jax-cuda13-plugin")
        return 13
    except importlib.metadata.PackageNotFoundError:
        pass

    try:
        importlib.metadata.version("jax-cuda12-plugin")
        return 12
    except importlib.metadata.PackageNotFoundError:
        pass

    try:
        import jax
    except Exception:
        return None

    try:
        gpu_devices = jax.devices("gpu")
    except Exception:
        gpu_devices = []

    if not gpu_devices:
        return None

    # Typical platform_version strings include tokens like "CUDA 13.0.1".
    platform_version = str(getattr(gpu_devices[0], "platform_version", ""))
    match = re.search(r"CUDA\s*([0-9]+)", platform_version, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def load_backend():
    """Import jztree_cuda and validate CUDA major compatibility with JAX."""
    skip_check = os.environ.get("JZTREE_SKIP_JAX_CUDA_CHECK", "").strip().lower()
    skip_check = skip_check in {"1", "true", "yes", "on"}

    try:
        import jztree_cuda
    except ImportError as exc:
        raise ImportError(
            "jztree backend package 'jztree_cuda' is not installed. "
            "Install one backend wheel, e.g. 'jztree-cu12' or 'jztree-cu13'."
        ) from exc

    # Used by docs or lightweight environments to bypass backend compatibility checks.
    if skip_check:
        return jztree_cuda

    backend_cuda_major = getattr(jztree_cuda, "CUDA_MAJOR", None)
    if backend_cuda_major is None:
        raise ImportError(
            "Installed jztree backend does not expose CUDA_MAJOR. "
            "Please reinstall a supported backend wheel (jztree-cu12 or jztree-cu13)."
        )

    try:
        backend_cuda_major = int(backend_cuda_major)
    except Exception as exc:
        raise ImportError(
            f"Installed jztree backend has invalid CUDA_MAJOR={backend_cuda_major!r}. "
            "Please reinstall a supported backend wheel (jztree-cu12 or jztree-cu13)."
        ) from exc

    if not skip_check:
        jax_cuda_major = _infer_jax_cuda_major()
        if jax_cuda_major is not None and jax_cuda_major != backend_cuda_major:
            raise ImportError(
                "Installed jztree backend is CUDA "
                f"{backend_cuda_major}, but JAX appears to use CUDA {jax_cuda_major}. "
                f"Please install jztree-cu{jax_cuda_major}."
            )

    return jztree_cuda

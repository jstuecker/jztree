#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# Reuse the cu13 image (already has Python + uv). Build it if not already present.
IMAGE_NAME="jztree-cu13-wheel-builder"
UV_VERSION="0.7.2"

docker build --build-arg UV_VERSION="$UV_VERSION" -t "$IMAGE_NAME" "$SCRIPT_DIR"

# packaging/jztree is a pure-Python meta-package (py3-none-any), so no CUDA
# compilation happens. Build once with any Python version.
docker_args=(
  --rm
  -t
  -v "$REPO_ROOT:/workspace"
  -e REPO_ROOT=/workspace
  -e PACKAGE_DIR="packaging/jztree"
  -e CUDA_ARCHS=""
  -e AUDITWHEEL_PLAT="manylinux_2_28_x86_64"
  -e OUTPUT_DIR="packaging/docker-wheel-builder-main/output"
  -e COPY_TO_PACKAGE_DIST=1
  -e PYTHON_VERSIONS_CSV="3.11"
  -e PIP_BUILD_DEPS_CSV="build,setuptools>=69,wheel"
)

docker run "${docker_args[@]}" "$IMAGE_NAME" /builder/build-in-container.sh

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
source "$SCRIPT_DIR/config.sh"

PYTHON_VERSIONS_CSV=$(IFS=,; echo "${PYTHON_VERSIONS[*]}")

docker build \
  --build-arg UV_VERSION="$UV_VERSION" \
  -t "$IMAGE_NAME" \
  "$SCRIPT_DIR"

docker run --rm -t \
  -v "$REPO_ROOT:/workspace" \
  -e REPO_ROOT=/workspace \
  -e PACKAGE_DIR="$PACKAGE_DIR" \
  -e CUDA_ARCHS="$CUDA_ARCHS" \
  -e AUDITWHEEL_PLAT="$AUDITWHEEL_PLAT" \
  -e OUTPUT_DIR="$OUTPUT_DIR" \
  -e COPY_TO_PACKAGE_DIST="$COPY_TO_PACKAGE_DIST" \
  -e PYTHON_VERSIONS_CSV="$PYTHON_VERSIONS_CSV" \
  -e MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" \
  "$IMAGE_NAME" \
  /builder/build-in-container.sh

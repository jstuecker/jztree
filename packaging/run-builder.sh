#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?Expected main, cu12, or cu13}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
case "$MODE" in
  main|cu13)
    BUILDER_DIR="$SCRIPT_DIR/docker-wheel-builder"
    source "$BUILDER_DIR/config.sh"
    ;;
  cu12)
    BUILDER_DIR="$SCRIPT_DIR/docker-wheel-builder-cu12"
    source "$BUILDER_DIR/config.sh"
    ;;
  *) echo "Expected main, cu12, or cu13" >&2; exit 2 ;;
esac
if [[ "$MODE" == main ]]; then
  PYTHON_VERSIONS=("3.11")
  OUTPUT_DIR="packaging/output/main"
fi
PYTHON_VERSIONS_CSV=${BUILD_PYTHONS:-$(IFS=,; echo "${PYTHON_VERSIONS[*]}")}
BUILD_JOBS=${BUILD_JOBS:-4}
if [[ ! "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "BUILD_JOBS must be a positive integer" >&2; exit 2
fi
BUILD_JAX_VERSION=${BUILD_JAX_VERSION:-0.8.3}
# Reuse the built image on subsequent invocations. Pass REBUILD_IMAGE=1 when
# deliberately updating the underlying container/toolchain; its ID keys the cache.
if [[ "${REBUILD_IMAGE:-0}" == 1 ]] || ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  docker build --build-arg UV_VERSION="$UV_VERSION" -t "$IMAGE_NAME" "$BUILDER_DIR"
fi
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE_NAME")
mkdir -p "$REPO_ROOT/$OUTPUT_DIR/logs" "$REPO_ROOT/packaging/.build-cache"
# micromamba also records environments in ~/.conda, independently of its caches.
# The host UID has no home entry inside the image, so this resolves to /.conda.
mkdir -p "$REPO_ROOT/packaging/.build-cache/conda-user"
LOG_FILE="$REPO_ROOT/$OUTPUT_DIR/logs/run-$(date -u +%Y%m%dT%H%M%S)-$$.log"
echo "Building $MODE; Python $PYTHON_VERSIONS_CSV; $BUILD_JOBS parallel compiler jobs"
echo "Log: $LOG_FILE"
docker_args=(--rm --init
  -v "$REPO_ROOT:/workspace"
  -v "$REPO_ROOT/packaging/.build-cache/conda-user:/.conda"
  -e REPO_ROOT=/workspace
  -e BUILD_MODE="$MODE"
  -e BUILDER_IMAGE_ID="$IMAGE_ID"
  -e OUTPUT_DIR="$OUTPUT_DIR"
  -e PYTHON_VERSIONS_CSV="$PYTHON_VERSIONS_CSV"
  -e BUILD_JAX_VERSION="$BUILD_JAX_VERSION"
  -e CUDAARCHS="$CUDA_ARCHS"
  -e AUDITWHEEL_PLAT="$AUDITWHEEL_PLAT"
  -e CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
  -e UV_CACHE_DIR=/workspace/packaging/.build-cache/uv
  -e MAMBA_ROOT_PREFIX=/workspace/packaging/.build-cache/micromamba
  -e CONDA_PKGS_DIRS=/workspace/packaging/.build-cache/micromamba/pkgs
  -e XDG_CACHE_HOME=/workspace/packaging/.build-cache/xdg
  -e PYTHONUNBUFFERED=1
)
# Use the host UID so persistent output files remain manageable from the terminal.
# The container's compiler/dependencies are readable by unprivileged users.
docker_args+=(--user "$(id -u):$(id -g)")
docker run "${docker_args[@]}" "$IMAGE_ID" \
  /opt/python/cp311-cp311/bin/python /workspace/packaging/wheel_builder.py 2>&1 | tee "$LOG_FILE"

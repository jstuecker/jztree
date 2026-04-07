#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace}
PACKAGE_DIR=${PACKAGE_DIR:-packaging/jztree-cu12}
CUDA_ARCHS=${CUDA_ARCHS:-all}
AUDITWHEEL_PLAT=${AUDITWHEEL_PLAT:-manylinux_2_17_x86_64}
OUTPUT_DIR=${OUTPUT_DIR:-packaging/docker-wheel-builder-cu12/output}
COPY_TO_PACKAGE_DIST=${COPY_TO_PACKAGE_DIST:-1}
PYTHON_VERSIONS_CSV=${PYTHON_VERSIONS_CSV:-3.11,3.12,3.13,3.14}
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/opt/micromamba}

IFS=',' read -r -a PYTHON_VERSIONS <<< "$PYTHON_VERSIONS_CSV"

RAW_DIR="${REPO_ROOT}/${OUTPUT_DIR}/raw"
WHEELHOUSE_DIR="${REPO_ROOT}/${OUTPUT_DIR}/wheelhouse"
LOG_DIR="${REPO_ROOT}/${OUTPUT_DIR}/logs"
PACKAGE_DIST_DIR="${REPO_ROOT}/${PACKAGE_DIR}/dist"

mkdir -p "$RAW_DIR" "$WHEELHOUSE_DIR" "$LOG_DIR" "$PACKAGE_DIST_DIR" "$MAMBA_ROOT_PREFIX"

for py_version in "${PYTHON_VERSIONS[@]}"; do
  py_nodot=${py_version//./}
  py_tag="cp${py_nodot}-cp${py_nodot}"
  env_name="jztree-cu12-py${py_nodot}"
  env_prefix="${MAMBA_ROOT_PREFIX}/envs/${env_name}"
  raw_out_dir="${RAW_DIR}/${py_tag}"
  log_file="${LOG_DIR}/${py_tag}.log"

  rm -rf "$raw_out_dir"
  mkdir -p "$raw_out_dir"

  echo "=== Building CUDA12 for Python ${py_version} (${py_tag}) ==="
  micromamba create -y -n "$env_name" -c conda-forge \
    "python=${py_version}" \
    pip \
    cuda-nvcc \
    cuda-version=12 \
    cudnn \
    nccl \
    libcufft \
    cuda-cupti \
    libcublas \
    libcusparse \
    openblas \
    libblas \
    liblapack \
    scipy \
    numpy

  py_bin="${env_prefix}/bin/python"
  if [[ ! -x "$py_bin" ]]; then
    echo "Python interpreter not found in env: $py_bin" >&2
    exit 1
  fi

  # Use the manylinux host toolchain to keep symbol versions compatible with
  # manylinux_2_17. Avoid conda compilers here, as they may produce newer
  # GLIBC/GLIBCXX symbol requirements.
  if [[ -x "/opt/rh/devtoolset-10/root/usr/bin/gcc" && -x "/opt/rh/devtoolset-10/root/usr/bin/g++" ]]; then
    export CC="/opt/rh/devtoolset-10/root/usr/bin/gcc"
    export CXX="/opt/rh/devtoolset-10/root/usr/bin/g++"
  else
    export CC="$(command -v gcc)"
    export CXX="$(command -v g++)"
  fi
  export CUDAHOSTCXX="${CXX}"

  uv pip install --python "$py_bin" --upgrade \
    build \
    "jax[cuda12-local]" \
    "scikit-build-core>=0.11" \
    "nanobind>=2.9.2" \
    "cmake>=3.24" \
    auditwheel

  "$py_bin" - <<'PY'
import sys
try:
    import jax  # noqa: F401
    import jaxlib  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "Failed to import jax/jaxlib after installing jax[cuda12-local]. "
        "This Python version may not be supported for CUDA12 wheels yet. "
        f"Details: {exc}"
    )
PY

  (
    cd "$REPO_ROOT"
    CUDAARCHS="$CUDA_ARCHS" \
      uv build --wheel --no-build-isolation --python "$py_bin" \
      -o "$raw_out_dir" "$PACKAGE_DIR"
  ) 2>&1 | tee "$log_file"

  shopt -s nullglob
  wheels=("${raw_out_dir}"/*.whl)
  shopt -u nullglob
  if [[ ${#wheels[@]} -ne 1 ]]; then
    echo "Expected exactly one wheel in ${raw_out_dir}, found ${#wheels[@]}" >&2
    exit 1
  fi

  "$py_bin" -m auditwheel repair \
    --plat "$AUDITWHEEL_PLAT" \
    -w "$WHEELHOUSE_DIR" \
    "${wheels[0]}"

  if [[ "$COPY_TO_PACKAGE_DIST" == "1" ]]; then
    cp -f "$WHEELHOUSE_DIR"/*.whl "$PACKAGE_DIST_DIR/"
  fi
done

echo "Built wheels are in: ${WHEELHOUSE_DIR}"
if [[ "$COPY_TO_PACKAGE_DIST" == "1" ]]; then
  echo "Copied repaired wheels to: ${PACKAGE_DIST_DIR}"
fi

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace}
PACKAGE_DIR=${PACKAGE_DIR:-packaging/jztree-cu13}
CUDA_ARCHS=${CUDA_ARCHS:-87}
AUDITWHEEL_PLAT=${AUDITWHEEL_PLAT:-manylinux_2_28_x86_64}
OUTPUT_DIR=${OUTPUT_DIR:-packaging/docker-wheel-builder/output}
COPY_TO_PACKAGE_DIST=${COPY_TO_PACKAGE_DIST:-1}
PYTHON_VERSIONS_CSV=${PYTHON_VERSIONS_CSV:-3.12,3.13}
PIP_BUILD_DEPS_CSV=${PIP_BUILD_DEPS_CSV:-build,jax[cuda13],scikit-build-core>=0.11,nanobind>=2.9.2,cmake>=3.24,auditwheel}

IFS=',' read -r -a PYTHON_VERSIONS <<< "$PYTHON_VERSIONS_CSV"
IFS=',' read -r -a PIP_BUILD_DEPS <<< "$PIP_BUILD_DEPS_CSV"

RAW_DIR="${REPO_ROOT}/${OUTPUT_DIR}/raw"
WHEELHOUSE_DIR="${REPO_ROOT}/${OUTPUT_DIR}/wheelhouse"
LOG_DIR="${REPO_ROOT}/${OUTPUT_DIR}/logs"
PACKAGE_DIST_DIR="${REPO_ROOT}/${PACKAGE_DIR}/dist"

mkdir -p "$RAW_DIR" "$WHEELHOUSE_DIR" "$LOG_DIR" "$PACKAGE_DIST_DIR"

for py_version in "${PYTHON_VERSIONS[@]}"; do
  py_nodot=${py_version//./}
  py_tag="cp${py_nodot}-cp${py_nodot}"
  py_bin="/opt/python/${py_tag}/bin/python"
  if [[ ! -x "$py_bin" ]]; then
    echo "Python interpreter not found in image: $py_bin" >&2
    exit 1
  fi

  venv_dir="/tmp/jztree-${py_tag}"
  raw_out_dir="${RAW_DIR}/${py_tag}"
  log_file="${LOG_DIR}/${py_tag}.log"

  rm -rf "$venv_dir" "$raw_out_dir"
  mkdir -p "$raw_out_dir"

  echo "=== Building for Python ${py_version} (${py_tag}) ==="
  uv venv --python "$py_bin" "$venv_dir"
  source "$venv_dir/bin/activate"

  uv pip install --upgrade "${PIP_BUILD_DEPS[@]}"

  (
    cd "$REPO_ROOT"
    CUDAARCHS="$CUDA_ARCHS" \
      uv build --wheel --no-build-isolation --python "$venv_dir/bin/python" \
      -o "$raw_out_dir" "$PACKAGE_DIR"
  ) 2>&1 | tee "$log_file"

  shopt -s nullglob
  wheels=("${raw_out_dir}"/*.whl)
  shopt -u nullglob
  if [[ ${#wheels[@]} -ne 1 ]]; then
    echo "Expected exactly one wheel in ${raw_out_dir}, found ${#wheels[@]}" >&2
    exit 1
  fi

  wheel_file="${wheels[0]}"
  wheel_name="$(basename "$wheel_file")"

  # Pure-Python wheels are platform-independent and do not need auditwheel.
  if [[ "$wheel_name" == *-none-any.whl ]]; then
    cp -f "$wheel_file" "$WHEELHOUSE_DIR/"
  else
    "$venv_dir/bin/python" -m auditwheel repair \
      --plat "$AUDITWHEEL_PLAT" \
      -w "$WHEELHOUSE_DIR" \
      "$wheel_file"
  fi

  if [[ "$COPY_TO_PACKAGE_DIST" == "1" ]]; then
    cp -f "$WHEELHOUSE_DIR"/*.whl "$PACKAGE_DIST_DIR/"
  fi

done

echo "Built wheels are in: ${WHEELHOUSE_DIR}"
if [[ "$COPY_TO_PACKAGE_DIST" == "1" ]]; then
  echo "Copied repaired wheels to: ${PACKAGE_DIST_DIR}"
fi

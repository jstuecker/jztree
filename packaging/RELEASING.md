# Preparing jz-tree 1.1.0

Status: prepared, not built or published. Wait for the cluster investigation of
JAX 0.11.1 before building binaries or finalizing JAX compatibility guidance.
The current JAX dependency bounds and builder dependency selections are unchanged.

## Packages and versions

Release the Python interface `jztree` and both backends, `jztree-cu12` and
`jztree-cu13`, as 1.1.0. The interface's CUDA extras pin the matching backend.
The CUDA FFI changed, so do not reuse the 1.0.1 backend wheels.

The root `pyproject.toml` intentionally keeps its SCM-derived version for source
builds. At release time, remove the unreleased label
in the changelog, commit the approved release changes, and tag that commit
`v1.1.0` before producing release artifacts. Do not manually edit the generated
`src/jztree/_version.py`.

## Before building

- Resolve the cluster JAX compatibility investigation. Record the validated JAX,
  jaxlib, CUDA and driver versions and any required workaround in the docs.
  Decide whether dependency bounds or builder pins need adjustment; the builders
  currently resolve JAX dynamically, so their resolved versions must be checked.
- Review the changelog and confirm matching versions in all three packaging
  `pyproject.toml` files and both CUDA extras.
- Run tests from `checks` with `pytest tests --quick`, and run the relevant x64
  and distributed checks on the cluster. Repeat checks against the built wheels
  in clean environments before publication, using one backend per environment.
- Keep older artifacts out of the upload selection. Existing `dist` and
  `wheelhouse` directories may contain older releases; the builders do not clear
  all of them. Preserve old artifacts, but stage only the new 1.1.0 wheels for
  validation and upload.

## Build commands (only after the hold is lifted)

From the repository root, the existing Docker builders are:

```bash
./packaging/run-wheel-builds.sh main
./packaging/run-wheel-builds.sh cu13
./packaging/run-wheel-builds.sh cu12
```

`./packaging/run-wheel-builds.sh all` runs these sequentially. The configured
matrix is one pure-Python wheel, four CUDA 13 wheels (Python 3.11–3.14), and
three CUDA 12 wheels (Python 3.11–3.13), with `CUDA_ARCHS="all"` for the backends.
CUDA 12 obtains its compiler through conda; CUDA 13 uses PyPI build dependencies.

Build logs and repaired wheels are under each builder's `output/logs` and
`output/wheelhouse`; repaired wheels are also copied into each package's `dist`.
Validate wheel metadata, matching backend dependencies, imports, CUDA-major
detection, tests and platform tags before uploading only the approved artifacts.

## Expected build time

Existing successful logs from 6 April 2026 give these configure/build/install
intervals (first to last timestamp in each log):

| Backend | Python versions | Minutes per wheel | Total |
| --- | --- | --- | --- |
| CUDA 13 | 3.11, 3.12, 3.13, 3.14 | 17.55, 17.41, 17.59, 15.36 | 67.9 min |
| CUDA 12 | 3.11, 3.12, 3.13 | 16.70, 16.43, 16.92 | 50.1 min |

These logs are in `packaging/docker-wheel-builder/output/logs` and
`packaging/docker-wheel-builder-cu12/output/logs`. Their approximately 118-minute
sum excludes Docker image builds, dependency/environment setup, final wheel
assembly and auditwheel repair. Budget roughly 2–3 hours for a sequential full
matrix on a comparable setup, potentially longer with cold caches or changed
toolchains. This is a historical estimate, not a measurement of 1.1.0.

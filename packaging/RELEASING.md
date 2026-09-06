# Releasing jz-tree 1.1.0

The cluster JAX investigation is complete: the selected four-A100 FMM, gradient, simulation, kNN, and FoF checks passed with JAX 0.11.1 using the documented workarounds. This validates that tested source setup, not yet the new release wheels. No release wheels have been built or published as part of preparation.

## Before building

- Review and commit the release changes. The Python interface and both backends declare version 1.1.0; the interface extras pin matching backends. Do not reuse older backend wheels because the CUDA FFI changed.
- Avoid modifying build inputs while a build runs. The build cache is keyed by source contents, package/build settings, Python version, and Docker image ID. Source changes select a fresh build directory; merely committing unchanged files does not invalidate it.
- Install Docker and ensure your user can run it. The scripts need internet access for the first image/environment setup. The build uses containers, not your active Python environment, and needs no GPU.
- The CUDA 12 builder now uses manylinux_2_28, like CUDA 13. Both release wheel families require Linux x86-64 with glibc >=2.28. This avoids trying to install the pinned JAX wheels (which require glibc >=2.27) in the previous glibc 2.17 container.

## Build in your terminal

From the repository root, a useful first check is one Python 3.12 CUDA 13 wheel:

```bash
BUILD_PYTHONS=3.12 BUILD_JOBS=4 ./packaging/run-wheel-builds.sh cu13
```

Then build the complete matrix:

```bash
BUILD_JOBS=4 ./packaging/run-wheel-builds.sh all
```

This builds one pure-Python wheel, four CUDA 13 wheels (Python 3.11–3.14), and three CUDA 12 wheels (Python 3.11–3.13). The already completed 3.12 wheel is verified and skipped. You can instead run `main`, `cu13`, and `cu12` separately. `BUILD_PYTHONS` is an optional comma-separated subset; omit it for the complete matrix.

`BUILD_JOBS` controls parallel compiler jobs (default 4). Reduce it to 2 or 1 if compilation exhausts memory. It does not change the resulting GPU architecture coverage: both backend configurations retain `CUDA_ARCHS="all"`. Run only one builder at a time per checkout; an exclusive lock prevents concurrent writers.

## Interruptions and progress

All build output is streamed to the terminal and saved to a timestamped file under `packaging/output/<mode>/logs`, where mode is `main`, `cu12`, or `cu13`. Each wheel prints BUILD, DONE (with elapsed time), or SKIP.

After Ctrl-C, a reboot, or a failed build, rerun the same command. Completed wheels are skipped only when their completion records and SHA-256 checksums agree. Completed raw wheels are reused if interruption occurred during wheel repair. An interrupted compilation reuses its stable CMake build directory and completed object files; an individual interrupted compiler invocation may need to start again.

Environments, CUDA 12 conda packages, and uv downloads persist under `packaging/.build-cache/`. Build directories and wheels persist under `packaging/output/<mode>/`. Do not delete these directories if you want to resume. Previous artifacts are preserved, not overwritten by a different source/settings fingerprint. The build runs as your host UID so newly created files remain manageable from your terminal. The new output location also avoids writing into the older, container-owned output directories.

Docker images are reused by default. `REBUILD_IMAGE=1` deliberately rebuilds the image; a changed image ID creates a new environment/build cache. The base images and transitive dependencies are not fully locked, so a fresh setup is not claimed to be bit-for-bit reproducible.

## Build-time dependencies

The builders pin JAX, jaxlib, and the selected CUDA plugin/PJRT packages to **0.8.3**, a tested older runtime baseline with wheels for this Python matrix. They also pin scikit-build-core 0.12.2, nanobind 2.9.2, and CMake 4.3.1. CUDA 12 obtains a 12.9 compiler/toolkit through conda; CUDA 13 uses the dependencies selected by `jax[cuda13]==0.8.3`. These are build-environment choices, not new runtime constraints in `pyproject.toml`.

Each prepared environment records its full resolved pip dependencies in `requirements-resolved.txt`; CUDA 12 also records `conda-explicit.txt`. The `inputs.json` beside each wheel records the source fingerprint, image ID, architectures, Python version, and requested requirements. `BUILD_JAX_VERSION` can override the default for a deliberate compatibility experiment; it selects a separate cache and requires renewed testing.

## Locate and validate artifacts

Each builder writes `packaging/output/<mode>/latest.json` listing the exact wheels and SHA-256 checksums from its latest successful invocation. A subset run lists only that subset: run the full matrix before release. Wheels live in fingerprint-specific `packaging/output/<mode>/<key>/cpXXX/wheelhouse/` directories. Nothing is automatically copied into the old package `dist` directories, which may contain stale releases.

Only stage the artifacts listed in the three current manifests for publication. Verify all manifests correspond to the same source fingerprint and that the expected eight wheels are present. Never upload a broad glob over old `dist` or `output` directories.

- Inspect metadata and platform tags, and run `twine check` on the selected wheels.
- Test installed wheels in clean environments, using only one CUDA backend per environment, and run tests from `checks` with `pytest tests --quick`.
- Include x64 and distributed checks on representative CUDA 12 and CUDA 13 setups. Test the same binaries with the build baseline and newer intended runtime JAX versions, including 0.11.1 and the cluster workarounds. The advertised `jax>=0.8.0` minimum still needs validation against these binaries or adjustment before publication.
- Check imports/basic execution for each supported Python version. The compact cluster source tests do not replace wheel testing.

The build-runner's CPU-only restart tests can be run from `checks` with `python -m unittest discover -s ../packaging/tests -v`; they use synthetic files and mocked commands, not CUDA compilation.

## Tag and publish after validation

Tag the tested release commit `v1.1.0` and push that tag once the wheels pass. The root `pyproject.toml` intentionally retains SCM-derived versioning; the tag makes source builds report 1.1.0. The three packaging projects use explicit versions and can build before the tag. Do not manually edit generated `src/jztree/_version.py`.

Upload the validated CUDA backend distributions first, then the matching Python interface package. Verify fresh PyPI installs and publish the GitHub release notes. No script here tags, commits, or uploads automatically.

## Historical build-time baseline

Successful logs from 6 April 2026 show approximately 68 minutes for four CUDA 13 wheels and 50 minutes for three CUDA 12 wheels: 15–18 minutes per wheel, or about 118 minutes in total, excluding image/environment setup and wheel repair. Those logs appear to show serial Make compilation. Allow 2–3 hours as an initial planning estimate; parallel compilation should help, but the new 1.1.0 build times have not yet been measured.

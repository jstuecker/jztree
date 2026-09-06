"""Resumable wheel build driver, run inside the manylinux containers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile


def run(*args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(root):
    """Hash build inputs, including untracked source files, but not build outputs."""
    paths = set()
    for name in ('src', 'packaging'):
        for path in (root / name).rglob('*'):
            relative = path.relative_to(root)
            if any(p in {'output', 'dist', '__pycache__', '.build-cache', 'build'}
                   or p.endswith('.egg-info') for p in relative.parts):
                continue
            if path.is_file() and path.suffix not in {'.so', '.whl', '.pyc'}:
                if relative.as_posix() != 'src/jztree/_version.py':
                    paths.add(path)
    paths.update(root / p for p in ('CMakeLists.txt', 'pyproject.toml', 'README.md', 'LICENSE'))
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.relative_to(root).as_posix().encode() + b'\0')
        h.update(path.read_bytes())
    return h.hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def wheel_manifest(directory):
    wheels = sorted(directory.glob('*.whl'))
    if len(wheels) != 1:
        raise RuntimeError(f'Expected one wheel in {directory}, found {len(wheels)}')
    with zipfile.ZipFile(wheels[0]) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'Corrupted wheel member: {bad}')
    return {wheels[0].name: digest(wheels[0])}


def completed(marker, directory):
    if not marker.exists():
        return False
    try:
        files = json.loads(marker.read_text())
        return bool(files) and all((directory / n).is_file() and digest(directory / n) == sha
                                   for n, sha in files.items())
    except (ValueError, OSError):
        return False


def dependencies(mode, jax_version):
    common = ['build', 'setuptools>=69', 'wheel', 'auditwheel', 'scikit-build-core==0.12.2',
              'nanobind==2.9.2', 'cmake==4.3.1', 'numpy>=2.2.6']
    if mode == 'main':
        return ['build', 'setuptools>=69', 'wheel']
    cuda = mode[2:]
    extra = f'cuda{cuda}' if mode == 'cu13' else 'cuda12-local'
    return common + [f'jax[{extra}]=={jax_version}', f'jaxlib=={jax_version}',
                     f'jax-cuda{cuda}-plugin=={jax_version}',
                     f'jax-cuda{cuda}-pjrt=={jax_version}']


def stage_python_package(root, state):
    """Keep setuptools' build/ and egg-info writes away from old checkout artifacts."""
    staged = state / 'python-source'
    package = staged / 'packaging/jztree'
    package.mkdir(parents=True, exist_ok=True)
    for name in ('pyproject.toml', 'README.md'):
        shutil.copy2(root / 'packaging/jztree' / name, package / name)
    shutil.copy2(root / 'LICENSE', package / 'LICENSE')
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.so', '*.egg-info')
    for name in ('jztree', 'jztree_utils'):
        shutil.copytree(root / 'src' / name, staged / 'src' / name,
                        ignore=ignore, dirs_exist_ok=True)
    return package


def prepare_environment(root, env_dir, mode, py_version, requirements):
    ready = env_dir / '.ready.json'
    py = env_dir / 'bin/python'
    if ready.exists() and py.exists():
        return py
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'cu12':
        run('micromamba', 'create', '-y', '-p', env_dir, '-c', 'conda-forge',
            f'python={py_version}', 'pip', 'cuda-nvcc', 'cuda-version=12.9',
            'cudnn', 'nccl', 'libcufft', 'cuda-cupti', 'libcublas', 'libcusparse',
            'openblas', 'libblas', 'liblapack', 'scipy', 'numpy')
    elif not py.exists():
        tag = 'cp' + py_version.replace('.', '')
        run('uv', 'venv', '--python', f'/opt/python/{tag}-{tag}/bin/python', env_dir)
    run('uv', 'pip', 'install', '--python', py, *requirements)
    # Record the full resolved environment, including transitive dependencies.
    freeze = subprocess.check_output(['uv', 'pip', 'freeze', '--python', str(py)], text=True)
    (env_dir / 'requirements-resolved.txt').write_text(freeze)
    if mode == 'cu12':
        with (env_dir / 'conda-explicit.txt').open('w') as out:
            run('micromamba', 'list', '--explicit', '-p', env_dir, stdout=out)
    if mode != 'main':
        run(py, '-c', 'import jax, jaxlib; print("Build JAX/jaxlib:", jax.__version__, jaxlib.__version__)')
    atomic_json(ready, {'requirements': requirements})
    return py


def build_one(root, mode, py_version, source, output, image, jax_version):
    package = 'jztree' if mode == 'main' else f'jztree-{mode}'
    requirements = dependencies(mode, jax_version)
    platform = os.environ['AUDITWHEEL_PLAT']
    # Environment paths survive source changes and container restarts. A new image
    # or dependency selection gets a fresh environment and CMake cache.
    env_key = hashlib.sha256(json.dumps([image, py_version, requirements, mode,
                                       'cuda12-toolkit-12.9']).encode()).hexdigest()[:16]
    env_dir = root / 'packaging/.build-cache' / mode / env_key
    environment = {'image': image, 'python': py_version, 'requirements': requirements,
                   'source': source, 'architectures': os.environ.get('CUDAARCHS', 'all'),
                   'platform': platform, 'mode': mode}
    key = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()[:16]
    state = output / key / ('cp' + py_version.replace('.', ''))
    state.mkdir(parents=True, exist_ok=True)
    raw = state / 'raw'
    repaired = state / 'wheelhouse'
    raw.mkdir(exist_ok=True)
    repaired.mkdir(exist_ok=True)
    marker = state / 'complete.json'
    if completed(marker, repaired):
        print(f'SKIP {mode} Python {py_version}: verified completed wheel in {repaired}', flush=True)
        return repaired
    started = time.monotonic()
    print(f'BUILD {mode} Python {py_version}; state: {state}', flush=True)
    atomic_json(state / 'inputs.json', environment)
    py = prepare_environment(root, env_dir, mode, py_version, requirements)
    env = os.environ.copy()
    env['PATH'] = str(env_dir / 'bin') + ':' + env['PATH']
    if mode == 'cu12':
        env['CUDACXX'] = str(env_dir / 'bin/nvcc')
        env['CUDAHOSTCXX'] = shutil.which('g++')
        env['CC'] = shutil.which('gcc')
        env['CXX'] = shutil.which('g++')
    if not completed(state / 'raw-complete.json', raw):
        package_dir = (stage_python_package(root, state) if mode == 'main'
                       else root / 'packaging' / package)
        args = ['uv', 'build', '--wheel', '--no-build-isolation', '--python', py,
                '-o', raw, package_dir]
        if mode != 'main':
            # Stable build directory retains completed object files after an interruption.
            args += [f'-Cbuild-dir={state / "cmake"}', '-Cbuild.verbose=true']
        run(*args, env=env, cwd=root)
        atomic_json(state / 'raw-complete.json', wheel_manifest(raw))
    wheel = next(raw.glob('*.whl'))
    if mode == 'main':
        shutil.copy2(wheel, repaired / wheel.name)
    else:
        # Interrupted repairs leave only an attempt directory, never a completed wheel.
        import tempfile
        attempt = Path(tempfile.mkdtemp(prefix='repair-', dir=state))
        run(py, '-m', 'auditwheel', 'repair', '--plat', platform, '-w', attempt, wheel, env=env)
        wheel_manifest(attempt)
        for result in attempt.glob('*.whl'):
            shutil.copy2(result, repaired / result.name)
    atomic_json(marker, wheel_manifest(repaired))
    print(f'DONE {mode} Python {py_version}: {time.monotonic()-started:.1f} s', flush=True)
    return repaired


def main():
    root = Path(os.environ.get('REPO_ROOT', '/workspace')).resolve()
    mode = os.environ['BUILD_MODE']
    output = root / os.environ['OUTPUT_DIR']
    output.mkdir(parents=True, exist_ok=True)
    cache = root / 'packaging/.build-cache'
    cache.mkdir(parents=True, exist_ok=True)
    # One writer avoids races between simultaneous builds sharing source/install paths.
    with (cache / 'build.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another wheel build is running in this checkout. Wait for it to finish.')
        source = source_digest(root)
        results = []
        for py_version in os.environ['PYTHON_VERSIONS_CSV'].split(','):
            results.append(build_one(root, mode, py_version, source, output,
                                     os.environ['BUILDER_IMAGE_ID'], os.environ['BUILD_JAX_VERSION']))
        if source_digest(root) != source:
            raise SystemExit('Build inputs changed during the run. Do not publish these wheels; rerun.')
        wheels = [p for directory in results for p in directory.glob('*.whl')]
        # This manifest lists only the selected, completed matrix, never stale output files.
        atomic_json(output / 'latest.json', {
            'source': source,
            'wheels': {str(p.relative_to(root)): digest(p) for p in wheels},
        })
        print('Validated artifact list:', output / 'latest.json', flush=True)


if __name__ == '__main__':
    main()

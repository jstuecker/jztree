"""CPU-only release-runner tests; no package installation or CUDA compilation."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('wheel_builder', Path(__file__).parents[1] / 'wheel_builder.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class WheelBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('src', 'packaging'):
            (self.root / name).mkdir()
        for name in ('CMakeLists.txt', 'pyproject.toml', 'README.md', 'LICENSE'):
            (self.root / name).write_text('fixture')
        project = self.root / 'packaging/jztree'
        project.mkdir()
        for name in ('pyproject.toml', 'README.md'):
            (project / name).write_text('fixture')
        for name in ('jztree', 'jztree_utils'):
            (self.root / 'src' / name).mkdir()
            (self.root / 'src' / name / '__init__.py').write_text('# fixture')
        self.env = patch.dict(os.environ, {'AUDITWHEEL_PLAT': 'manylinux_2_28_x86_64', 'CUDAARCHS': 'all'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.calls = []
        self.interrupt_build = False
        self.interrupt_repair = False

    def fake_run(self, *args, **kwargs):
        args = list(map(str, args))
        self.calls.append(args)
        if args[:2] == ['uv', 'build']:
            build_args = [a for a in args if a.startswith('-Cbuild-dir=')]
            if build_args:
                build_dir = Path(build_args[0].split('=', 1)[1])
                build_dir.mkdir(parents=True, exist_ok=True)
                (build_dir / 'compiled-object').write_text('retained')
            if self.interrupt_build:
                self.interrupt_build = False
                raise subprocess.CalledProcessError(130, args)
            out = Path(args[args.index('-o') + 1])
            with zipfile.ZipFile(out / 'jztree_cu13-1.1.0-cp312-cp312-linux_x86_64.whl', 'w') as z:
                z.writestr('fixture', 'not a real binary')
        elif 'auditwheel' in args:
            if self.interrupt_repair:
                self.interrupt_repair = False
                raise subprocess.CalledProcessError(130, args)
            shutil.copy2(args[-1], Path(args[args.index('-w') + 1]) / Path(args[-1]).name)

    def build(self, source='source', image='image'):
        with patch.object(builder, 'prepare_environment', return_value=Path('/fixture/python')), \
             patch.object(builder, 'run', side_effect=self.fake_run):
            return builder.build_one(self.root, 'cu13', '3.12', source,
                                     self.root / 'output', image, '0.8.3')

    def test_completed_wheel_is_skipped(self):
        first = self.build()
        self.calls.clear()
        self.assertEqual(first, self.build())
        self.assertEqual(self.calls, [])

    def test_interrupted_compilation_retains_build_directory(self):
        self.interrupt_build = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.build()
        objects = list(self.root.rglob('compiled-object'))
        self.assertEqual(len(objects), 1)
        self.build()
        self.assertTrue(objects[0].exists())
        paths = [next(a for a in c if a.startswith('-Cbuild-dir='))
                 for c in self.calls if c[:2] == ['uv', 'build']]
        self.assertEqual(paths[0], paths[1])

    def test_interrupted_repair_does_not_recompile(self):
        self.interrupt_repair = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.build()
        self.calls.clear()
        self.build()
        self.assertFalse(any(c[:2] == ['uv', 'build'] for c in self.calls))

    def test_changed_inputs_get_new_state(self):
        initial = self.build()
        self.assertNotEqual(initial, self.build(source='new source'))
        self.assertNotEqual(initial, self.build(image='new image'))

    def test_corrupted_completed_wheel_is_repaired(self):
        out = self.build()
        next(out.glob('*.whl')).write_text('corrupted')
        self.calls.clear()
        self.build()
        self.assertTrue(any('auditwheel' in c for c in self.calls))
        self.assertFalse(any(c[:2] == ['uv', 'build'] for c in self.calls))

    def test_source_hash_ignores_outputs_but_includes_untracked_sources(self):
        before = builder.source_digest(self.root)
        output = self.root / 'packaging/output'
        output.mkdir()
        (output / 'log.txt').write_text('build log')
        self.assertEqual(before, builder.source_digest(self.root))
        (self.root / 'src/new.py').write_text('new source')
        self.assertNotEqual(before, builder.source_digest(self.root))

    def test_pin_jax_and_backends(self):
        for cuda in ('12', '13'):
            deps = builder.dependencies('cu' + cuda, '0.8.3')
            self.assertIn('jaxlib==0.8.3', deps)
            self.assertIn(f'jax-cuda{cuda}-plugin==0.8.3', deps)
            self.assertIn(f'jax-cuda{cuda}-pjrt==0.8.3', deps)

    def test_main_does_not_compile_or_repair(self):
        with patch.object(builder, 'prepare_environment', return_value=Path('/fixture/python')), \
             patch.object(builder, 'run', side_effect=self.fake_run):
            result = builder.build_one(self.root, 'main', '3.11', 'source',
                                       self.root / 'output', 'image', '0.8.3')
        self.assertTrue(next(result.glob('*.whl')).is_file())
        self.assertFalse(any('auditwheel' in c for c in self.calls))
        self.assertFalse(any(a.startswith('-Cbuild-dir=') for c in self.calls for a in c))
        build = next(c for c in self.calls if c[:2] == ['uv', 'build'])
        self.assertIn('python-source', build[-1])
        self.assertNotEqual(Path(build[-1]), self.root / 'packaging/jztree')

    def test_setuptools_staging_excludes_old_build_artifacts(self):
        old = self.root / 'packaging/jztree/build/lib'
        old.mkdir(parents=True)
        sentinel = old / 'old-file'
        sentinel.write_text('old container-owned output')
        package = builder.stage_python_package(self.root, self.root / 'state')
        self.assertFalse((package / 'build').exists())
        self.assertTrue((package / '../../src/jztree/__init__.py').is_file())
        self.assertTrue((package / 'LICENSE').is_file())
        self.assertEqual(sentinel.read_text(), 'old container-owned output')


if __name__ == '__main__':
    unittest.main()

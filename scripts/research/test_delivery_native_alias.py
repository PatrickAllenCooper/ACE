"""New fabricated provenance checks; no target library or science imports."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).with_name('qualify_delivery_replay_runtime.py')
spec = importlib.util.spec_from_file_location('new_qualifier', SOURCE)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
ALIAS = 'torch._C._dynamo.autograd_compiler'


class NativeAlias(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.prefix = Path(self.temp.name).resolve() / 'environment'
        torchroot = self.prefix / 'lib/python3.11/site-packages/torch'
        (torchroot / 'lib').mkdir(parents=True)
        self.backing = torchroot / '_C.so'
        self.implementation = torchroot / 'lib/libtorch_python.so'
        self.backing.write_bytes(b'fabricated binary, never loaded')
        self.implementation.write_bytes(b'fabricated implementation, never loaded')
        self.alias = types.ModuleType(ALIAS)
        self.parent = types.ModuleType('torch._C')
        self.parent.__file__ = str(self.backing)
        self.dynamo = types.ModuleType('torch._C._dynamo')
        self.dynamo.compiled_autograd = self.alias
        self.parent._dynamo = self.dynamo
        self.torch = types.ModuleType('torch')
        self.torch._C = self.parent
        self.modules = {'torch': self.torch, 'torch._C': self.parent,
                        'torch._C._dynamo': self.dynamo, ALIAS: self.alias}
        self.pins = {str(f): q.sha(f) for f in (self.backing, self.implementation)}

    def check(self, name=ALIAS, module=None):
        with patch.dict(sys.modules, self.modules):
            return q.native_autograd_alias(name, self.alias if module is None else module,
                                           self.prefix, self.pins)

    def test_exact_alias_and_both_binary_witnesses(self):
        result = self.check()
        self.assertEqual(result['attribute_path'], 'torch._C._dynamo.compiled_autograd')
        self.assertEqual({w['backing_file'] for w in result['witnesses']}, set(self.pins))

    def test_substituted_alias_rejected(self):
        with self.assertRaisesRegex(ValueError, 'alias identity'):
            self.check(module=types.ModuleType(ALIAS))

    def test_substituted_native_parent_rejected(self):
        self.torch._C = types.ModuleType('torch._C')
        with self.assertRaisesRegex(ValueError, 'parent identity'):
            self.check()

    def test_substituted_dynamo_parent_rejected(self):
        self.modules['torch._C._dynamo'] = types.ModuleType('torch._C._dynamo')
        with self.assertRaisesRegex(ValueError, 'alias identity'):
            self.check()

    def test_changed_implementation_rejected(self):
        self.implementation.write_bytes(b'changed native code')
        with self.assertRaisesRegex(ValueError, 'backing bytes'):
            self.check()

    def test_missing_native_pin_rejected(self):
        self.pins.pop(str(self.implementation))
        with self.assertRaisesRegex(ValueError, 'pin set'):
            self.check()

    def test_external_native_parent_rejected(self):
        external = Path(self.temp.name).resolve() / 'external.so'
        external.write_bytes(b'external binary, never loaded')
        self.parent.__file__ = str(external)
        self.pins = {str(external): q.sha(external), str(self.implementation): q.sha(self.implementation)}
        with self.assertRaisesRegex(ValueError, 'backing bytes/origin'):
            self.check()

    def test_unrelated_alias_and_file_backed_substitution_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unexpected alias'):
            self.check(name='torch._C.some_other_alias')
        self.alias.__file__ = str(self.backing)
        with self.assertRaisesRegex(ValueError, 'invalid native alias'):
            self.check()

    def test_unknown_fileless_module_still_rejected(self):
        unknown = types.ModuleType('unexplained_module')
        with patch.object(sys, 'modules', {'unexplained_module': unknown}):
            with self.assertRaisesRegex(ValueError, 'unexplained fileless'):
                q.loaded_module_records(self.prefix, Path(self.temp.name), self.pins)

    def test_current_search_path_rejects_cwd_and_external(self):
        base = Path(self.temp.name).resolve() / 'base'
        valid = sorted(q.allowed_search_paths(self.prefix, base))
        with patch.object(sys, 'path', valid):
            q.search_path_check(self.prefix, base)
        for bad in ('', str(Path(self.temp.name).resolve() / 'external')):
            with self.subTest(path=bad), patch.object(sys, 'path', valid + [bad]):
                with self.assertRaisesRegex(ValueError, 'current sys.path'):
                    q.search_path_check(self.prefix, base)

    def test_cached_dependency_rejected_before_import(self):
        base = Path(self.temp.name).resolve() / 'base'
        reg = {'environment': str(self.prefix), 'approved_base_prefix': str(base),
               'exact_dependencies': {'torch': '2.9.1'}}
        with patch.object(sys, 'path', sorted(q.allowed_search_paths(self.prefix, base))), \
                patch.dict(sys.modules, {'torch': self.torch}):
            with self.assertRaisesRegex(ValueError, 'fresh dependency imports'):
                q.dependency_records(reg)

    def test_only_hash_bound_package_namespace_is_allowed(self):
        folder = Path(self.temp.name).resolve() / 'captured_learner/ace'
        folder.mkdir(parents=True)
        source = folder / '__init__.py'
        source.write_bytes(b'# fabricated source, not executed\n')
        module = types.ModuleType('ace')
        module.__file__ = str(source)
        module.__path__ = [str(folder)]
        allowed = {str(source): q.sha(source)}
        with patch.object(sys, 'modules', {'ace': module}):
            records = q.loaded_module_records(self.prefix, Path(self.temp.name) / 'base', self.pins, allowed)
            self.assertEqual(records[0]['sha256'], allowed[str(source)])
            module.__path__.append(str(folder.parent))
            with self.assertRaisesRegex(AssertionError, 'external namespace'):
                q.loaded_module_records(self.prefix, Path(self.temp.name) / 'base', self.pins, allowed)
            module.__path__ = [str(folder), str(self.prefix)]
            with self.assertRaisesRegex(AssertionError, 'external namespace'):
                q.loaded_module_records(self.prefix, Path(self.temp.name) / 'base', self.pins, allowed)

    def test_live_inventory_detects_change_and_new_members(self):
        f = self.prefix / 'recorded.txt'
        f.write_bytes(b'original')
        inv = {'prefix': str(self.prefix), 'files': [
            {'path': x.relative_to(self.prefix).as_posix(), 'kind': 'file',
             'bytes': x.stat().st_size, 'sha256': q.sha(x)}
            for x in sorted(self.prefix.rglob('*')) if x.is_file()]}
        q.verify_environment_inventory(self.prefix, inv)
        f.write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError, 'live environment file changed'):
            q.verify_environment_inventory(self.prefix, inv)
        f.write_bytes(b'original')
        (self.prefix / 'new.txt').write_bytes(b'not qualified')
        with self.assertRaisesRegex(ValueError, 'membership differs'):
            q.verify_environment_inventory(self.prefix, inv)

    def test_live_inventory_rejects_symlink_target_change(self):
        a, b = self.prefix/'a.txt', self.prefix/'b.txt'
        a.write_bytes(b'A'); b.write_bytes(b'B')
        link = self.prefix/'link'; link.symlink_to(a)
        inv = {'prefix': str(self.prefix), 'files': []}
        for x in sorted(self.prefix.rglob('*')):
            if x.is_symlink():
                inv['files'].append({'path': x.relative_to(self.prefix).as_posix(),
                                    'kind': 'symlink', 'link': str(a),
                                    'resolved': str(a.resolve()), 'target_sha256': q.sha(a)})
            elif x.is_file():
                inv['files'].append({'path': x.relative_to(self.prefix).as_posix(),
                                    'kind': 'file', 'bytes': x.stat().st_size, 'sha256': q.sha(x)})
        q.verify_environment_inventory(self.prefix, inv)
        link.unlink(); link.symlink_to(b)
        with self.assertRaisesRegex(ValueError, 'live environment link changed'):
            q.verify_environment_inventory(self.prefix, inv)


if __name__ == '__main__':
    unittest.main(verbosity=2)

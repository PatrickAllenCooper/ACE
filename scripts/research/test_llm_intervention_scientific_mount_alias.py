"""New filesystem-only alias cases. No original tests, runtime or science calls."""
import ast
from pathlib import Path
import tempfile
import types
import unittest


SOURCE = Path(__file__).absolute().parent
NAMES = ('supervise_llm_intervention_scientific_cpu.py',
         'prepare_llm_intervention_scientific_campaign_cpu.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def functions(name):
    tree = ast.parse((SOURCE/name).read_bytes())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ('_output_path_binding', 'output_path_binding')]
    namespace = dict(Path=Path, B=types.SimpleNamespace(need=require))
    exec(compile(ast.Module(body=body, type_ignores=[]), name, 'exec'), namespace)
    return namespace, ast.dump(ast.Module(body=body, type_ignores=[]))


class MountAlias(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.physical = self.root/'physical'
        self.physical.mkdir()
        self.logical = self.root/'logical'
        self.logical.symlink_to(self.physical, target_is_directory=True)
        self.apis = [functions(n)[0]['_output_path_binding'] for n in NAMES]

    def test_identity_and_exact_alias_with_missing_output(self):
        for fn in self.apis:
            self.assertFalse(fn(self.physical/'job/output', self.logical, self.physical)['mount_alias'])
            for tail in ('job/output', 'job/output/session/worker'):
                result = fn(self.logical/tail, self.logical, self.physical)
                self.assertEqual(result['logical'], str(self.logical/tail))
                self.assertEqual(result['resolved'], str(self.physical/tail))
                self.assertTrue(result['mount_alias'])
                self.assertEqual(result['root_inode'], self.physical.stat().st_ino)

    def test_unknown_alias_and_mismatched_mount(self):
        other = self.root/'other'
        other.mkdir()
        unknown = self.root/'unknown'
        unknown.symlink_to(self.physical, target_is_directory=True)
        for fn in self.apis:
            for path, physical in ((unknown/'job/output', self.physical),
                                   (self.logical/'job/output', other)):
                with self.assertRaises(ValueError):
                    fn(path, self.logical, physical)

    def test_descendant_symlink_even_when_suffix_resolves_identically(self):
        # An alias below the admitted mount is forbidden even when its target
        # remains inside the same filesystem.
        (self.physical/'actual').mkdir()
        (self.physical/'link').symlink_to(self.physical/'actual', target_is_directory=True)
        for fn in self.apis:
            with self.assertRaises(ValueError):
                fn(self.logical/'link/output', self.logical, self.physical)

    def test_escape_and_dangling_symlink(self):
        (self.physical/'escape').symlink_to(self.root, target_is_directory=True)
        (self.physical/'dangling').symlink_to(self.root/'absent')
        for fn in self.apis:
            for name in ('escape', 'dangling'):
                with self.assertRaises(ValueError):
                    fn(self.logical/name/'output', self.logical, self.physical)

    def test_relative_and_dotdot_rejected(self):
        for fn in self.apis:
            for path in (Path('relative/output'), self.logical/'..'/'output'):
                with self.assertRaises(ValueError):
                    fn(path, self.logical, self.physical)

    def test_production_roots_fixed_and_functions_identical(self):
        a, first = functions(NAMES[0])
        b, second = functions(NAMES[1])
        self.assertEqual(first, second)
        for api in (a, b):
            calls = []
            api['_output_path_binding'] = lambda *args: calls.append(args)
            api['output_path_binding']('/scratch/alpine/job/output')
            self.assertEqual(calls, [('/scratch/alpine/job/output', '/scratch/alpine', '/gpfs/alpine1/scratch')])

    def test_changed_sites_record_binding_before_work_and_gpu_pin_matches(self):
        import hashlib
        for name in NAMES:
            text = (SOURCE/name).read_text()
            self.assertNotIn("'resolved output required'", text)
            main = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == 'output_path_binding']
            self.assertTrue(calls)
            self.assertLess(min(n.lineno for n in calls), text.splitlines().index('        ctx=authenticate(args)')+1)
            self.assertIn('output_path_binding', text[text.index("save(out/'start.json'"):])
        gpu = ast.parse((SOURCE/'qualify_llm_intervention_scientific_campaign_gpu.py').read_bytes())
        pin = next(ast.literal_eval(n.value) for n in gpu.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'CPU_ENTRY_SHA' for t in n.targets))
        self.assertEqual(pin, hashlib.sha256((SOURCE/NAMES[1]).read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()

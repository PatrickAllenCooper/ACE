"""Focused R1/R2 integration cases. Stops before any scientific import or draw."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

SOURCE = Path(__file__).absolute().parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


class AliasIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.physical = self.root/'physical'
        self.physical.mkdir()
        self.logical = self.root/'logical'
        self.logical.symlink_to(self.physical, target_is_directory=True)

    def test_physical_overlap_rejected_at_both_outer_boundaries(self):
        def extract(file, name, namespace):
            tree = ast.parse((SOURCE/file).read_bytes())
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[fn], type_ignores=[]), file, 'exec'), namespace)
            return namespace[name]
        original = extract('prepare_llm_intervention_campaign_cpu.py', 'separate', dict(Path=Path, need=need))
        campaign = extract('prepare_llm_intervention_scientific_campaign_cpu.py', 'separate',
                           dict(Path=Path, B=types.SimpleNamespace(separate=original)))
        supervisor = extract('finalize_llm_intervention_campaign_cpu.py', 'separate', dict(Path=Path, need=need))
        for separate in (campaign, supervisor):
            for tail in ('inputs', 'inputs/descendant', ''):
                with self.assertRaises(ValueError):
                    separate([self.logical/tail, self.physical/'inputs'])
            separate([self.logical/'output', self.physical/'inputs'])
        text = (SOURCE/'prepare_llm_intervention_scientific_campaign_cpu.py').read_text()
        # Only the adapter itself calls the closed lexical implementation.
        self.assertEqual(text.count('B.separate('), 1)
        self.assertEqual(text.count('separate(['), 5)

    def test_collection_reaches_authentication_without_collection_and_preserves_paths(self):
        name = 'collect_llm_intervention_scientific_training_cpu.py'
        spec = importlib.util.spec_from_file_location('_alias_fixture_collection', SOURCE/name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        checked = []
        module.output_path_binding = lambda p: module._output_path_binding(p, self.logical, self.physical)
        def controlled_authentication(*args):
            checked.append(str(args[-1]))
            raise RuntimeError('artificial authentication boundary')
        module._authenticate = controlled_authentication
        target = self.logical/'task'
        with self.assertRaisesRegex(RuntimeError, 'artificial authentication boundary'):
            module.run_collection(packet_raw=b'{}', packet_sha256='0'*64,
                                  admission_raw=b'{}', admission_sha256='1'*64,
                                  inputs={}, output=target)
        self.assertEqual(checked, [str(target)])
        intent = json.loads((target/'intent.json').read_bytes())
        self.assertEqual(intent['output_path_binding']['logical'], str(target))
        self.assertEqual(intent['output_path_binding']['resolved'], str(self.physical/'task'))
        failure = json.loads((target/'failure.json').read_bytes())
        self.assertEqual(failure['stage'], 'admission')
        self.assertFalse(failure['collection_invoked'])
        self.assertEqual(failure['responses_before_collection'], 0)
        with self.assertRaises(FileExistsError):
            module.run_collection(packet_raw=b'{}', packet_sha256='0'*64,
                                  admission_raw=b'{}', admission_sha256='1'*64,
                                  inputs={}, output=target)
        (self.physical/'redirect').symlink_to(self.root)
        with self.assertRaises(ValueError):
            module.run_collection(packet_raw=b'{}', packet_sha256='0'*64,
                                  admission_raw=b'{}', admission_sha256='1'*64,
                                  inputs={}, output=self.logical/'redirect'/'unclaimed')
        self.assertFalse((self.root/'unclaimed').exists())

    def test_collection_helper_and_dependent_gpu_pin(self):
        def bodies(name):
            tree = ast.parse((SOURCE/name).read_bytes())
            selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name in ('_output_path_binding','output_path_binding')]
            # Collection has its own unchanged validation exception helper.
            class Normalize(ast.NodeTransformer):
                def visit_Attribute(self, node):
                    if isinstance(node.value, ast.Name) and node.value.id == 'B' and node.attr == 'need':
                        return ast.Name(id='_need',ctx=ast.Load())
                    return self.generic_visit(node)
            return ast.dump(Normalize().visit(ast.Module(body=selected,type_ignores=[])))
        self.assertEqual(bodies('collect_llm_intervention_scientific_training_cpu.py'),
                         bodies('supervise_llm_intervention_scientific_cpu.py'))
        tree = ast.parse((SOURCE/'qualify_llm_intervention_scientific_campaign_gpu.py').read_bytes())
        actual = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='CPU_ENTRY_SHA' for t in n.targets))
        self.assertEqual(actual,hashlib.sha256((SOURCE/'prepare_llm_intervention_scientific_campaign_cpu.py').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()

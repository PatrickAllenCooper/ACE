"""Operational source-delta checks only, with artificial helper bytes."""
import ast
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('derive_entry', HERE/'derive_gpu_entry.py')
derive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(derive)
ORIGINAL = HERE.parents[2]/'ace_llm_intervention_scientific_campaign_gpu_20261108/launch01/inputs/qualify_llm_intervention_scientific_campaign_gpu.py'


class OperationalDelta(unittest.TestCase):
    def test_only_declared_operational_functions_change(self):
        before = ORIGINAL.read_bytes()
        after = derive.derive_entry(before, b'# artificial receipt helper\n',
                                    b'# artificial history helper\n')
        nodes = lambda data: {n.name: ast.dump(n, include_attributes=False)
                             for n in ast.parse(data).body
                             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        old, new = nodes(before), nodes(after)
        changed = {name for name in old if old[name] != new[name]}
        self.assertEqual(changed, {'admit_cpu', 'authenticate', 'run_controller'})
        self.assertEqual(set(new)-set(old), {'recovery_contract'})
        for name in ('main', 'stage_and_exec', 'controller_handoff', 'resource_contract',
                     'admission_contract', 'validated_output', 'recheck_inputs',
                     'transfer_phase_deadline', 'CapturedSources'):
            self.assertEqual(old[name], new[name], name)
        controller = next(n for n in ast.parse(after).body
                          if isinstance(n, ast.FunctionDef) and n.name == 'run_controller')
        inventory_calls = [ast.unparse(n.args[0]) for n in ast.walk(controller)
                           if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                           and n.func.id == 'inventory']
        self.assertEqual(inventory_calls, ['worker'])
        self.assertIn(b"'full_external_postcustody_required':True", after)

    def test_changed_original_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'original entry digest'):
            derive.derive_entry(ORIGINAL.read_bytes()+b'\n', b'# fixture', b'# fixture')


if __name__ == '__main__':
    unittest.main()

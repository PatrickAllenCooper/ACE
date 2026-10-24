"""Artificial custody failures for the new report orchestrator, no science."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import run_delivery_prospective_reporting as runner


class ReportingCustody(unittest.TestCase):
    def invoke(self, root, result, bad_output=False):
        output = root/'primary'; output.mkdir()
        (output/'test.tex').write_text('fabricated')
        receipt = {'status': 'complete', 'freeze_sha256': 'freeze',
                   'outputs': {'primary/test.tex': runner.sha(b'wrong' if bad_output else b'fabricated')}}
        (root/'report_receipt.json').write_text(json.dumps(receipt))
        reg = {'wall_seconds': 3600, 'rss_bytes': 3*2**30, 'source_pins': {'run_delivery_prospective_reporting.py': 'source'}}
        modules = {
            'supervise_delivery_qualified_replay': types.SimpleNamespace(allocation=lambda *a: {'fabricated': True, 'raw_scontrol': 'RunTime=00:00:00'}),
            'delivery_replay_supervision': types.SimpleNamespace(supervise=lambda *a: result),
            'runner_delivery_confirmation': types.SimpleNamespace(process_rss=lambda *a: 0)}
        with patch.object(runner, 'context', return_value=(root, reg, {})), \
                patch.object(runner, 'module', side_effect=lambda root, name, src: modules[name]):
            runner.parent(root, 'freeze')

    def test_failed_child_retains_explicit_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SystemExit):
                self.invoke(root, {'status': 'time_limit', 'exit_code': -9, 'cleanup': {'complete': True}})
            record = json.loads((root/'report_execution.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertEqual(record['child']['status'], 'time_limit')
            self.assertTrue((root/'report_receipt.json').exists())

    def test_success_requires_output_bytes_and_cleanup(self):
        for cleanup, bad_output in [(False, False), (True, True), (True, False)]:
            with self.subTest(cleanup=cleanup, bad_output=bad_output), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result = {'status': 'complete', 'exit_code': 0, 'cleanup': {'complete': cleanup}}
                if cleanup and not bad_output:
                    self.invoke(root, result, bad_output)
                else:
                    with self.assertRaises(SystemExit): self.invoke(root, result, bad_output)
                record = json.loads((root/'report_execution.json').read_text())
                self.assertEqual(record['status'], 'complete' if cleanup and not bad_output else 'failed')

    def test_output_retry_is_rejected_and_original_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runner.write(root, 'report_started.json', {'original': True})
            before = (root/'report_started.json').read_bytes()
            with self.assertRaises(FileExistsError): runner.write(root, 'report_started.json', {'original': False})
            self.assertEqual((root/'report_started.json').read_bytes(), before)

    def test_source_capture_and_cached_module_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'fake.py'; source.write_bytes(b'x=3\n')
            pin = runner.sha(source.read_bytes())
            self.assertEqual(runner.capture(source, pin), b'x=3\n')
            source.write_bytes(b'x=4\n')
            with self.assertRaisesRegex(ValueError, 'input changed'): runner.capture(source, pin)
            with self.assertRaisesRegex(ValueError, 'cached project'): runner.module(root, 'json', {})

    def test_original_lineage_rejects_omissions_and_independently_valid_other_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); package = root/'package'; (package/'B').mkdir(parents=True)
            original = root/'original'; original.mkdir()
            pins = {}
            for name in runner.ORIGINAL_INPUTS:
                value = ('fabricated '+name).encode(); (original/name).write_bytes(value)
                pins[name] = runner.sha(value)
            contract = {'registration_sha256': pins['registration.json'],
                'original_complete_sha256': pins['complete.json'], 'fit_seal_sha256': pins['fit_seal.json'],
                'upstream_gate': {k: pins[n] for k, n in [('acceptance_sha256','acceptance.json'),
                    ('scores_sha256','scores.json'), ('audit_execution_sha256','audit_execution.json')]}}
            raw = json.dumps(contract).encode(); (package/'B/replay_contract.json').write_bytes(raw)
            contract_pin = runner.sha(raw)
            raw = json.dumps({'files': [{'path': 'B/replay_contract.json', 'sha256': contract_pin}]}).encode()
            (package/'manifest.json').write_bytes(raw); manifest_pin = runner.sha(raw)
            raw = json.dumps({'contract_sha256': contract_pin}).encode(); (root/'replay_acceptance.json').write_bytes(raw)
            reg = {'package': str(package), 'original_results': str(original), 'manifest_sha256': manifest_pin,
                   'metadata_pins': {'replay_acceptance.json': runner.sha(raw)}, 'original_input_pins': dict(pins)}
            stub = types.SimpleNamespace(gate=lambda c: None)
            self.assertEqual(runner.original_lineage(root, reg, stub), original.resolve())
            reg['original_input_pins'] = {}
            with self.assertRaisesRegex(ValueError, 'complete original-input'): runner.original_lineage(root, reg, stub)
            reg['original_input_pins'] = dict(pins); reg['original_input_pins']['scores.json'] = runner.sha(b'other study')
            (original/'scores.json').write_bytes(b'other study')
            with self.assertRaisesRegex(ValueError, 'study lineage'): runner.original_lineage(root, reg, stub)


if __name__ == '__main__': unittest.main()

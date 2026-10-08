"""Synthetic rejection and analytically solved inference checks; no B outcomes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import replay_delivery_prospective_release as r
import test_delivery_prospective_release_core as fixtures


def fixture(root):
    helper = fixtures.ReleaseCoreTests()
    rh, supervisor, a = helper.fixture(root)
    import prepare_delivery_prospective_release_core as c
    g = c.accepted_before_scores(root, rh)
    # The fixture is deliberately independent of the actual study receipt.
    g['registration_sha256'] = r.REGISTRATION
    a = g['custody_bound_acceptance_projection']['metadata']
    a['study_registration_sha256'] = r.REGISTRATION
    a['auditor_sha256'] = r.AUDITOR_SHA
    supervisor['registration_sha256'] = r.REGISTRATION
    complete = json.loads((root/'complete.json').read_text())
    complete['registration_sha256'] = r.REGISTRATION
    cells = []
    for case in r.WORLDS:
        for history in r.HISTORIES:
            for arm, init in r.ARMS:
                cells.append({'index': len(cells), 'case': case, 'history': history, 'arm': arm, 'init': init,
                              'graph_size': int(case.split(':')[0]),
                              'folder': f"B/fits/{case.replace(':', '-')}/{history}/{arm}-i{init}"})
    return {'schema': 'delivery-prospective-replay-v1', 'registration_sha256': r.REGISTRATION,
            'original_revision': r.REVISION, 'dependencies': r.DEPENDENCIES, 'matrix_fits': 640,
            'core_sha256': r.CORE_SHA, 'core_contract_sha256': r.CORE_CONTRACT_SHA,
            'primary_cells': 240, 'scope': 'fixture only', 'upstream_gate': g, 'supervisor': supervisor,
            'complete': complete, 'fit_seal_sha256': complete['fit_seal_sha256'],
            'original_complete_sha256': a['complete_sha256'], 'resources': {'rss_bytes': 1024, 'audit_wall_seconds': 900},
            'cells': cells, 'training': {c+'/'+h: 'unused' for c in r.WORLDS for h in r.HISTORIES},
            'evaluation': {c: 'unused' for c in r.WORLDS}, 'descriptors': {c: 'unused' for c in r.WORLDS},
            'world_execution': {c: 'unused' for c in r.WORLDS}}


class ReplayTests(unittest.TestCase):
    def test_python_floor_precedes_any_artifact_access(self):
        with patch.object(r.sys, 'version_info', (3, 10, 19)), \
                patch.object(r, 'captured', side_effect=AssertionError('unsupported interpreter opened artifact')):
            with self.assertRaisesRegex(ValueError, 'Python >=3.11'):
                r.replay(Path('absent'), 'a' * 64)
        # At the supported boundary the normal manifest-pin barrier is reached.
        with patch.object(r.sys, 'version_info', (3, 11, 0)), \
                patch.object(r, 'captured', side_effect=RuntimeError('manifest barrier reached')):
            with self.assertRaisesRegex(RuntimeError, 'manifest barrier reached'):
                r.replay(Path('absent'), 'a' * 64)

    def test_dependency_mismatch_reports_actual_import(self):
        module = SimpleNamespace(__version__='2.2.5', __file__='/fixture/numpy/__init__.py')
        dist = SimpleNamespace(version='2.2.6', locate_file=lambda name: module.__file__)
        with patch.dict(r.DEPENDENCIES, {'numpy': '2.2.6'}, clear=True), \
                patch.object(r.importlib.metadata, 'distribution', return_value=dist), \
                patch.object(r, '__import__', return_value=module, create=True):
            with self.assertRaisesRegex(ValueError, 'required=2.2.6, metadata=2.2.6, imported=2.2.5'):
                r.dependency_modules()
            module.__version__ = '2.2.6'
            r.dependency_modules()

    def test_imported_dependency_origin_is_checked(self):
        import numpy as np
        dist = SimpleNamespace(version=np.__version__, locate_file=lambda name: np.__file__)
        with patch.dict(r.DEPENDENCIES, {'numpy': np.__version__}, clear=True), \
                patch.object(r.importlib.metadata, 'distribution', return_value=dist):
            r.dependency_modules()
            dist.locate_file = lambda name: '/different/package/__init__.py'
            with self.assertRaises(ValueError): r.dependency_modules()

    def test_archived_learner_executes_authenticated_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'source/ace').mkdir(parents=True)
            p = root/'source/ace/__init__.py'; raw = b'answer = 42\n'; p.write_bytes(raw)
            loader = r.LearnerSnapshots(root, 'source', {'ace/__init__.py': hashlib.sha256(raw).hexdigest()})
            p.write_text('raise AssertionError("source reopened")\n')
            spec = loader.find_spec('ace'); module = r.importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            self.assertEqual(module.answer, 42)
            with self.assertRaises(ValueError): loader.find_spec('ace.unrecorded')

    def test_missing_external_pin_rejects_without_reading_files(self):
        with patch.object(r, 'captured', side_effect=AssertionError('manifest opened without pin')):
            for pin in (None, '', 'a'*63, 'A'*64):
                with self.assertRaises(ValueError): r.replay(Path('absent'), pin)

    def test_byte_verification_does_not_decode_scientific_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = b'{"scientific_sentinel": not-decoded-before-barrier}'
            (root/'scores.json').write_bytes(raw)
            (root/'manifest.json').write_text('{}'); (root/'manifest.sha256').write_text('unused')
            h = hashlib.sha256(raw).hexdigest()
            manifest = {'schema': 'delivery-anonymous-derived-v1', 'files': [{
                'path': 'scores.json', 'bytes': len(raw), 'sha256': h, 'original_sha256': h,
                'transform': {'kind': 'identity'}}]}
            with patch.object(r.json, 'loads', side_effect=AssertionError('scientific JSON decoded')):
                r.byte_integrity(root, manifest)
            (root/'scores.json').write_bytes(raw+b' ')
            with self.assertRaises(ValueError): r.byte_integrity(root, manifest)

    def test_missing_world_and_reordered_cells_reject(self):
        with tempfile.TemporaryDirectory() as td:
            good = fixture(Path(td)); r.gate(good)
            bad = copy.deepcopy(good); del bad['evaluation']['30:19']
            with self.assertRaises(ValueError): r.gate(bad)
            bad = copy.deepcopy(good); bad['cells'][0], bad['cells'][1] = bad['cells'][1], bad['cells'][0]
            with self.assertRaises(ValueError): r.gate(bad)

    def test_failed_supervisor_and_nonfinite_evidence_reject(self):
        with tempfile.TemporaryDirectory() as td:
            good = fixture(Path(td))
            for field, value in [('exit_code', False), ('status', 'failed'), ('elapsed_seconds', float('nan')),
                                 ('peak_tree_rss_bytes', -1)]:
                bad = copy.deepcopy(good); bad['supervisor'][field] = value
                with self.assertRaises(ValueError): r.gate(bad)
            bad = copy.deepcopy(good)
            bad['upstream_gate']['custody_bound_acceptance_projection']['metadata']['replay_max_abs_deltas']['639'] = float('inf')
            with self.assertRaises(ValueError): r.gate(bad)
            bad = copy.deepcopy(good)
            bad['upstream_gate']['custody_bound_acceptance_projection']['original_acceptance_sha256'] = 'f'*64
            with self.assertRaises(ValueError): r.gate(bad)

    def test_wrong_runtime_rejects_before_verifier_or_scores(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'B').mkdir()
            contract = fixture(root); raw = json.dumps(contract).encode()
            (root/'B/replay_contract.json').write_bytes(raw)
            manifest = {'files': [{'path': 'B/replay_contract.json', 'sha256': hashlib.sha256(raw).hexdigest()}]}
            raw = json.dumps(manifest).encode(); (root/'manifest.json').write_bytes(raw)
            with patch.object(r.importlib.metadata, 'version', return_value='wrong'), \
                    patch.object(r, 'verify', side_effect=AssertionError('outcome scanner too early')):
                with self.assertRaisesRegex(ValueError, 'exact B runtime'): r.replay(root, hashlib.sha256(raw).hexdigest())

    def test_authenticated_code_executes_one_captured_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); name = 'analytical_snapshot_module'
            path = root/'helper.py'; original = b'value = 42\n'; path.write_bytes(original)
            native = Path.read_bytes
            def replaced(p):
                raw = native(p)
                if p == path: path.write_text('raise AssertionError("reopened code")\n')
                return raw
            try:
                with patch.object(Path, 'read_bytes', replaced):
                    module = r.load_code(root, path.name, hashlib.sha256(original).hexdigest(), name)
                self.assertEqual(module.value, 42)
            finally:
                r.sys.modules.pop(name, None)

    def test_snapshot_journal_keeps_exact_charge_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); p = root/'queries.ndjson'
            for values in ([{'attempt': 1}, {'attempt': 2}], [{'attempt': 1}, {'attempt': 3}], [{'attempt': True}]):
                p.write_text('\n'.join(json.dumps(v) for v in values)+'\n')
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                _, _, _, journal = r.snapshot_io(root, {'queries.ndjson': {'sha256': h}})
                if len(values) == 2 and values[-1]['attempt'] == 2:
                    self.assertEqual(journal(p), 2)
                else:
                    with self.assertRaises(ValueError): journal(p)

    def test_wrong_core_binding_rejects_before_runtime_or_losses(self):
        with tempfile.TemporaryDirectory() as td:
            contract = fixture(Path(td)); contract['core_sha256'] = 'f'*64
            with self.assertRaises(ValueError): r.gate(contract)

    def test_chain_metric_uses_cached_predictions_and_training_variance(self):
        import numpy as np
        import torch
        class Linear(torch.nn.Module):
            def __init__(self, a): super().__init__(); self.a = a
            def forward(self, x): return self.a*x[:, 0]
        # Chain X2=2X1, X3=3X2. Perturbed free-running X2 produces X3 error;
        # the observed-parent residual can still be zero at X3.
        data = {'order': ['X1', 'X2', 'X3'], 'roots': ['X1'], 'target': 'X3',
                'parents': {'X1': [], 'X2': ['X1'], 'X3': ['X2']},
                'rows': [{'node_values': {'X1': x, 'X2': 2*x, 'X3': 6*x}} for x in (-1., 1.)]}
        prediction = {'X1': np.array([-1., 1.]), 'X2': np.array([-1., 1.]), 'X3': np.array([-3., 3.])}
        metric = r.metric_from_cached(data, data, prediction, {'X2': Linear(1.), 'X3': Linear(3.)}, 'delivery', torch)
        self.assertEqual(metric['mse'], 9.)
        self.assertEqual(metric['training_target_variance'], 36.)
        self.assertEqual(metric['nmse'], .25)
        self.assertEqual(metric['secondary_mechanism_diagnostics']['X3']['observed_parent_mse'], 0.)
        self.assertEqual(metric['secondary_mechanism_diagnostics']['X3']['propagated_prediction_shift_mse'], 9.)
        changed = copy.deepcopy(data)
        for row in changed['rows']: row['node_values']['X3'] *= 10
        scaled = r.metric_from_cached(data, changed, prediction, {}, 'simpler', torch)
        self.assertEqual(scaled['nmse'], .0025)


if __name__ == '__main__': unittest.main()

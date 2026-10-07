"""Composite inventory ordering/conflict checks with metadata-only fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reconcile_delivery_prospective_custody as c
import replay_delivery_prospective_release as constants
from verify_delivery_release import sha


class CustodyTests(unittest.TestCase):
    def test_upstream_gate_is_first(self):
        with patch.object(c, 'accepted_before_scores', side_effect=ValueError('not accepted')), \
                patch.object(c, 'read_snapshot', side_effect=AssertionError('metadata before gate')):
            with self.assertRaisesRegex(ValueError, 'not accepted'):
                c.inventory('absent', ['absent'], 'absent', 'absent')

    def fixture(self, root):
        study, project, source = root/'study', root/'project', root/'source'
        for p in (study, project, source): p.mkdir()
        cells = []
        for case in constants.WORLDS:
            for history in constants.HISTORIES:
                for arm, init in constants.ARMS:
                    cells.append({'index': len(cells), 'case': case, 'history': history, 'arm': arm, 'init': init,
                                  'out': 'canonical-fit-'+str(len(cells))})
        (study/'matrix.json').write_text(json.dumps({'cells': cells}))
        (study/'fit_seal.json').write_text(json.dumps({'matrix_sha256': sha(study/'matrix.json'),
            'artifacts': {v['out']: {'receipt_sha256': 'e'*64, 'model_sha256': 'f'*64} for v in cells}}))
        (source/'learner.py').write_text('value = 1\n')
        (project/'worker.py').write_text('value = 2\n')
        (project/'source_commit_receipt.json').write_text(json.dumps({'source_revision': 'fixture',
                                                                    'files': {'worker.py': sha(project/'worker.py')}}))
        (study/'registration.json').write_text(json.dumps({'source_revision': 'fixture',
            'source_commit_receipt_sha256': sha(project/'source_commit_receipt.json'),
            'source_hashes': {'learner.py': sha(source/'learner.py')}}))
        gate = {'acceptance_sha256': 'a'*64, 'audit_execution_sha256': 'b'*64, 'scores_sha256': 'c'*64,
                'custody_bound_acceptance_projection': {'metadata': {
                    'fit_seal_sha256': sha(study/'fit_seal.json'), 'complete_sha256': 'd'*64,
                    'phase_execution_hashes': {}, 'training_receipt_hashes': {}, 'evaluation_receipt_hashes': {}}}}
        fits = {'cells_verified': 640, 'missing_indices': [], 'registration_sha256': c.REGISTRATION_SHA,
                'duplicate_copies_verified': 0}
        return study, project, source, gate, fits

    def test_partial_inventory_rejects_before_raw_outcome_walk(self):
        with tempfile.TemporaryDirectory() as td:
            study, project, source, gate, fits = self.fixture(Path(td).resolve())
            fits['cells_verified'] = 560; fits['missing_indices'] = [639]
            with patch.object(c, 'accepted_before_scores', return_value=gate), \
                    patch.object(c, 'fit_inventory', return_value=fits), \
                    patch.object(c, 'REGISTRATION_SHA', sha(study/'registration.json')):
                fits['registration_sha256'] = c.REGISTRATION_SHA
                with self.assertRaisesRegex(ValueError, '640'): c.inventory(study, [study], project, source)

    def test_omitted_original_seal_cell_rejects(self):
        with tempfile.TemporaryDirectory() as td:
            study, project, source, gate, fits = self.fixture(Path(td).resolve())
            seal = json.loads((study/'fit_seal.json').read_text())
            del seal['artifacts']['canonical-fit-639']
            (study/'fit_seal.json').write_text(json.dumps(seal))
            gate['custody_bound_acceptance_projection']['metadata']['fit_seal_sha256'] = sha(study/'fit_seal.json')
            with patch.object(c, 'accepted_before_scores', return_value=gate), \
                    patch.object(c, 'fit_inventory', return_value=fits), \
                    patch.object(c, 'REGISTRATION_SHA', sha(study/'registration.json')):
                fits['registration_sha256'] = c.REGISTRATION_SHA
                with self.assertRaisesRegex(ValueError, 'fit-seal membership'): c.inventory(study, [study], project, source)

    def test_claimed_duplicate_conflict_rejects_without_decoding_outcomes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve(); study, project, source, gate, fits = self.fixture(root)
            copy = root/'other'; copy.mkdir()
            (study/'scores.json').write_text('not parsed as JSON during inventory')
            (copy/'scores.json').write_text('different bytes')
            with patch.object(c, 'accepted_before_scores', return_value=gate), \
                    patch.object(c, 'fit_inventory', return_value=fits), \
                    patch.object(c, 'REGISTRATION_SHA', sha(study/'registration.json')):
                fits['registration_sha256'] = c.REGISTRATION_SHA
                with self.assertRaisesRegex(ValueError, 'conflicting'): c.inventory(study, [study, copy], project, source)
            (copy/'scores.json').write_bytes((study/'scores.json').read_bytes())
            with patch.object(c, 'accepted_before_scores', return_value=gate), \
                    patch.object(c, 'fit_inventory', return_value=fits), \
                    patch.object(c, 'REGISTRATION_SHA', sha(study/'registration.json')):
                fits['registration_sha256'] = c.REGISTRATION_SHA
                with self.assertRaisesRegex(ValueError, 'closure incomplete'):
                    c.inventory(study, [study, copy], project, source)


if __name__ == '__main__': unittest.main()

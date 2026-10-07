"""Distinct interrupted reservations, retained aliases and malformed journals."""
import json
from pathlib import Path
import tempfile
import unittest

from replay_delivery_confirmation_release import FROZEN_HISTORIES, accounting, journal_count
from verify_delivery_release import sha


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value))


def journal(p, ids):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(json.dumps({'attempt': i, 'at': 'fixed fixture timestamp'}) for i in ids)+'\n')


class ConfirmationReleaseTests(unittest.TestCase):
    def test_journal_rejects_gaps_duplicates_booleans_and_reordering(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'calls.jsonl'
            for ids in ([1, 3], [1, 1], [True], [2, 1], []):
                journal(p, ids)
                with self.assertRaises(ValueError): journal_count(p)
            journal(p, [1, 2]); self.assertEqual(journal_count(p), 2)

    def fixture(self, root):
        import hashlib
        cases = root/'F/cases'
        for seed in FROZEN_HISTORIES:
            p = cases/seed; journal(p/'calls.jsonl', [1])
            rows = [{'query_index': 0, 'method': 'ace', 'role': 'seed', 'selected': False}]
            write(p/'complete.json', {'seed': int(seed), 'inits': [0, 1, 2], 'calls': 1,
                                     'rows_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()})
            (p/'online').mkdir(); (p/'online/observations.ndjson').write_text(json.dumps(rows[0])+'\n')
            write(p/'online/meta.json', {'query_counts': {'ace': {'seed': 1, 'total': 1, 'startup': 1, 'executed': 0}}})
        write(root/'F/sealed.json', {'case_hashes': {str(p.relative_to(cases)): sha(p) for p in cases.rglob('*') if p.is_file()}})
        write(root/'F/protocol.json', {'resource_proposal': {'call_cap_per_case': 3, 'total_call_cap': 15}})
        journal(root/'F/attempts/884825602-interrupted/calls.jsonl', [1, 2])
        write(root/'F/prior_terminal.json', {'completed_cases': [{'seed': int(s)} for s in FROZEN_HISTORIES if s != '884825602'],
                                           'complete_calls': 11, 'partial_journal_calls': 2, 'partial_seed': 884825602, 'aggregate_calls': 13})
        write(root/'F/original_terminal.json', {'calls': 1, 'first_case': {'seed': 27424209}})
        write(root/'F/terminal.json', {'aggregate_calls_including_discarded': 14})
        write(root/'F/execution.json', {'aggregate_calls': 14})

    def test_interrupted_reservations_add_charges_not_persisted_responses(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.fixture(root); result = accounting(root)
            self.assertEqual(result['persisted_complete_responses'], 12)
            self.assertEqual(result['aggregate_charged_attempts'], 14)
            self.assertEqual(result['interrupted_charged_reservations'], 2)
            self.assertEqual(result['prior_aggregate_charged_attempts'], 13)
            self.assertEqual(result['distinct_acquisition_attempts'], 13)
            # Accounting must reject declaring only complete histories' calls.
            write(root/'F/terminal.json', {'aggregate_calls_including_discarded': 12})
            with self.assertRaises(ValueError): accounting(root)

    def test_interrupted_journal_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.fixture(root)
            (root/'F/attempts/884825602-interrupted/calls.jsonl').unlink()
            with self.assertRaises(ValueError): accounting(root)


if __name__ == '__main__':
    unittest.main()

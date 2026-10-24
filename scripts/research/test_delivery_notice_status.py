"""New notice/status boundary fixtures only; no scientific or package execution."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_delivery_publication_release as v

class NoticeStatus(unittest.TestCase):
    def setUp(self):
        self.before = {n: {'sha256': v.digest(n.encode())} for n in
            ('notices/provenance.json', 'notices/ACE_RUNNER_MIT.txt',
             'notices/RUNNER_LICENSE_SCOPE.md', 'B/acceptance.json')}
        self.private = {n: v.digest(n.encode()) for n in
            ('replay_acceptance', 'report_acceptance', 'accounting')}
        self.status = v.notice_status(self.before, self.private)

    def test_current_status_and_evidence_links(self):
        v.validate_notice_status(self.status, self.before, self.private)
        self.assertEqual(self.status['historical_notice']['remaining_selector'], '/remaining')
        self.assertTrue(self.status['historical_notice']['remaining_is_historical'])
        self.assertEqual(self.status['saved_checkpoint_replay']['manifest_sha256'], v.CANDIDATE_SHA)
        self.assertEqual(self.status['reports']['private_acceptance_sha256'], self.private['report_acceptance'])
        self.assertFalse(self.status['strictly_anonymous'])
        self.assertFalse(self.status['anonymous_original_training_qualified'])

    def test_stale_and_overstated_status_rejected(self):
        changes = [('strictly_anonymous', True), ('strictly_anonymous', 0),
                   ('human_anonymity_handling_approved', True),
                   ('human_release_submission_approved', True),
                   ('anonymous_original_training_qualified', True)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(self.status); bad[key] = value
                with self.assertRaises(ValueError): v.validate_notice_status(bad, self.before, self.private)
        for key in ('historical_notice', 'reports', 'accounting'):
            with self.subTest(missing=key):
                bad = copy.deepcopy(self.status); bad.pop(key)
                with self.assertRaises(ValueError): v.validate_notice_status(bad, self.before, self.private)
        bad = copy.deepcopy(self.status); bad['historical_notice']['sha256'] = 'f'*64
        with self.assertRaises(ValueError): v.validate_notice_status(bad, self.before, self.private)

    def test_four_original_notice_and_acceptance_binding_edges(self):
        contract = {'artifacts': {}, 'notice_status': self.status}
        bindings = v.publication_bindings(contract)
        self.assertEqual(len(bindings), 5)  # inherited source contract plus four new edges
        for edge in bindings[1:]:
            value = v.pointer(contract, edge['pointer'])
            self.assertEqual(value, self.before[edge['artifact']]['sha256'])
            self.assertEqual(edge['record'], v.CONTRACT)
        self.assertEqual(len({e['artifact'] for e in bindings[1:]}), 4)

if __name__ == '__main__': unittest.main()

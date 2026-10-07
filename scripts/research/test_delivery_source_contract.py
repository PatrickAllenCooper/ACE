"""Focused custody and truthful-interface failure tests; no study execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from extend_delivery_source_contract import make_contract, snapshot_entry, extend, ROLES, INTERFACES


class ContractTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for n, name in enumerate(ROLES):
            rows.append(dict(path=name, sha256=f'{min(n,21):064x}', stage=name[0],
                protocol_field='fixture', source_revision='fixture', transformation='identity',
                public_release_approved=False, known_identifier_screen_positive=False))
        inv = dict(files=rows, worker_bindings=24)
        snapshots = {n:b'fixture adapter' for n in INTERFACES}
        snapshots.update({n:b'{}' for n in ['A/protocol.json','C/protocol.json','F/protocol.json','F/runtime.json']})
        snapshots['notices/provenance.json'] = json.dumps(dict(runner_grant_authority_resolved=True,public_release_approved=False)).encode()
        snapshots['notices/ACE_RUNNER_MIT.txt'] = b'fixture license'
        snapshots['notices/ACE_APACHE_2_0.txt'] = b'fixture ACE license'
        files = [dict(path=n, original_sha256=hashlib.sha256(v).hexdigest(),
                      role=INTERFACES[n][0] if n in INTERFACES else 'fixture') for n,v in snapshots.items()]
        return dict(files=files), inv, snapshots

    def contract(self, plan, inv, snapshots):
        with patch('extend_delivery_source_contract.RUNNER_LICENSE_SHA', hashlib.sha256(snapshots['notices/ACE_RUNNER_MIT.txt']).hexdigest()), patch('extend_delivery_source_contract.ACE_LICENSE_SHA', hashlib.sha256(snapshots['notices/ACE_APACHE_2_0.txt']).hexdigest()):
            return make_contract(plan, inv, snapshots)

    def test_exact_edges_and_truthful_roles(self):
        p,i,s = self.fixture(); c=self.contract(p,i,s)
        self.assertEqual([r['original_location'] for r in c['bindings']], list(ROLES))
        self.assertEqual(len(c['interfaces']),5)
        self.assertFalse(c['training_reproduction_available'])
        self.assertFalse(c['B_included'])
        self.assertTrue(all(r['released_original_path'] is None for r in c['bindings']))
        self.assertIn('--trust-original-classical-pickles',c['interfaces'][1]['additional_full_replay_arguments'])

    def test_omitted_or_repeated_guard_edge_rejected(self):
        p,i,s=self.fixture(); i['files'][3]=copy.deepcopy(i['files'][1])
        with self.assertRaisesRegex(ValueError,'exact 24'):
            self.contract(p,i,s)

    def test_original_worker_copy_requires_new_review(self):
        p,i,s=self.fixture();p['files'].append(dict(path='unreviewed.py',original_sha256=i['files'][0]['sha256'],role='fixture'))
        with self.assertRaisesRegex(ValueError,'unexpectedly included'):
            self.contract(p,i,s)

    def test_unresolved_notice_or_approved_publication_rejected(self):
        for fields in [dict(runner_grant_authority_resolved=False,public_release_approved=False),dict(runner_grant_authority_resolved=True,public_release_approved=True)]:
            p,i,s=self.fixture();s['notices/provenance.json']=json.dumps(fields).encode()
            with self.assertRaisesRegex(ValueError,'resolved owner'):
                self.contract(p,i,s)

    def test_wrong_adapter_role_rejected(self):
        p,i,s=self.fixture();p['files'][1]['role']='wrong-adapter-role'
        with self.assertRaisesRegex(ValueError,'role mismatch'):
            self.contract(p,i,s)

    def test_B_or_training_inclusion_rejected_without_reading_outcomes(self):
        for name, role in [('B/scores.json','fixture'), ('delivery_prospective_design.py','original-safe-helper'),('extra.py','original-fit-worker')]:
            p,i,s=self.fixture();p['files'].append(dict(path=name,role=role,original_sha256='fixture'))
            with self.assertRaisesRegex(ValueError,'successor|unsupported'):
                self.contract(p,i,s)

    def test_unknown_base_pin_rejected_before_any_path_read(self):
        with self.assertRaisesRegex(ValueError,'reviewed candidate12'):
            extend('nonexistent.json','unknown','nonexistent','nonexistent','nonexistent','nonexistent')

    def test_replaced_source_does_not_authenticate_executing_code(self):
        original=Path(__file__).with_name('extend_delivery_source_contract.py').read_bytes()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'prepare.py'
            old_code=compile(original,str(path),'exec',dont_inherit=True)
            path.write_bytes(original.replace(b'Bind source dispositions',b'Altered source dispositions',1))
            module=types.ModuleType('isolated_source_contract');module.__file__=str(path)
            with self.assertRaisesRegex(ValueError,'executing implementation'):
                exec(old_code,module.__dict__)

    def test_projected_runtime_bytes_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'original.json'
            source.write_text('{"private_path":"private fixture", "z":2,"a":1}')
            entry=dict(path='A/protocol.json', sources=[str(source)],
                original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                transform=dict(kind='project-json',keys=['z','a']))
            self.assertEqual(snapshot_entry(entry),b'{\n  "a": 1,\n  "z": 2\n}\n')
            entry['path']='replay_delivery_attribution_release.py'
            with self.assertRaisesRegex(ValueError,'only recorded'):
                snapshot_entry(entry)
            entry['path']='A/protocol.json'
            source.write_text('{}')
            with self.assertRaisesRegex(ValueError,'custody artifact'):
                snapshot_entry(entry)


if __name__=='__main__':
    unittest.main()

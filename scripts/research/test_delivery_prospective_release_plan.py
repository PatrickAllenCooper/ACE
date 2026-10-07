"""Focused preparation rejection and snapshot tests; no real Stage B data.

Synthetic acceptance metadata exercises the unchanged original gate. These
tests deliberately do not construct a qualified full release or run inference.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import extend_delivery_prospective_release_plan as planner
import prepare_delivery_prospective_release_core as core
import test_delivery_prospective_release_core as fixtures


def put(path, value):
    path.write_text(json.dumps(value))


class ProspectiveReleasePlanTests(unittest.TestCase):
    def fixture(self, directory):
        """Only gate metadata, plus a deliberately incomplete inventory."""
        root = Path(directory).resolve()
        _, supervisor, accepted = fixtures.ReleaseCoreTests().fixture(root)
        reg = json.loads((root/'registration.json').read_text())
        reg.update(worlds=planner.WORLDS, histories=planner.HISTORIES,
                   cells=[list(a) for a in planner.ARMS],
                   source_hashes={f'ace/module{i}.py': 'e'*64 for i in range(19)})
        put(root/'registration.json', reg); rh = core.sha(root/'registration.json')
        complete = json.loads((root/'complete.json').read_text())
        complete['registration_sha256'] = rh; put(root/'complete.json', complete)
        accepted.update(study_registration_sha256=rh, complete_sha256=core.sha(root/'complete.json'))
        put(root/'acceptance.json', accepted)
        supervisor.update(registration_sha256=rh, acceptance_sha256=core.sha(root/'acceptance.json'))
        put(root/'audit_execution.json', supervisor)
        inventory = {'schema': 'delivery-prospective-composite-v1', 'completed': False,
                     'conflict_policy': 'reject', 'conflicts': [], 'missing_indices': list(range(560, 640)),
                     'cells_verified': 560, 'registration_sha256': rh, 'matrix_sha256': 'f'*64, 'files': []}
        put(root/'inventory.json', inventory)
        args = dict(plan_file=root/'absent-plan.json', prospective=root, inventory=root/'inventory.json',
                    source_inventory=root/'absent-source-inventory.json', source_root=root/'absent-source',
                    core_file=root/'absent-core.py', core_contract=root/'absent-contract.json',
                    expected_core_contract_sha256='a'*64, private_dir=root.parent/(root.name+'-private'),
                    out=root/'new-plan.json', expected_registration=rh)
        return root, inventory, args, accepted, supervisor

    def assert_outcomes_unopened(self, args):
        original = Path.open; opened = []
        def checked(path, mode='r', *a, **kw):
            if 'r' in mode:
                opened.append(path.name)
                if path.name == 'scores.json' or 'evaluation' in path.parts or path.suffix == '.npz':
                    raise AssertionError('scientific artifact opened before complete custody')
            return original(path, mode, *a, **kw)
        with patch.object(Path, 'open', checked):
            with self.assertRaises((ValueError, FileNotFoundError)):
                planner.extend(**args)
        self.assertNotIn('scores.json', opened)
        self.assertFalse(args['private_dir'].exists())
        self.assertFalse(args['out'].exists())

    def test_original_gate_is_first_operation(self):
        with patch.object(planner.core, 'accepted_before_scores', side_effect=ValueError('blocked')) as gate:
            with patch.object(planner, 'snapshot', side_effect=AssertionError('read before gate')):
                with self.assertRaisesRegex(ValueError, 'blocked'):
                    planner.extend('plan', 'study', 'inventory', 'sources', 'source-root',
                                   'core', 'contract', 'a'*64, 'private', 'out')
        gate.assert_called_once_with('study', core.REGISTRATION_SHA)

    def test_missing_original_completion_precedes_every_other_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(planner, 'snapshot', side_effect=AssertionError('planner read before gate')):
                with self.assertRaises(FileNotFoundError):
                    planner.extend(root/'missing-plan', root, root/'missing-inventory', root/'missing-source',
                                   root, root/'missing-core', root/'missing-contract', 'a'*64,
                                   root/'private', root/'out')

    def test_partial_acceptance_rejects_before_inventory_or_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, args, accepted, supervisor = self.fixture(directory)
            accepted['checkpoints_replayed'] = 560; put(root/'acceptance.json', accepted)
            supervisor['acceptance_sha256'] = core.sha(root/'acceptance.json')
            put(root/'audit_execution.json', supervisor)
            with patch.object(planner, 'snapshot', side_effect=AssertionError('partial acceptance passed')):
                self.assert_outcomes_unopened(args)

    def test_accepted_metadata_still_rejects_560_inventory_before_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, args, _, _ = self.fixture(directory)
            self.assert_outcomes_unopened(args)

    def test_inventory_completion_policy_counts_and_conflicts_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, inv, args, _, _ = self.fixture(directory)
            valid = {**inv, 'completed': True, 'cells_verified': 640, 'missing_indices': []}
            for key, value in [('completed', 1), ('cells_verified', True), ('cells_verified', 560),
                               ('conflict_policy', 'pick-first'), ('conflicts', ['different copy']),
                               ('missing_indices', [639]), ('registration_sha256', 'b'*64)]:
                with self.subTest(key=key, value=value):
                    put(root/'inventory.json', {**valid, key: value})
                    self.assert_outcomes_unopened(args)

    def test_conflicting_second_copy_rejects_before_score_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root, inv, args, _, _ = self.fixture(directory)
            a, b = root/'copy-one.bin', root/'copy-two.bin'
            a.write_bytes(b'original'); b.write_bytes(b'conflicting')
            inv.update(completed=True, cells_verified=640, missing_indices=[], files=[
                {'path': 'B/custody.bin', 'sha256': planner.sha(a), 'sources': [str(a), str(b)]}])
            put(root/'inventory.json', inv); self.assert_outcomes_unopened(args)

    def test_copy_checks_all_claims_and_rejects_missing_digest_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); a, b = root/'a', root/'b'
            a.write_bytes(b'fixed bytes'); b.write_bytes(a.read_bytes()); h = planner.sha(a)
            self.assertEqual(planner.checked_copies([str(a), str(b)], h), a)
            b.unlink()
            for sources, digest in [([str(a), str(b)], h), ([str(a)], 'bad'), ([], h), ([1], h)]:
                with self.subTest(sources=sources):
                    with self.assertRaises(ValueError): planner.checked_copies(sources, digest)
            b.symlink_to(a)
            with self.assertRaises(ValueError): planner.checked_copies([str(a), str(b)], h)

    def test_relative_inventory_aliases_reject_before_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root, inv, args, _, _ = self.fixture(directory)
            a = root/'copy.bin'; a.write_bytes(b'fixed bytes')
            for name in ('../escape', '/absolute', 'B/a/../b', 'B//b', 'B\\b'):
                with self.subTest(name=name):
                    inv.update(completed=True, cells_verified=640, missing_indices=[], files=[
                        {'path': name, 'sha256': planner.sha(a), 'sources': [str(a)]}])
                    put(root/'inventory.json', inv); self.assert_outcomes_unopened(args)

    def test_metadata_snapshot_hash_and_parse_share_captured_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'metadata.json'; raw = b'{"status":"original"}'
            path.write_bytes(raw); h = hashlib.sha256(raw).hexdigest()
            original = Path.read_bytes
            def mutate_after_read(p):
                data = original(p)
                if p == path: p.write_bytes(b'{"status":"changed"}')
                return data
            with patch.object(Path, 'read_bytes', mutate_after_read):
                value, digest, captured = planner.snapshot(path, h)
            self.assertEqual(value, {'status': 'original'})
            self.assertEqual((digest, captured), (h, raw))
            with self.assertRaises(ValueError): planner.snapshot(path, h)

    def test_duplicate_nonfinite_and_oversized_metadata_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'metadata.json'
            for raw in (b'{"count":640,"count":560}', b'{"value":NaN}', b'{"value":Infinity}'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError): planner.snapshot(path)
            path.write_bytes(b'{"value":640}')
            with patch.object(planner, 'MAX_JSON_BYTES', 4):
                with self.assertRaises(ValueError): planner.snapshot(path)

    def test_execution_rejects_boolean_and_invalid_measurements(self):
        reg = {'_sha': 'a'*64, 'resources': {'rss_bytes': 1024, 'audit_wall_seconds': 10}}
        value = {'status': 'complete', 'exit_code': 0, 'registration_sha256': 'a'*64,
                 'account': 'ucb736_asc1', 'peak_tree_rss_bytes': 512, 'elapsed_seconds': 2.0}
        planner.execution(value, reg, 'audit_wall_seconds')
        for key, bad in [('exit_code', False), ('peak_tree_rss_bytes', True),
                         ('peak_tree_rss_bytes', -1), ('peak_tree_rss_bytes', 2048),
                         ('elapsed_seconds', float('nan')), ('elapsed_seconds', float('inf')),
                         ('elapsed_seconds', False), ('elapsed_seconds', 11)]:
            with self.subTest(key=key, bad=bad):
                with self.assertRaises(ValueError): planner.execution({**value, key: bad}, reg, 'audit_wall_seconds')


if __name__ == '__main__':
    unittest.main()

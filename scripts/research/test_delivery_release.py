"""Custody risks: conflicting snapshots, relocation, metadata and tampering."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_delivery_release import build, reconcile, write
from verify_delivery_release import sha, relative, verify, identifying_bytes
from prepare_delivery_release_plan import require_acceptance


class ReleaseTests(unittest.TestCase):
    def test_conflicting_snapshot_is_not_silently_selected(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td)/'a', Path(td)/'b'
            a.write_bytes(b'checkpoint'); b.write_bytes(b'different')
            with self.assertRaises(ValueError): reconcile([a, b], sha(a))
            b.write_bytes(a.read_bytes())
            self.assertEqual(reconcile([a, b], sha(a)), a)

    def test_traversal_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'package'; root.mkdir()
            for name in ('../outside', '/absolute', 'a/../b', 'a//b', 'a\\b'):
                with self.assertRaises(ValueError): relative(root, name)
            (root/'link').symlink_to(Path(td), target_is_directory=True)
            with self.assertRaises(ValueError): relative(root, 'link/file')

    def test_projection_relocation_bindings_and_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); original = base/'original.json'; model = base/'model.bin'
            model.write_bytes(b'unchanged checkpoint bytes')
            write(original, {'private_location': '/Users/private/project', 'model_sha256': sha(model)})
            tool = Path(__file__).with_name('verify_delivery_release.py')
            plan = {'status': 'preparation only', 'files': [
                {'path': 'receipt.json', 'sources': [str(original)], 'role': 'receipt',
                 'original_sha256': sha(original), 'transform': {'kind': 'project-json', 'keys': ['model_sha256']}},
                {'path': 'model.bin', 'sources': [str(model)], 'role': 'model', 'original_sha256': sha(model)},
                {'path': 'verify_delivery_release.py', 'sources': [str(tool)], 'role': 'verification-tool', 'original_sha256': sha(tool)}],
                'bindings': [{'record': 'receipt.json', 'pointer': '/model_sha256', 'artifact': 'model.bin', 'digest': 'sha256'}]}
            write(base/'plan.json', plan)
            result = build(base/'plan.json', base/'release', base/'private.json')
            self.assertFalse(result['pin_origin_verified'])
            self.assertEqual(sha(original), plan['files'][0]['original_sha256'])
            relocated = base/'relocated'; (base/'release').rename(relocated)
            completed = subprocess.run([sys.executable, str(relocated/'verify_delivery_release.py'),
                                       '--expected-manifest-sha256', result['manifest_sha256']],
                                      cwd='/', capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(completed.stdout)['bindings_verified'], 1)
            (relocated/'model.bin').write_bytes(b'tampered')
            with self.assertRaises(ValueError): verify(relocated, result['manifest_sha256'])

    def test_json_escaped_identifiers_are_screened_after_decoding(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'metadata.json'
            p.write_text(r'{"path":"\/Users\/private\/study"}')
            self.assertTrue(identifying_bytes(p))
            p.write_text(r'{"account":"\u0075cb736_asc1"}')
            self.assertTrue(identifying_bytes(p))

    def test_verification_role_cannot_exempt_arbitrary_script(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); source = base/'fake.py'; source.write_text('account="ucb736_asc1"\n')
            write(base/'plan.json', {'status': 'preparation', 'bindings': [], 'files': [
                {'path': 'verify_delivery_release.py', 'role': 'verification-tool',
                 'sources': [str(source)], 'original_sha256': sha(source)}]})
            with self.assertRaises(ValueError): build(base/'plan.json', base/'release', base/'private.json')

    def test_original_digest_binding_is_distinct_from_derived_digest(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); original = base/'protocol.json'; fit = base/'fit.json'
            write(original, {'source': '/Users/private/study', 'epochs': 100})
            write(fit, {'protocol_sha256': sha(original)})
            entries = [
                {'path': 'protocol.json', 'sources': [str(original)], 'role': 'protocol',
                 'original_sha256': sha(original), 'transform': {'kind': 'project-json', 'keys': ['epochs']}},
                {'path': 'fit.json', 'sources': [str(fit)], 'role': 'receipt', 'original_sha256': sha(fit)}]
            binding = {'record': 'fit.json', 'pointer': '/protocol_sha256', 'artifact': 'protocol.json', 'digest': 'sha256'}
            write(base/'bad-plan.json', {'status': 'preparation', 'files': entries, 'bindings': [binding]})
            with self.assertRaises(ValueError): build(base/'bad-plan.json', base/'bad', base/'bad-private.json')
            binding['digest'] = 'original_sha256'
            write(base/'good-plan.json', {'status': 'preparation', 'files': entries, 'bindings': [binding]})
            result = build(base/'good-plan.json', base/'good', base/'good-private.json')
            self.assertEqual(result['bindings_verified'], 1)

    def test_planner_rejects_changed_accepted_receipt_set(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); a = base/'A'; c = base/'C'; a.mkdir(); c.mkdir()
            write(a/'registration.json', {'frozen': True})
            write(a/'complete.json', {'complete': True})
            write(base/'gate.json', {'complete_receipt_sha256': sha(a/'complete.json'),
                                    'custody_audit': {'full_acceptance': True,
                                     'registration_sha256': sha(a/'registration.json'),
                                     'verified_receipt_hashes_sha256': 'accepted-set'}})
            write(base/'physical.json', {'full_acceptance': True, 'receipt_hashes': {}})
            with patch('audit_delivery_attribution.audit', return_value={'verified_receipt_hashes_sha256': 'different-set'}) as audit:
                with self.assertRaises(ValueError): require_acceptance(a, c, base/'gate.json', base/'physical.json')
                audit.assert_called_once_with(a, require_complete=True)

    def test_identifying_fields_cannot_leak_as_unmodified_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); source = base/'source.json'; write(source, {'account': 'ucb736_asc1'})
            write(base/'plan.json', {'status': 'preparation', 'bindings': [], 'files': [
                {'path': 'receipt.json', 'role': 'receipt', 'sources': [str(source)], 'original_sha256': sha(source)}]})
            with self.assertRaises(ValueError): build(base/'plan.json', base/'release', base/'private.json')
            self.assertFalse((base/'private.json').exists())


if __name__ == '__main__':
    unittest.main()

"""Only new diagnostic cleanup/source-evidence fixtures. No ML imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('diagnostic_under_test', Path(__file__).with_name('probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Corrections(unittest.TestCase):
    def test_already_exited_timeout_race_is_retained(self):
        proc = Mock(pid=4242)
        proc.wait.return_value = 1
        with patch.object(probe.os, 'killpg', side_effect=ProcessLookupError):
            result = probe.kill_owned(proc)
        self.assertEqual(result['returncode'], 1)
        self.assertTrue(result['already_gone'])
        self.assertFalse(result['group_still_exists'])
        self.assertEqual(result['errors'], [])

    def test_uncertain_reap_is_recorded(self):
        proc = Mock(pid=4242)
        proc.wait.side_effect = subprocess.TimeoutExpired('fake', 5)
        with patch.object(probe.os, 'killpg', return_value=None):
            result = probe.kill_owned(proc)
        self.assertIsNone(result['returncode'])
        self.assertTrue(result['group_still_exists'])
        self.assertTrue(result['kill_sent'])
        self.assertEqual(len(result['errors']), 1)

    def test_capture_does_not_depend_on_successful_imports(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)/'out'
            out.mkdir()
            source = Path(name)/'fake.py'
            source.write_bytes(b'raise ImportError("artificial failed import")\n')
            distribution = Mock(version='artificial')
            distribution.locate_file.return_value = source
            with patch.object(probe.importlib.metadata, 'distribution', return_value=distribution):
                probe.capture_installed_sources(out)
            records = json.loads((out/'installed_sources.json').read_text())
            self.assertEqual(len(records), 16)
            self.assertTrue(all(v['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
                                for v in records.values()))
            self.assertEqual((out/'installed_sources/scipy/sparse/linalg/_eigen/_svds.py').read_bytes(), source.read_bytes())

    def test_evidence_read_failure_is_separate(self):
        with tempfile.TemporaryDirectory() as name:
            with patch.object(probe.importlib.metadata, 'distribution', side_effect=OSError('artificial unavailable')):
                probe.capture_installed_sources(Path(name))
            records = json.loads((Path(name)/'installed_sources.json').read_text())
            self.assertEqual(len(records), 16)
            self.assertTrue(all(v['status'] == 'evidence_read_failed' for v in records.values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)

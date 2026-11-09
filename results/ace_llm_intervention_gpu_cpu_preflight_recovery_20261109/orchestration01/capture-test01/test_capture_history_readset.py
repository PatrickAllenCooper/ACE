"""New artificial boundary checks. No original study data is read."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('capture_readset',
    Path(__file__).with_name('capture_history_readset.py'))
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class CaptureBoundary(unittest.TestCase):
    def fixture(self, root):
        raw = b'{"fixture": "artificial"}  \n'
        files = {}
        for name in capture.FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            files[name] = capture.sha(raw)
        value = dict(schema='ace-scientific-campaign-cpu-acceptance-v1',
                     status='development_cpu_phase_accepted', scientific_qualified=False,
                     campaign_root='/artificial/original/campaign', campaign_files=files)
        encoded = capture.canonical(value)
        return encoded, capture.sha(encoded), raw

    def test_lossless_selected_capture_and_changed_input_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            acceptance, pin, original = self.fixture(root)
            (root/'unused-tensor.bin').write_bytes(b'unselected')
            value = json.loads(capture.capture(acceptance, pin, root))
            self.assertEqual(set(value['files']), set(capture.FILES))
            self.assertEqual(len(value['files']), 133)
            self.assertTrue(all(v['utf8'].encode() == original for v in value['files'].values()))
            self.assertEqual(value['cpu_acceptance_sha256'], pin)
            (root/capture.FILES[-1]).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'digest differs'):
                capture.capture(acceptance, pin, root)
            with self.assertRaisesRegex(ValueError, 'acceptance digest'):
                capture.capture(acceptance, '0'*64, root)

    def test_links_bounds_and_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            original = root/'source'
            original.write_bytes(b'abc')
            expected = capture.sha(b'abc')
            link = root/'symlink'
            link.symlink_to(original)
            with self.assertRaises(OSError):
                capture.pinned_read(link, expected)
            os.link(original, root/'hardlink')
            with self.assertRaisesRegex(ValueError, 'regular file'):
                capture.pinned_read(original, expected)
            (root/'hardlink').unlink()
            with self.assertRaisesRegex(ValueError, 'regular file'):
                capture.pinned_read(original, expected, maximum=2)
            (root/'parent-alias').symlink_to(root, target_is_directory=True)
            with self.assertRaises(OSError):
                capture.pinned_read(root/'parent-alias'/'source', expected)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                capture.decode(b'{"same":1,"same":2}')


if __name__ == '__main__':
    unittest.main()

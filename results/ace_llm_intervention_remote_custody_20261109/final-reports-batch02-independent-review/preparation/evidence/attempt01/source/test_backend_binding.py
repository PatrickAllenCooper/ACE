"""One new binding-only check; no archive/report/remote/scientific operations."""
from pathlib import Path
import hashlib
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
NEW = '4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f'
OLD = '527d225b8e859c67edb4eb5ca6203e9b38e31aca1e80c8544668ccc09c6dc24b'
EXTENSION = 'f62337123abe2eec7c7624e9c32692309d4469db23d28a8e94902a7e1548837c'


def captured_extension():
    source = (HERE/'batch_reports.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != EXTENSION:
        raise ValueError('independent extension SHA differs before exec')
    module = types.ModuleType('fabricated_batch02_binding')
    exec(compile(source, '<captured-batch02-binding>', 'exec'), vars(module))
    return module, source


class BindingDeltaTest(unittest.TestCase):
    def test_actual_captured_backend_binds_and_wrong_bytes_or_old_pin_reject(self):
        backend = (HERE/'dependencies/remote_custody.py').read_bytes()
        old = (HERE/'predecessor/remote_custody.py').read_bytes()
        self.assertEqual(hashlib.sha256(backend).hexdigest(), NEW)
        self.assertEqual(hashlib.sha256(old).hexdigest(), OLD)
        module, source = captured_extension()
        with patch('subprocess.Popen', side_effect=AssertionError('no child allowed')):
            module.bind_sources(extension_source=source, extension_sha256=EXTENSION,
                                backend_source=backend, backend_sha256=NEW)
        self.assertEqual(module._backend()._BOUND_SOURCE, (backend, NEW))
        self.assertEqual(module.BACKEND_SHA256, NEW)
        self.assertIn("bundle['backend_sha256']=='"+NEW+"'", module.BOOTSTRAP)
        self.assertNotIn(OLD, module.BOOTSTRAP)
        self.assertEqual(module._backend().TIMEOUT, 1800)
        for label, wrong_bytes, wrong_pin in (
                ('tampered captured bytes', backend+b'\n# fabricated tampering\n', NEW),
                ('valid original bytes with obsolete pin', old, OLD)):
            with self.subTest(case=label):
                fresh, captured = captured_extension()
                with patch('builtins.exec', side_effect=AssertionError('backend executed before rejecting binding')):
                    with self.assertRaises(ValueError):
                        fresh.bind_sources(extension_source=captured, extension_sha256=EXTENSION,
                                           backend_source=wrong_bytes, backend_sha256=wrong_pin)
                self.assertIsNone(fresh._BOUND)


if __name__ == '__main__':
    unittest.main(verbosity=2)

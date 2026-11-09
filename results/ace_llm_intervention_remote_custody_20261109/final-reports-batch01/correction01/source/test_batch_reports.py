"""New batch-only synthetic checks. No SSH, actual receipt, scientific JSON or model.

Each fresh captured backend uses test-only project paths for tiny local archives.
The exact original archive/Receiver/verification code runs on fabricated bytes.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent/'originals/remote_custody.py'
EVIDENCE = Path(os.environ['BATCH_TEST_EVIDENCE']).resolve()


def fresh():
    source = (HERE/'batch_reports.py').read_bytes()
    backend = BACKEND.read_bytes()
    m = types.ModuleType('fixture_batch_reports')
    exec(compile(source, str(HERE/'batch_reports.py'), 'exec'), m.__dict__)
    m.bind_sources(extension_source=source, extension_sha256=hashlib.sha256(source).hexdigest(),
                   backend_source=backend, backend_sha256=hashlib.sha256(backend).hexdigest())
    return m


class BatchReportsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix=self._testMethodName+'-', dir=EVIDENCE)).resolve()
        self.m = fresh()
        self.b = b = self.m._backend()
        # Test-only aliases; production backend remains unchanged and source pinned.
        b.PROJECT_PARENT = self.root
        b.PROJECT_BASE = self.root/'projects'
        b.PROJECT_BASE.mkdir(mode=0o700)
        backup = b.PROJECT_BASE/'backup'
        backup.mkdir(mode=0o700)
        payload = {n: b'opaque-not-json\x00\xff:'+n.encode() for n in self.m.report_names()}
        payload['z/unselected.bin'] = b'nonreport binary never selected'
        dirs = sorted({'/'.join(n.split('/')[:i]) for n in payload for i in range(1, len(n.split('/')))})
        self.full = b.normalized(dict(files={n:dict(bytes=len(data), sha256=b.sha(data)) for n,data in payload.items()}, directories=dirs))
        archive = io.BytesIO()
        for n in self.full['directories']:
            archive.write(b.header(n))
        for n, info in self.full['files'].items():
            archive.write(b.header(n, info['bytes']))
            archive.write(payload[n]); archive.write(b'\0'*(-info['bytes'] % 512))
        archive.write(b'\0'*(b.tar_size(self.full)-archive.tell()))
        self.archive = backup/'raw.tar'
        self.archive.write_bytes(archive.getvalue())
        selected, omitted = b.selected_inventory(self.full)
        self.receipt = dict(schema='ace-remote-custody-v1', full_backup_verified=True, full_local_copy=False,
            acceptance=False, retry_allowed=False, helper_sha256=self.m.BACKEND_SHA256,
            selection_policy='current-launch-text-and-namespaces-v1', full_inventory=self.full,
            local_inventory=selected, omitted_files=omitted, omitted_bytes=sum(v['bytes'] for v in omitted.values()),
            backup_root=str(backup), backup_kind='ustar', backup_archive_path=str(self.archive),
            backup_archive_sha256=b.sha(archive.getvalue()), backup_archive_bytes=len(archive.getvalue()))
        (backup/'receipt.json').write_bytes(b.raw(self.receipt))
        (backup/'inventory.json').write_bytes(b.raw(self.full))
        members = {n:self.full['files'][n] for n in self.m.report_names()}
        self.args = dict(receipt=self.receipt, receipt_sha256=b.sha(b.raw(self.receipt)),
            members=members, members_sha256=b.sha(b.raw(members)), finalizer_acceptance_sha256='a'*64,
            finalizer_custody_sha256='c'*64, destination=str(self.root/'reports'),
            helper_path=str(self.root/'fake-never-run-helper'), log_dir=str(self.root/'logs'))
        self.calls = 0
        self.original_tar = b.check_tar
        def counted(*args, **kwargs):
            self.calls += 1
            return self.original_tar(*args, **kwargs)
        b.check_tar = counted
        self.original_binding = b._BOUND_SOURCE
        self.original_bootstrap = b.BOOTSTRAP
        self.original_receipt = b.raw(self.receipt)

    def loopback(self, transform=None):
        original = self.m._specialize
        def specialize(function, **bindings):
            if function is not self.b.transport:
                return original(function, **bindings)
            def transport(request, receiver, helper_path, logs):
                self.b.write_new(logs/'command.json', self.b.raw({'synthetic':True,'remote':False}))
                class Output:
                    def write(_, data):
                        data = transform(data) if transform else data
                        for i in range(0, len(data), 113):
                            receiver.feed(data[i:i+113])
                    def flush(_):
                        pass
                try:
                    self.m._remote_emit(request, Output())
                    receiver.finish()
                finally:
                    self.b.write_new(logs/'transport.json', self.b.raw({'synthetic':True,'passes':self.calls}))
            return transport
        return patch.object(self.m, '_specialize', specialize)

    def test_two_pass_batch_opaque_readback_and_originals_unchanged(self):
        with self.loopback():
            result = self.m.fetch_reports(**self.args)
        self.assertEqual(self.calls, 2)
        self.assertEqual(result['member_count'], 242)
        self.assertFalse(result['acceptance'])
        self.assertFalse(result['full_archive_local_copy'])
        self.assertEqual(self.b.raw(self.receipt), self.original_receipt)
        self.assertEqual(self.b._BOUND_SOURCE, self.original_binding)
        self.assertEqual(self.b.BOOTSTRAP, self.original_bootstrap)
        self.assertEqual(self.b.local_inventory(Path(self.args['destination']))['files'], self.args['members'])
        self.assertEqual(os.stat(self.args['destination']).st_mode & 0o777, 0o700)
        logs = Path(self.args['log_dir'])
        manifest = self.b.decode((logs/'operation-records.json').read_bytes())
        self.assertIn('stdout.bin', manifest['files'])
        self.assertIn('receipt.json', manifest['files'])
        self.assertNotIn('z/unselected.bin', result['local_inventory']['files'])

    def test_wrong_pin_roster_bounds_and_existing_destination_reject_before_transport(self):
        variants = []
        variants.append(dict(self.args, receipt_sha256='0'*64))
        bad = dict(self.args['members']); bad.pop(next(iter(bad)))
        variants.append(dict(self.args, members=bad, members_sha256=self.b.sha(self.b.raw(bad))))
        bad = dict(self.args['members']); key = next(iter(bad)); bad[key] = dict(bad[key], sha256='0'*64)
        variants.append(dict(self.args, members=bad, members_sha256=self.b.sha(self.b.raw(bad))))
        for args in variants:
            with self.subTest(kind=args['receipt_sha256'], count=len(args['members'])):
                with self.assertRaises(ValueError):
                    self.m.fetch_reports(**args)
        # Exercise both fixed capacity bounds with the identical valid inventory.
        with patch.object(self.b, 'LOCAL_MEMBER', 1):
            with self.assertRaisesRegex(ValueError, 'over32MiB'):
                self.m.fetch_reports(**self.args)
        with patch.object(self.b, 'LOCAL_TOTAL', 1):
            with self.assertRaisesRegex(ValueError, 'over512MiB'):
                self.m.fetch_reports(**self.args)
        self.assertFalse(Path(self.args['log_dir']).exists())
        Path(self.args['destination']).mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError, 'exclusive new'):
            self.m.fetch_reports(**self.args)
        self.assertEqual(self.calls, 0)

    def test_authenticated_bootstrap_rejects_each_tampered_source_before_exec(self):
        b = self.b
        original = b.decode(self.m._bundle(b))
        for key in ('extension', 'backend'):
            with self.subTest(key=key):
                marker = self.root/(key+'-executed')
                bad = dict(original)
                malicious = ('from pathlib import Path\nPath('+repr(str(marker))+').write_text("bad")\n').encode()
                bad[key] = base64.b64encode(malicious).decode()
                payload = b.raw(bad)
                proc = subprocess.run([sys.executable, '-I', '-B', '-c', self.m.BOOTSTRAP, b.sha(payload)],
                    input=str(len(payload)).encode()+b'\n'+payload, capture_output=True, timeout=5)
                (self.root/(key+'-stdout.bin')).write_bytes(proc.stdout)
                (self.root/(key+'-stderr.bin')).write_bytes(proc.stderr)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(b'individual source SHA differs before exec', proc.stderr)
                self.assertFalse(marker.exists())
        with self.assertRaisesRegex(ValueError, 'source SHA'):
            self.m.bind_sources(extension_source=self.m._BOUND[0]+b'\n', extension_sha256=self.m._BOUND[1],
                backend_source=self.m._BOUND[2], backend_sha256=self.m._BOUND[3])

    def test_framing_tampering_retains_failure_and_no_success_receipt(self):
        # Malformed frame length fails before ANY selected payload file is opened.
        with self.loopback(lambda data: b'\0'*8 if len(data)==8 else data):
            with self.assertRaisesRegex(ValueError, 'bounded stream metadata'):
                self.m.fetch_reports(**self.args)
        logs = Path(self.args['log_dir'])
        self.assertEqual((logs/'stdout.bin').read_bytes(), b'\0'*8)
        self.assertTrue((logs/'failure.json').is_file())
        self.assertFalse((logs/'receipt.json').exists())
        manifest = self.b.decode((logs/'operation-records.json').read_bytes())
        self.assertIn('failure.json', manifest['files'])
        self.assertEqual(self.calls, 1)

    def test_unselected_archive_corruption_after_verify_preserves_stream_and_rejects(self):
        original_verify = self.b.remote_verify
        def verify_then_corrupt(request):
            verified = original_verify(request)
            data = bytearray(self.archive.read_bytes())
            offset = data.index(b'nonreport binary never selected')
            data[offset] ^= 1
            self.archive.write_bytes(data)
            return verified
        with patch.object(self.b, 'remote_verify', verify_then_corrupt), self.loopback():
            with self.assertRaisesRegex(ValueError, 'archive member digest differs'):
                self.m.fetch_reports(**self.args)
        logs = Path(self.args['log_dir'])
        self.assertEqual(self.calls, 2)
        self.assertEqual(len(self.b.local_inventory(Path(self.args['destination']))['files']), 242)
        self.assertTrue((logs/'stdout.bin').stat().st_size > 0)
        self.assertTrue((logs/'failure.json').is_file())
        self.assertFalse((logs/'receipt.json').exists())
        # No automatic retry, even though all selected report bytes arrived.
        with self.assertRaisesRegex(ValueError, 'exclusive new'):
            self.m.fetch_reports(**self.args)
        self.assertEqual(self.calls, 2)

    def test_receiver_truncated_and_extra_payloads_cannot_finish_successfully(self):
        # New batch framing checks through original Receiver on two tiny opaque files.
        for kind in ('truncated','extra','wrong_hash'):
            with self.subTest(kind=kind):
                root = self.root/kind; root.mkdir(mode=0o700)
                values = {'a.json':b'abc','b.json':b'xyz'}
                allowed = {n:dict(bytes=len(v),sha256=self.b.sha(v)) for n,v in values.items()}
                receiver = self.b.Receiver(allowed,{n:root/n for n in values},lambda result:None)
                frame = self.b.raw(dict(result={'synthetic':True},stream_files=allowed))
                data = len(frame).to_bytes(8,'big')+frame+b'abcxyz'
                data = data[:-1] if kind=='truncated' else data+b'!' if kind=='extra' else data[:-3]+b'bad'
                (root/'supplied-stream.bin').write_bytes(data)
                try:
                    with self.assertRaises(ValueError):
                        receiver.feed(data); receiver.finish()
                finally:
                    receiver.close()

    def test_secondary_inventory_publication_failure_preserves_first_stream_error(self):
        original = self.b.write_new
        def write(path, data):
            if Path(path).name == 'operation-records.json':
                raise OSError('fabricated secondary evidence write failure')
            return original(path, data)
        with patch.object(self.b, 'write_new', write), self.loopback(lambda data: b'\0'*8 if len(data)==8 else data):
            with self.assertRaisesRegex(ValueError, 'bounded stream metadata'):
                self.m.fetch_reports(**self.args)
        logs = Path(self.args['log_dir'])
        evidence = self.b.decode((logs/'evidence-publication-error.json').read_bytes())
        self.assertEqual(evidence['primary_type'], 'ValueError')
        self.assertEqual(evidence['secondary_type'], 'OSError')
        self.assertTrue((logs/'failure.json').exists())
        self.assertFalse((logs/'receipt.json').exists())

    def test_raced_empty_destination_is_rejected_without_adoption(self):
        begin = self.b.begin
        raced = Path(self.args['destination'])
        created = []
        def insert_empty(*args, **kwargs):
            logs = begin(*args, **kwargs)
            raced.mkdir(mode=0o700)
            created.append(raced.stat().st_ino)
            return logs
        with patch.object(self.b, 'begin', insert_empty), self.loopback():
            with self.assertRaises(FileExistsError):
                self.m.fetch_reports(**self.args)
        self.assertEqual(self.calls, 0)
        self.assertEqual(raced.stat().st_ino, created[0])
        self.assertEqual(list(raced.iterdir()), [])
        logs = Path(self.args['log_dir'])
        self.assertTrue((logs/'failure.json').exists())
        self.assertTrue((logs/'operation-records.json').exists())
        self.assertFalse((logs/'receipt.json').exists())

    def test_transfer_error_survives_transcript_flush_fsync_and_close_failures(self):
        Receiver = self.b.Receiver
        class BrokenStream:
            def __init__(self, file):
                self.file, self.flushes = file, 0
            def write(self, data):
                return self.file.write(data)
            def flush(self):
                self.flushes += 1
                self.file.flush()
                if self.flushes > 1:
                    raise OSError('fabricated transcript flush failure')
            def fileno(self):
                raise OSError('fabricated transcript fsync failure')
            def close(self):
                self.file.close()
                raise OSError('fabricated transcript close failure')
        class BrokenTranscriptReceiver(Receiver):
            @property
            def fd(self):
                return self._fd
            @fd.setter
            def fd(self, value):
                self._fd = BrokenStream(value) if value is not None and self.allowed == {} else value
        with patch.object(self.b, 'Receiver', BrokenTranscriptReceiver), self.loopback(lambda data: b'\0'*8 if len(data)==8 else data):
            with self.assertRaisesRegex(ValueError, 'bounded stream metadata'):
                self.m.fetch_reports(**self.args)
        logs = Path(self.args['log_dir'])
        self.assertEqual((logs/'stdout.bin').read_bytes(), b'\0'*8)
        cleanup = self.b.decode((logs/'transcript-cleanup/receiver-cleanup-error.json').read_bytes())
        self.assertEqual(cleanup['primary']['type'], 'ValueError')
        self.assertEqual([x['phase'] for x in cleanup['cleanup_errors']],
                         ['transcript_flush', 'transcript_fsync', 'buffered_file'])
        self.assertFalse((logs/'receipt.json').exists())
        self.assertTrue((logs/'operation-records.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)

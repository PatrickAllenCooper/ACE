"""New fabricated downstream glue checks only; fake archive transport, no remote."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
import compact_inputs as C


class Backend:
    def __init__(self, payload):
        self.payload=payload;self.verify_calls=0;self.fetch_calls=0;self.fail=False;self.wrong=False
    def verify_backup(self, *, receipt, observation, helper_path, log_dir):
        self.verify_calls+=1;Path(log_dir).mkdir()
        view=copy.deepcopy(receipt['full_inventory'])
        if self.wrong:view['files']['launch/inputs/adapter.pt']['sha256']='b'*64
        return dict(full_backup_verified=True,full_inventory=view,local_inventory=receipt['local_inventory'],
                    receipt_sha256=C.sha(C.canonical(receipt)),backup_archive_sha256=receipt['backup_archive_sha256'],
                    helper_sha256=receipt['helper_sha256'],acceptance=False)
    def fetch_exact(self, *, receipt, member, expected_sha256, destination, helper_path, log_dir):
        self.fetch_calls+=1;Path(log_dir).mkdir();Path(log_dir,'intent.json').write_bytes(b'{}\n')
        if self.fail:
            Path(destination).write_bytes(b'partial');raise RuntimeError('fabricated transfer failure')
        Path(destination).write_bytes(self.payload)
        return dict(verified=True,member=member,sha256=C.sha(self.payload),bytes=len(self.payload),
                    destination=str(destination),backup_archive_sha256=receipt['backup_archive_sha256'],
                    helper_sha256=receipt['helper_sha256'],acceptance=False)


class CompactInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='tiny-compact-input-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.a=self.root/'a';self.b=self.root/'b'
        self.text=b'opaque private JSON text not decoded\n';self.binary=b'not-a-tensor\x00\xffsavedopaque'
        for p in (self.a,self.b):
            (p/'launch/inputs').mkdir(parents=True);(p/'launch/inputs/input.json').write_bytes(self.text)
        self.local=C.selected_inventory(self.a);full=copy.deepcopy(self.local)
        full['files']['launch/inputs/adapter.pt']=dict(bytes=len(self.binary),sha256=C.sha(self.binary))
        self.obs=dict(files=full['files'],directories=full['directories'],terminal_gate=True)
        self.receipt=dict(schema='ace-remote-custody-v1',full_backup_verified=True,full_local_copy=False,
                          backup_kind='ustar',backup_root='/projects/paco0228/ACE_evidence/artificial',
                          backup_archive_path='/projects/paco0228/ACE_evidence/artificial/raw.tar',
                          backup_archive_sha256='a'*64,backup_archive_bytes=10240,local_root=str(self.a),
                          full_inventory=full,local_inventory=self.local,
                          omitted_files={'launch/inputs/adapter.pt':full['files']['launch/inputs/adapter.pt']},
                          omitted_bytes=len(self.binary),helper_sha256='c'*64,
                          roots={'launch':'/scratch/alpine/paco0228/artificial'},acceptance=False,retry_allowed=False)
        self.backend=Backend(self.binary)
    def open(self, **changes):
        args=dict(backend=self.backend,receipt_raw=C.canonical(self.receipt),receipt_sha256=C.sha(C.canonical(self.receipt)),
                  observation=self.obs,expected_roots=self.receipt['roots'],helper_sha256='c'*64,
                  local_roots=[self.a,self.b],staging=self.root/'stage',helper_path='FAKE-NO-REMOTE')
        args.update(changes);return C.Inputs(**args)
    def test_selected_and_opaque_fetch_preserve_complete_map_and_partial_mirrors(self):
        gate=self.open();self.assertEqual(gate.read('launch/inputs/input.json',C.sha(self.text)),self.text)
        self.assertEqual(gate.read('launch/inputs/adapter.pt',C.sha(self.binary)),self.binary)
        self.assertEqual(self.backend.fetch_calls,1);self.assertFalse((self.a/'launch/inputs/adapter.pt').exists())
        gate.recheck();self.assertEqual(C.selected_inventory(self.a),self.local);self.assertEqual(C.selected_inventory(self.b),self.local)
        self.assertEqual(gate.full,self.receipt['full_inventory'])
    def test_wrong_independent_pin_and_oversize_reject_before_fetch(self):
        gate=self.open()
        with self.assertRaises(ValueError):gate.read('launch/inputs/adapter.pt','d'*64)
        with self.assertRaises(ValueError):gate.read('launch/inputs/adapter.pt',C.sha(self.binary),len(self.binary)-1)
        self.assertEqual(self.backend.fetch_calls,0)
    def test_unbound_receipt_and_root_reject_before_remote_call(self):
        with self.assertRaises(ValueError):self.open(receipt_sha256='d'*64)
        with self.assertRaises(ValueError):self.open(expected_roots={'launch':'/scratch/wrong'})
        self.assertEqual(self.backend.verify_calls,0);self.assertFalse((self.root/'stage').exists())
    def test_false_partition_rejects_before_remote_call(self):
        self.receipt['omitted_files']={}
        with self.assertRaises(ValueError):self.open()
        self.assertEqual(self.backend.verify_calls,0)
    def test_second_selected_mirror_changed_is_not_accepted_as_full(self):
        (self.b/'launch/inputs/input.json').write_bytes(b'different')
        with self.assertRaises(ValueError):self.open()
        self.assertTrue((self.root/'stage/verify-000').is_dir())
    def test_failed_fetch_preserves_partial_and_blocks_retry(self):
        gate=self.open();self.backend.fail=True
        with self.assertRaises(RuntimeError):gate.read('launch/inputs/adapter.pt',C.sha(self.binary))
        self.assertEqual((self.root/'stage/fetch-000/adapter.pt').read_bytes(),b'partial')
        with self.assertRaises(ValueError):gate.read('launch/inputs/adapter.pt',C.sha(self.binary))
        self.assertEqual(self.backend.fetch_calls,1)
    def test_true_flag_cannot_override_changed_full_backend_inventory(self):
        self.backend.wrong=True
        with self.assertRaises(ValueError):self.open()
    def test_unrequested_or_traversal_member_cannot_be_fetched(self):
        gate=self.open()
        for name in ('launch/inputs/../adapter.pt','cpu/inputs/adapter.pt','launch/inputs/unknown.pt','/launch/inputs/adapter.pt'):
            with self.subTest(name=name),self.assertRaises(ValueError):gate.read(name,C.sha(self.binary))
        self.assertEqual(self.backend.fetch_calls,0)

if __name__=='__main__':unittest.main(verbosity=2)

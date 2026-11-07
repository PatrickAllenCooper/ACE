"""Tiny artificial file fixtures, with no real runtime execution or copying."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import llm_intervention_runtime_archive as a


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = self.root/'source'
        self.src.mkdir()
        (self.src/'bin').mkdir()
        (self.src/'lib').mkdir()
        (self.src/'bin/python3').write_bytes(b'artificial executable bytes\n')
        (self.src/'bin/python3').chmod(0o755)
        (self.src/'bin/python').symlink_to('python3')
        (self.src/'lib/data').write_bytes(bytes(range(256)))
        (self.src/'lib/backlink').symlink_to('../bin/python')
        self.archive = self.root/'packed'

    def packed(self):
        m=a.pack(self.src,self.archive)
        path=self.archive/'manifest.json'
        return m,path,hashlib.sha256(path.read_bytes()).hexdigest()

    def test_roundtrip_byte_oracle_modes_links_and_exclusive_output(self):
        (self.src/'lib').chmod(0o2755)
        m,p,h=self.packed()
        events=[]
        r=a.stage(self.archive/'runtime.tar',p,h,self.root/'staged',progress=events.append)
        self.assertEqual((self.root/'staged/lib/data').read_bytes(),bytes(range(256)))
        self.assertEqual((self.root/'staged/lib/backlink').read_bytes(),b'artificial executable bytes\n')
        self.assertEqual((self.root/'staged/bin/python3').stat().st_mode&0o777,0o755)
        self.assertEqual((self.root/'staged/lib').stat().st_mode&0o7777,0o755)
        self.assertEqual(next(x for x in m['records'] if x['path']=='lib')['source_mode'],0o2755)
        self.assertFalse(r['runtime_qualified'])
        self.assertTrue(events)
        with self.assertRaises(FileExistsError):
            a.stage(self.archive/'runtime.tar',p,h,self.root/'staged')

    def test_independent_manifest_and_archive_pin_before_output(self):
        m,p,h=self.packed()
        with self.assertRaisesRegex(ValueError,'manifest pin'):
            a.stage(self.archive/'runtime.tar',p,'0'*64,self.root/'stage')
        with (self.archive/'runtime.tar').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'archive pin'):
            a.stage(self.archive/'runtime.tar',p,h,self.root/'stage')
        self.assertFalse((self.root/'stage').exists())

    def test_escape_symlink_special_file_and_caps_reject(self):
        bad=self.src/'escape'
        for target in ['/tmp','../outside']:
            bad.symlink_to(target)
            with self.assertRaises(ValueError):a.pack(self.src,self.root/('bad'+str(len(target))))
            bad.unlink()
        os.mkfifo(bad)
        with self.assertRaises(ValueError):a.pack(self.src,self.root/'fifo')
        bad.unlink()
        with patch.object(a,'MAX_BYTES',4):
            with self.assertRaisesRegex(ValueError,'byte cap'):a.pack(self.src,self.root/'bytes')
        with patch.object(a,'MAX_OBJECTS',1):
            with self.assertRaisesRegex(ValueError,'object cap'):a.pack(self.src,self.root/'objects')

    def test_content_and_extra_member_reject_even_with_new_archive_pin(self):
        m,p,h=self.packed()
        changed=dict(m)
        changed['records']=[dict(r) for r in m['records']]
        next(r for r in changed['records'] if r['kind']=='file')['sha256']='0'*64
        q=self.root/'badmanifest.json';q.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'staged bytes'):
            a.stage(self.archive/'runtime.tar',q,hashlib.sha256(q.read_bytes()).hexdigest(),self.root/'badstage')
        with tarfile.open(self.archive/'runtime.tar','a') as t:
            t.addfile(tarfile.TarInfo('extra'))
        m['archive_bytes']=(self.archive/'runtime.tar').stat().st_size
        m['archive_sha256']=a._hash(self.archive/'runtime.tar')
        q2=self.root/'extramanifest.json';q2.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'extra archive'):
            a.stage(self.archive/'runtime.tar',q2,hashlib.sha256(q2.read_bytes()).hexdigest(),self.root/'extra-stage')

    def test_source_changed_during_inventory_retains_failure(self):
        def progress(event):
            if event['phase']=='inventory_complete':
                (self.src/'lib/data').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed before'):
            a.pack(self.src,self.archive,progress=progress)
        self.assertTrue((self.archive/'runtime.tar').exists())
        self.assertFalse((self.archive/'manifest.json').exists())

    def test_malformed_manifest_rejects_ancestor_type_and_duplicate_paths(self):
        m,p,h=self.packed()
        for mutate in [lambda x:x['records'].append(x['records'][-1]),
                       lambda x:x['records'][0].update(path='../escape'),
                       lambda x:x['records'][0].update(kind='symlink',target='lib'),
                       lambda x:x.update(archive_bytes=-1),
                       lambda x:x['records'][0].update(source_mode=0o4755)]:
            bad=json.loads(json.dumps(m));mutate(bad)
            with self.assertRaises(ValueError):a._validate(bad)


if __name__=='__main__':unittest.main()

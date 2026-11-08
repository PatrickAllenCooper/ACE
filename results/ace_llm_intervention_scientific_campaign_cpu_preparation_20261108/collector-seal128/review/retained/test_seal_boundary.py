"""New fabricated metadata fixtures only; no actual custody or collector main."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cpu_collector_seal128',HERE/'pull_terminal.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
ROOT=Path(os.environ['ACE_SEAL_FIXTURE_ROOT'])

class SealBoundary(unittest.TestCase):
    def fixture(self,name,size):
        folder=ROOT/name;folder.mkdir()
        p=folder/'custody-inventory.json';digest=hashlib.sha256()
        with p.open('xb') as f:
            def emit(x):f.write(x);digest.update(x)
            emit(b'{"fabricated_padding":"')
            left=size-len(b'{"fabricated_padding":"')-len(b'"}\n')
            block=b' '*(1024**2)
            while left:
                n=min(left,len(block));emit(block[:n]);left-=n
            emit(b'"}\n');f.flush();os.fsync(f.fileno())
        self.assertEqual(p.stat().st_size,size)
        return p,digest.hexdigest()

    def test_observed_size_class_and_ordinary_limit(self):
        p,pin=self.fixture('observed-72434765',72434765)
        data=c.read_upstream_custody_seal(p,pin)
        self.assertEqual(len(data),72434765)
        self.assertEqual(hashlib.sha256(data).hexdigest(),pin)
        self.assertEqual(set(c.decode(data)),{'fabricated_padding'})
        with self.assertRaisesRegex(ValueError,'bounded coordinator metadata'):
            c.read_metadata(p,pin)

    def test_128mib_plus_one_rejected_before_second_read(self):
        p,pin=self.fixture('oversize-134217729',128*1024**2+1)
        with mock.patch.object(c.os,'open',wraps=c.os.open) as opened:
            with self.assertRaisesRegex(ValueError,'bounded128MiB'):
                c.read_upstream_custody_seal(p,pin)
            # Fingerprint streams the file once; oversized bytes are never buffered/decoded.
            self.assertEqual(opened.call_count,1)

    def test_pin_basename_and_independent_file_guards(self):
        p,pin=self.fixture('tiny-guards',128)
        self.assertEqual(c.read_metadata(p,pin),c.read_upstream_custody_seal(p,pin))
        for wrong in (None,'bad','0'*64):
            with self.subTest(pin=wrong),self.assertRaises(ValueError):c.read_upstream_custody_seal(p,wrong)
        other=p.with_name('other.json');other.write_bytes(p.read_bytes())
        with self.assertRaisesRegex(ValueError,'independently pinned upstream custody seal'):
            c.read_upstream_custody_seal(other,pin)
        linkdir=ROOT/'symlink-guard';linkdir.mkdir();link=linkdir/'custody-inventory.json';link.symlink_to(p)
        with self.assertRaisesRegex(ValueError,'symlink custody forbidden'):
            c.read_upstream_custody_seal(link,pin)
        harddir=ROOT/'hardlink-guard';harddir.mkdir();hard=harddir/'custody-inventory.json';os.link(p,hard)
        with self.assertRaisesRegex(ValueError,'regular independent custody file'):
            c.read_upstream_custody_seal(p,pin)
        with self.assertRaisesRegex(ValueError,'regular independent custody file'):
            c.read_upstream_custody_seal(hard,pin)

if __name__=='__main__':unittest.main(verbosity=2)

"""New custody-seal size boundaries only, fabricated local bytes; never collector main."""
import hashlib,importlib.util,os,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=Path(os.environ['ACE_SEAL_BOUNDARY_EVIDENCE'])
spec=importlib.util.spec_from_file_location('new_seal_collector',HERE/'pull_terminal.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
class SealBoundary(unittest.TestCase):
 def test_actual_size_class_is_admitted_only_as_upstream_seal(self):
  root=OUT/'observed-size';root.mkdir();path=root/'custody-inventory.json'
  size=33854831;head=b'{"fabricated":true}';data=head+b' '*(size-len(head));path.write_bytes(data)
  expected=hashlib.sha256(data).hexdigest();del data
  captured=C.read_upstream_custody_seal(path,expected)
  self.assertEqual(len(captured),size);self.assertEqual(C.decode(captured),{'fabricated':True})
  self.assertEqual(hashlib.sha256(captured).hexdigest(),expected)
  with self.assertRaisesRegex(ValueError,'bounded coordinator metadata'):C.read_metadata(path,expected)
 def test_over64MiB_seal_rejects_before_whole_file_read(self):
  root=OUT/'over-limit';root.mkdir();path=root/'custody-inventory.json'
  with path.open('xb') as f:f.truncate(64*1024**2+1)
  self.assertEqual(path.stat().st_size,C.UPSTREAM_CUSTODY_SEAL_MAX_BYTES+1)
  with self.assertRaisesRegex(ValueError,'bounded64MiB upstream custody seal'):C.read_upstream_custody_seal(path,'0'*64)
if __name__=='__main__':unittest.main(verbosity=2)

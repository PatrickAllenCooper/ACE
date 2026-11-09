"""New boundary delta only; prior seven wiring methods are not run."""
from pathlib import Path
import ast,types,unittest
from unittest.mock import patch
import compact_inputs as C
import test_compact_wiring as F

class Delta(unittest.TestCase):
 def test_two_hop_finalizer_reference_and_independent_pin_conflict(self):
  fixture=F.Wiring();fixture.setUp();self.addCleanup(fixture.doCleanups)
  ctx,s,a,b,training=fixture.fixture()
  raw=C.canonical(dict(path=str(a),custody_inventory_sha256=C.sha(s['storage-seal.json']),
   disposition_sha256=ctx['pins']['cpu-disposition'],raw_inventory_sha256=ctx['pins']['cpu-inventory'],
   recursive_copy_created=False,original_preserved=True))
  ctx['metadata']['gpu-launch-packet']={'new_input_files':{'storage-seal.json':C.sha(s['storage-seal.json'])}}
  chain=C.verify_ancestry(ctx,raw,'gpu');self.assertEqual([v['stage'] for v in chain],['cpu','training'])
  self.assertFalse(chain[0]['all_original_named_bytes_verified']);self.assertTrue(chain[1]['all_original_named_bytes_verified'])
  ctx['metadata']['gpu-launch-packet']['new_input_files']['storage-seal.json']='a'*64
  with self.assertRaisesRegex(ValueError,'ancestor seal pin'):C.verify_ancestry(ctx,raw,'gpu')
 def test_launcher_once_and_git_gates_precede_transport_seal_bound_explicit(self):
  for stage in ('gpu','finalizer'):
   source=(F.HERE/stage/'launch_once.py').read_text()
   bind=source.index('        b.configure_packet_storage(')
   self.assertLess(source.index('existing build/controller/submission attempt'),bind)
   self.assertLess(source.index('committed independently pinned packet review required'),bind)
   text=(F.HERE/stage/'pull_terminal.py').read_text();tree=ast.parse(text)
   nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='read_metadata']
   module=ast.Module(body=nodes,type_ignores=[]);ns={'Path':Path,'need':C.need,'sha':C.sha,'pin':C.pin}
   class FakePath:
    def read_bytes(self):return b'x'
    def __str__(self):return 'fabricated'
   ns['Path']=lambda _:FakePath()
   ns['fingerprint']=lambda *a,**k:dict(bytes=33*1024**2,sha256=C.sha(b'x'))
   exec(compile(module,'captured-read-metadata-only','exec'),ns)
   with self.assertRaisesRegex(ValueError,'bounded'):ns['read_metadata']('fake')
   self.assertEqual(ns['read_metadata']('fake',maximum=128*1024**2),b'x')
   with self.assertRaisesRegex(ValueError,'bounded'):ns['read_metadata']('fake',maximum=129*1024**2)
   # No remote observation or protected resource/accounting function changed.
   oldpath=F.ROOT/'predecessor'/stage/('pull_terminal.py' if stage=='gpu' else 'collector/pull_terminal.py')
   funcs=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)}
   old,new=funcs(oldpath.read_text()),funcs(text)
   self.assertEqual(old['compare_observations'],new['compare_observations'])

if __name__=='__main__':unittest.main(verbosity=2)

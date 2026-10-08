import importlib.util,types,sys,os,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).parent
spec=importlib.util.spec_from_file_location('prep04',P/'prepare_environment.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('qual04',P/'qualify_runtime.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
class FinalClosures(unittest.TestCase):
 def test_descriptor_freeze_and_no_fallback(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);logical=root/'logical';logical.mkdir();(logical/'freeze.json').write_text('{"original":true}');fd=os.open(logical,os.O_RDONLY|os.O_DIRECTORY)
   try:
    actual=root/'actual';logical.rename(actual);logical.mkdir();(logical/'freeze.json').write_text('{"replacement":true}')
    self.assertEqual(json.loads(m.read_fd(fd,'freeze.json')),{'original':True})
    with patch.object(m,'__output_dir_fd__',fd,create=True),patch.object(m,'__logical_output__',str(logical),create=True):
     m.write(logical/'terminal.json',{'failed':True})
     with self.assertRaisesRegex(ValueError,'binding'):m.write(root/'escaped.json',{})
    with patch.object(m,'__output_dir_fd__',fd,create=True),patch.object(m,'__logical_output__',None,create=True):
     with self.assertRaisesRegex(ValueError,'binding'):m.write(logical/'unbound.json',{})
    self.assertTrue((actual/'terminal.json').is_file());self.assertFalse((logical/'terminal.json').exists());self.assertFalse((root/'escaped.json').exists())
   finally:os.close(fd)
 def test_fileless_canonical_native_and_unknown(self):
  with tempfile.TemporaryDirectory() as td:
   prefix=Path(td).resolve();base=prefix/'base';f=prefix/'creator.py';f.write_text('not executed')
   torch=types.ModuleType('torch');torch.__file__=str(f);creator=types.ModuleType('torch._ops');creator.__file__=str(f)
   cls=type('_Ops',(types.ModuleType,),{});creator._Ops=cls;ops=cls('torch.ops');torch.ops=ops;creator.ops=ops
   native=types.ModuleType('native');so=prefix/'native.so';so.write_bytes(b'not loaded');native.__file__=str(so);child=types.ModuleType('native.child');native.child=child
   modules={'torch':torch,'torch._ops':creator,'torch.ops':ops,'native':native,'native.child':child}
   with patch.dict(q.sys.modules,modules,clear=True):
    records=q.loaded_module_records(prefix,base);self.assertTrue(any(r.get('kind')=='native_child' for r in records))
    creator.ops=cls('wrong')
    with self.assertRaisesRegex(AssertionError,'synthetic torch identity'):q.loaded_module_records(prefix,base)
    creator.ops=ops;q.sys.modules['unexplained']=types.ModuleType('unexplained')
    with self.assertRaisesRegex(ValueError,'unexplained fileless'):q.loaded_module_records(prefix,base)
if __name__=='__main__':unittest.main(verbosity=2)

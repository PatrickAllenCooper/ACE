"""Source-derived artificial wrappers; no torch imports or native execution.

ACE_BACKEND_OBSERVATION points to the retained installed-source observation.
Only wrapper class methods are compiled; descriptors and module bodies are
excluded. This checks compatibility/provenance logic, not actual runtime.
"""
import ast
import builtins
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('q',Path(__file__).with_name('qualify_delivery_replay_runtime.py'))
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
PINS=['1667b68264dd4e6e2905d7845b1325819973a0b0678b31dc71d85766e7385de0',
 '74a6a25942eb32b576b9ca3547393a8236b59164d8c2eb5b872dcd862745ddac',
 '445f5081a3b82c21f24ddc01a2a7292672ba0beb61c6ef7a3bb7d7e89702853b',
 '44384600fe467a4f7d5695cb9285ab6846897e3ed14d4d13754b5824c248a2b3',
 '969d5271687e5540861b9e3605eb8fccc0a00eca48570625e12fdd5e40cdd05d',
 'bc3d83b3cd662ad406accac0632d45028fce3dfcb3be10ce6c5dbbea44ffb7e5',
 '04d2fb08dc9395a4cdc6ee0954313c559822bb5ea644ee89a59790d0db3736bf']

class BackendProxies(unittest.TestCase):
 def setUp(self):
  data=json.loads(Path(os.environ['ACE_BACKEND_OBSERVATION']).read_text())['files']
  self.assertEqual([f['sha256'] for f in data],PINS)
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.prefix=Path(self.temp.name).resolve();self.modules={'torch':types.ModuleType('torch')}
  self.pins={};self.originals={}
  for (name,classname),entry in zip(q._BACKEND_CLASSES.items(),data):
   raw=entry['text'].encode();self.assertEqual(q.hashlib.sha256(raw).hexdigest(),entry['sha256'])
   source=self.prefix/'lib/python3.11/site-packages'/Path(*name.split('.'))/'__init__.py'
   source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(raw);self.pins[str(source)]=entry['sha256']
   original=types.ModuleType(name);ctx=vars(original)
   ctx.update(__file__=str(source),__builtins__=vars(builtins),sys=sys,types=types)
   if name!='torch.backends':ctx['PropModule']=vars(self.originals['torch.backends'])['PropModule']
   tree=ast.parse(raw,filename=str(source));classes=[]
   for node in tree.body:
    if isinstance(node,ast.ClassDef) and node.name in (classname,'PropModule'):
     node.body=[n for n in node.body if isinstance(n,ast.FunctionDef)]
     classes.append(node)
   exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec',dont_inherit=True),ctx)
   proxy=ctx[classname](original,name);self.modules[name]=proxy;self.originals[name]=original
   parent,leaf=name.rsplit('.',1);vars(self.modules[parent])[leaf]=proxy
  self.patcher=patch.dict(sys.modules,self.modules);self.patcher.start();self.addCleanup(self.patcher.stop)
  binding=patch.object(q,'_BACKEND_BINDINGS',None);binding.start();self.addCleanup(binding.stop)
  q.capture_backend_import_bindings(self.prefix,self.pins)

 def check(self,name='torch.backends.cudnn'):
  return q.torch_backend_proxy(name,self.modules[name],self.prefix,self.pins)

 def test_seven_source_derived_wrappers_and_inventory(self):
  entries={Path(p).relative_to(self.prefix).as_posix():{'kind':'file','sha256':h} for p,h in self.pins.items()}
  for name in q._BACKEND_CLASSES:
   with self.subTest(name=name):
    r=self.check(name);self.assertEqual(r['kind'],'python_delegated_backend')
    self.assertFalse(r['native_execution_attested']);q.bind_loaded_inventory([r],self.prefix,entries)

 def test_missing_extra_changed_pins_reject(self):
  old=self.pins
  for pins in ({},{**old,'/unapproved':'0'*64},{p:'0'*64 for p in old}):
   with self.subTest(pins=pins),self.assertRaises(ValueError):self.pins=pins;self.check()
  self.pins=old

 def test_identity_parent_delegate_and_instance_overrides_reject(self):
  proxy=self.modules['torch.backends.cudnn'];root=self.modules['torch.backends']
  for target,key,new in ((vars(root),'cudnn',types.ModuleType('other')),
                         (vars(proxy),'m',types.ModuleType('other')),
                         (vars(proxy),'__file__','/external.py')):
   old=target.copy();target[key]=new
   with self.assertRaises(ValueError):self.check()
   target.clear();target.update(old)

 def test_live_method_and_class_override_reject(self):
  klass=type(self.modules['torch.backends.cudnn']);method=klass.__init__
  klass.__init__=lambda *a:None
  with self.assertRaises(ValueError):self.check()
  klass.__init__=method;klass.unapproved=object()
  with self.assertRaisesRegex(ValueError,'class override'):self.check()

 def test_base_method_globals_and_closure_reject(self):
  ctx=vars(self.originals['torch.backends']);prop=ctx['PropModule'];method=prop.__getattr__
  prop.__getattr__=lambda *a:None
  with self.assertRaisesRegex(ValueError,'method'):self.check()
  prop.__getattr__=method;ctx['super']=lambda *a:None
  with self.assertRaisesRegex(ValueError,'super'):self.check()
  del ctx['super'];prop.__init__.__closure__[0].cell_contents=types.ModuleType
  with self.assertRaisesRegex(ValueError,'closure'):self.check()

 def test_directory_alias_accepted_leaf_symlink_rejected(self):
  alias=self.prefix/'alias';alias.symlink_to(self.prefix/'lib',target_is_directory=True)
  pins={p.replace(str(self.prefix/'lib'),str(alias)):h for p,h in self.pins.items()}
  q.torch_backend_proxy('torch.backends.cudnn',self.modules['torch.backends.cudnn'],self.prefix,pins)
  source=Path(next(p for p in self.pins if '/cudnn/' in p));raw=source.read_bytes()
  other=source.with_name('other.py');other.write_bytes(raw);source.unlink();source.symlink_to(other)
  with self.assertRaises(ValueError):self.check()

 def test_changed_bytes_and_unknown_fileless_reject(self):
  source=Path(next(iter(self.pins)));source.write_bytes(source.read_bytes()+b'\n# changed')
  with self.assertRaisesRegex(ValueError,'bytes'):self.check()
  with patch.dict(sys.modules,{'unknown_one':types.ModuleType('unknown_one'),'unknown_two':types.ModuleType('unknown_two')},clear=True):
   with self.assertRaisesRegex(ValueError,'unknown_one, unknown_two'):
    q.loaded_module_records(self.prefix,self.prefix,{},backend_pins=self.pins)

if __name__=='__main__':unittest.main()

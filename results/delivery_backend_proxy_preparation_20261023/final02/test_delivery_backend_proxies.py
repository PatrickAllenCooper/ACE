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
  native=types.ModuleType('torch._C');native._has_mkldnn=True;self.modules['torch._C']=native;self.modules['torch']._C=native
  for (name,classname),entry in zip(q._BACKEND_CLASSES.items(),data):
   raw=entry['text'].encode();self.assertEqual(q.hashlib.sha256(raw).hexdigest(),entry['sha256'])
   source=self.prefix/'lib/python3.11/site-packages'/Path(*name.split('.'))/'__init__.py'
   source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(raw);self.pins[str(source)]=entry['sha256']
   original=types.ModuleType(name);ctx=vars(original)
   ctx.update(__path__=[str(source.parent)],__file__=str(source),__builtins__=vars(builtins),sys=sys,types=types)
   if name!='torch.backends.opt_einsum':ctx['torch']=self.modules['torch']
   if name!='torch.backends':ctx['PropModule']=vars(self.originals['torch.backends'])['PropModule']
   tree=ast.parse(raw,filename=str(source));classes=[]
   for node in tree.body:
    if name=='torch.backends.mkldnn' and isinstance(node,ast.FunctionDef) and node.name=='is_available':classes.append(node)
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
  with self.assertRaisesRegex(ValueError,'method|base class'):self.check()
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

class BackendReviewRegression(BackendProxies):
 # Run these new methods explicitly; inherited prior methods are not repeated.
 def test_delegate_paths_and_lookup_shadow_reject(self):
  ctx=vars(self.originals['torch.backends']);old=ctx['__path__'];ctx['__path__']=['/tmp']
  with self.assertRaisesRegex(ValueError,'package path'):self.check('torch.backends')
  ctx['__path__']=old;ctx['__getattribute__']=lambda name:None
  with self.assertRaisesRegex(ValueError,'lookup override'):self.check('torch.backends')

 def test_mkldnn_effective_helper_torch_native_reject(self):
  ctx=vars(self.originals['torch.backends.mkldnn'])
  for key,value in [('is_available',lambda:False),('torch',types.ModuleType('fake'))]:
   old=ctx[key];ctx[key]=value
   with self.assertRaises(ValueError):self.check('torch.backends.mkldnn')
   ctx[key]=old
  self.modules['torch._C']._has_mkldnn=False
  with self.assertRaisesRegex(ValueError,'effective binding'):self.check('torch.backends.mkldnn')

 def test_nested_child_import_and_all_wrappers_accept(self):
  name='torch.backends.cudnn.rnn';child=types.ModuleType(name)
  sys.modules[name]=child;vars(self.modules['torch.backends.cudnn'])['rnn']=child
  for name in q._BACKEND_CLASSES:self.check(name)
  vars(self.modules['torch.backends.cudnn'])['rnn']=object()
  with self.assertRaisesRegex(ValueError,'override'):self.check()

 def test_absent_and_none_child_bindings_reject(self):
  proxy=self.modules['torch.backends.mkldnn']
  for key in ['is_available','__path__']:
   for registered in [False,True]:
    with self.subTest(key=key,registered=registered):
     if registered:sys.modules['torch.backends.mkldnn.'+key]=None
     vars(proxy)[key]=None
     with self.assertRaisesRegex(ValueError,'instance override'):self.check('torch.backends.mkldnn')
     del vars(proxy)[key];sys.modules.pop('torch.backends.mkldnn.'+key,None)

 def test_inherited_descriptor_and_replaced_base_binding_reject(self):
  prop=vars(self.originals['torch.backends'])['PropModule'];prop.external=property(lambda self:0)
  with self.assertRaisesRegex(ValueError,'base class'):self.check()
  del prop.external;prop.__doc__='changed'
  with self.assertRaisesRegex(ValueError,'base class'):self.check()

if __name__=='__main__':unittest.main()

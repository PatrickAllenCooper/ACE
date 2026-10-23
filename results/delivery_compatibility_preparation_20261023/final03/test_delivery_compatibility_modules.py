"""Pinned pure-six/artificial-Cython checks; no numerical or native imports."""
import importlib.util,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('q',Path(__file__).with_name('qualify_delivery_replay_runtime.py'))
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

class Compatibility(unittest.TestCase):
 def setUp(self):
  observation=json.loads(Path(os.environ['ACE_COMPAT_OBSERVATION']).read_text())
  entry=observation['files'][0];raw=entry['text'].encode()
  self.assertEqual(q.hashlib.sha256(raw).hexdigest(),'c51c91f703d3d4b3696c923cb5fec213e05e75d9215393befac7f2fa6a3904df')
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.prefix=Path(self.temp.name).resolve()
  self.source=self.prefix/'lib/python3.11/site-packages/six.py';self.source.parent.mkdir(parents=True);self.source.write_bytes(raw)
  self.pins={str(self.source):q.sha(self.source)}
  self.module_patch=patch.dict(sys.modules);self.module_patch.start();self.addCleanup(self.module_patch.stop)
  self.meta_patch=patch.object(sys,'meta_path',list(sys.meta_path));self.meta_patch.start();self.addCleanup(self.meta_patch.stop)
  for n in list(sys.modules):
   if n=='six' or n.startswith('six.') or n=='cython_runtime':del sys.modules[n]
  spec=importlib.util.spec_from_file_location('six',self.source);self.six=importlib.util.module_from_spec(spec);sys.modules['six']=self.six
  exec(compile(raw,str(self.source),'exec',dont_inherit=True),vars(self.six))
  self.moves=__import__('six.moves',fromlist=['moves']);self.cython=types.ModuleType('cython_runtime');sys.modules['cython_runtime']=self.cython
  self.binding_patch=patch.object(q,'_COMPAT_BINDINGS',None);self.binding_patch.start();self.addCleanup(self.binding_patch.stop)
  q.capture_compatibility_bindings(self.prefix,self.pins)
 def check(self,name='six.moves'):
  return q.compatibility_record(name,sys.modules[name],self.prefix,self.pins)
 def test_pinned_six_lazy_cache_and_boolean_runtime(self):
  self.assertEqual(self.check()['kind'],'six_lazy_package')
  self.assertIs(self.moves.range,range)
  self.assertEqual(self.check()['kind'],'six_lazy_package')
  self.cython.cline_in_traceback=False;self.assertFalse(self.check('cython_runtime')['native_creation_attested'])
 def test_source_pin_alias_and_changed_bytes(self):
  alias=self.prefix/'alias';alias.symlink_to(self.source.parent,target_is_directory=True)
  q.compatibility_record('six.moves',self.moves,self.prefix,{str(alias/'six.py'):q.sha(self.source)})
  self.source.write_bytes(self.source.read_bytes()+b'\n# changed')
  with self.assertRaises(ValueError):self.check()
 def test_wrong_identity_importer_and_class_reject(self):
  ctx=vars(self.six);old=ctx['moves'];ctx['moves']=types.ModuleType('six.moves')
  with self.assertRaises(ValueError):self.check()
  ctx['moves']=old;sys.meta_path.remove(ctx['_importer'])
  with self.assertRaises(ValueError):self.check()
 def test_altered_method_helper_and_globals_reject(self):
  ctx=vars(self.six);old=ctx['_import_module'];ctx['_import_module']=lambda n:None
  with self.assertRaises(ValueError):self.check()
  ctx['_import_module']=old;ctx['sys']=object()
  with self.assertRaises(ValueError):self.check()
 def test_paths_and_spec_substitution_reject(self):
  self.moves.__path__=['/unapproved']
  with self.assertRaises(ValueError):self.check()
  del self.moves.__path__;self.moves.__spec__.origin='/external'
  with self.assertRaises(ValueError):self.check()
 def test_cython_inert_contract_and_reference_reject(self):
  for key,value in [('__getattr__',lambda n:None),('__path__',[]),('cline_in_traceback',object())]:
   setattr(self.cython,key,value)
   with self.assertRaises(ValueError):self.check('cython_runtime')
   delattr(self.cython,key)
  sys.modules['cython_runtime']=types.ModuleType('cython_runtime')
  with self.assertRaises(ValueError):self.check('cython_runtime')
 def test_unknown_name_and_missing_capture_reject(self):
  with self.assertRaises(ValueError):self.check('six')
  q._COMPAT_BINDINGS=None
  with self.assertRaises(ValueError):self.check()
 def test_review_swapped_methods_and_instance_hooks_reject(self):
  cls=vars(self.six)['_SixMetaPathImporter'];a,b=cls.find_spec,cls.find_module
  cls.find_spec,cls.find_module=b,a
  with self.assertRaises(ValueError):self.check()
  cls.find_spec,cls.find_module=a,b
  importer=vars(self.six)['_importer'];importer.find_spec=lambda *a,**kw:None
  with self.assertRaises(ValueError):self.check()
  del importer.find_spec
  self.moves.__getattr__=lambda name:None
  with self.assertRaises(ValueError):self.check()
  del self.moves.__getattr__
  self.assertIs(self.moves.range,range);self.assertEqual(self.check()['kind'],'six_lazy_package')

 def test_review_metadata_type_guards_do_not_call_hooks(self):
  importer=vars(self.six)['_importer'];old=importer.known_modules
  class BadMap(dict):
   def get(self,*a):raise AssertionError('untrusted mapping executed')
  importer.known_modules=BadMap(old)
  with self.assertRaises(ValueError):self.check()
  importer.known_modules=old
  cls=vars(self.six)['_MovedItems'];items=cls._moved_attributes
  class BadItem:
   def __getattr__(self,name):raise AssertionError('untrusted descriptor executed')
  cls._moved_attributes=[BadItem()]
  with self.assertRaises(ValueError):self.check()
  cls._moved_attributes=items
  spec=self.moves.__spec__;self.moves.__spec__=BadItem()
  with self.assertRaises(ValueError):self.check()
  self.moves.__spec__=spec;self.assertEqual(self.check()['kind'],'six_lazy_package')

if __name__=='__main__':unittest.main()

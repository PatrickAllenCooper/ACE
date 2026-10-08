"""Actual imported runtime and complete environment byte inventory; no models."""
import argparse,hashlib,importlib,importlib.metadata,json,sys,os,datetime,re,_imp,importlib.machinery
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def loaded_module_records(prefix,base):
 loaded=[]
 for name,module in list(sys.modules.items()):
  if module is None:continue
  attrs=vars(module)
  if '__path__' in attrs:
   for location in attrs['__path__']:
    location=Path(location).resolve();assert location.is_relative_to(prefix) or location.is_relative_to(base/'lib/python3.11') and not location.is_relative_to(base/'lib/python3.11/site-packages'),('external namespace path',name,str(location))
  if name in ('torch.ops','torch.classes'):
   creator_name,klass,attribute={'torch.ops':('torch._ops','_Ops','ops'),'torch.classes':('torch._classes','_Classes','classes')}[name]
   creator=sys.modules[creator_name];assert module is vars(sys.modules['torch'])[attribute] and module is vars(creator)[attribute] and type(module) is vars(creator)[klass],'synthetic torch identity differs'
   backing=Path(creator.__file__).resolve();assert backing.is_relative_to(prefix) and backing.is_file(),'external synthetic backing'
   loaded.append({'module':name,'synthetic':True,'creator_module':creator_name,'backing_file':str(backing),'sha256':sha(backing)});continue
  f=attrs.get('__file__',getattr(type(module),'__file__',None))
  if not f:
   spec=attrs.get('__spec__')
   if name in sys.builtin_module_names and spec is not None and spec.origin=='built-in' and spec.loader is importlib.machinery.BuiltinImporter:
    loaded.append({'module':name,'kind':'builtin','interpreter_sha256':sha(sys.executable)});continue
   if spec is not None and spec.origin=='frozen' and spec.loader is importlib.machinery.FrozenImporter and _imp.is_frozen(name):
    loaded.append({'module':name,'kind':'frozen','interpreter_sha256':sha(sys.executable)});continue
   if spec is not None and isinstance(spec.loader,importlib.machinery.NamespaceLoader) and spec.submodule_search_locations is not None and list(spec.submodule_search_locations)==list(attrs.get('__path__',[])) and list(spec.submodule_search_locations):
    loaded.append({'module':name,'kind':'namespace','locations':[str(Path(x).resolve()) for x in spec.submodule_search_locations]});continue
   if name in ('typing.io','typing.re') and module is vars(sys.modules['typing'])[name.split('.')[1]]:
    backing=Path(sys.modules['typing'].__file__).resolve();assert backing.is_relative_to(base/'lib/python3.11') and not backing.is_relative_to(base/'lib/python3.11/site-packages')
    loaded.append({'module':name,'kind':'typing_alias','backing_file':str(backing),'sha256':sha(backing)});continue
   native=[]
   parts=name.split('.')
   for length in range(len(parts)-1,0,-1):
    parent_name='.'.join(parts[:length]);parent=sys.modules.get(parent_name)
    if parent is None:continue
    backing_file=vars(parent).get('__file__')
    if not backing_file or not any(str(backing_file).endswith(ext) for ext in importlib.machinery.EXTENSION_SUFFIXES):continue
    canonical=parent
    for attribute in parts[length:]:canonical=vars(canonical).get(attribute) if canonical is not None else None
    if canonical is not module:continue
    backing=Path(backing_file).resolve();assert backing.is_relative_to(prefix) and backing.is_file(),('external native child',name)
    native.append({'creator':parent_name,'backing_file':str(backing),'sha256':sha(backing)});break
   if native:
    loaded.append({'module':name,'kind':'native_child','witnesses':native});continue
   # Cython's process-wide ABI module is created by authenticated extensions.
   if re.fullmatch(r'_cython_[0-9]+_[0-9]+_[0-9]+',name):
    cytype=attrs.get('cython_function_or_method');assert isinstance(cytype,type) and cytype.__name__=='cython_function_or_method',('unknown Cython ABI module',name)
    witnesses=[]
    for creator_name,creator in list(sys.modules.items()):
     if creator is None:continue
     origin=vars(creator).get('__file__')
     if not origin or not any(str(origin).endswith(ext) for ext in importlib.machinery.EXTENSION_SUFFIXES):continue
     if not any(type(value) is cytype for value in vars(creator).values()):continue
     backing=Path(origin).resolve();assert backing.is_relative_to(prefix) and backing.is_file(),('external Cython witness',creator_name)
     witnesses.append({'creator':creator_name,'backing_file':str(backing),'sha256':sha(backing)})
    assert witnesses,('unexplained Cython ABI module',name)
    loaded.append({'module':name,'kind':'cython_abi','witnesses':witnesses});continue
   raise ValueError('unexplained fileless module: '+name)

  origin=Path(f).resolve()
  assert origin.is_file() and (origin.is_relative_to(prefix) or origin.is_relative_to(base/'lib/python3.11') and not origin.is_relative_to(base/'lib/python3.11/site-packages') or origin==Path(__file__).resolve()),('external loaded module',name,str(origin))
  loaded.append({'module':name,'path':str(origin),'sha256':sha(origin)})
 return loaded

def main():
 if sys.flags.optimize:raise ValueError('optimized qualification forbidden')
 p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--freeze-sha256',required=True);a=p.parse_args()
 raw=Path(a.freeze).read_bytes();assert hashlib.sha256(raw).hexdigest()==a.freeze_sha256
 reg=json.loads(raw);assert globals().get('__executed_source_sha256__')==reg['qualification_worker_sha256'], 'captured qualifier source differs'
 assert reg['exact_dependencies']=={'torch':'2.9.1','numpy':'2.2.6','scipy':'1.15.3','pandas':'2.3.3','sympy':'1.14.0','PyYAML':'6.0.3'}
 assert reg['rss_bytes']==3*2**30 and reg['wall_seconds']==1800 and reg['slurm_wall_limit']=='00:30:00'
 prefix=Path(reg['environment']).resolve();fd=globals().get('__output_dir_fd__');assert type(fd) is int and os.fstat(fd).st_ino==reg['output_inode'];out=Path('/proc/self/fd/'+str(fd));assert Path(sys.prefix).resolve()==prefix and sys.version_info[:2]==(3,11)
 assert Path(reg['base_python']).resolve(strict=True)==Path(reg['base_python_resolved']).resolve(strict=True) and sha(reg['base_python'])==reg['base_python_sha256'] and sha('/proc/self/exe')==reg['base_python_sha256']
 base=Path(reg['approved_base_prefix']).resolve();assert Path(sys.base_prefix).resolve()==base
 allowed=[base/'lib/python311.zip',base/'lib/python3.11',base/'lib/python3.11/lib-dynload',prefix/'lib/python3.11/site-packages']
 assert all(Path(s).is_absolute() and Path(s).resolve() in allowed for s in sys.path),('unapproved sys.path',sys.path)
 records={}
 for name,v in reg['exact_dependencies'].items():
  ds=[d for d in importlib.metadata.distributions() if d.metadata.get('Name','').lower()==name.lower()]
  assert len(ds)==1 and ds[0].version==v,(name,'metadata ambiguity/mismatch')
  assert Path(ds[0]._path).resolve().is_relative_to(prefix),'external metadata directory'
  module_name='yaml' if name=='PyYAML' else name;m=importlib.import_module(module_name)
  origin=Path(m.__file__).resolve();expected=Path(ds[0].locate_file(module_name+'/__init__.py')).resolve()
  assert origin==expected and origin.is_relative_to(prefix) and m.__version__.split('+',1)[0]==v,(name,'actual module mismatch')
  records[name]={'distribution':ds[0].version,'imported':m.__version__,'origin':str(origin),'metadata_path':str(ds[0]._path),'module_file_sha256':sha(origin)}
 assert all(Path(s).is_absolute() and Path(s).resolve() in allowed for s in sys.path),('post-import unapproved sys.path',sys.path)
 loaded=loaded_module_records(prefix,base)
 files=[]
 for f in sorted(prefix.rglob('*')):
  if f.is_symlink():
   target=f.resolve(strict=True);assert target.is_relative_to(prefix) or target==Path(reg['base_python_resolved']),'unexpected external environment symlink'
   files.append({'path':str(f.relative_to(prefix)),'kind':'symlink','link':os.readlink(f),'resolved':str(target),'target_sha256':sha(target) if target.is_file() else None})
  elif f.is_file():files.append({'path':str(f.relative_to(prefix)),'kind':'file','bytes':f.stat().st_size,'sha256':sha(f)})
  else:assert f.is_dir(),'nonregular environment object'
 inventory=out/'runtime_inventory.json'
 with inventory.open('x') as s:json.dump({'prefix':str(prefix),'files':files},s,indent=2);s.write('\n')
 result={'qualified':True,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.executable,'python_version':sys.version,'python_sha256':sha(sys.executable),'base_python':reg['base_python'],'base_python_sha256':reg['base_python_sha256'],'base_prefix':sys.base_prefix,'sys_path':sys.path,'dependencies':records,'loaded_modules':loaded,'executed_source_sha256':__executed_source_sha256__,'runtime_inventory_sha256':sha(inventory),'objects':len(files),'freeze_sha256':a.freeze_sha256,'new_fits':0,'new_responses':0}
 with (out/'runtime_qualification.json').open('x') as s:json.dump(result,s,indent=2);s.write('\n')
if __name__=='__main__':main()

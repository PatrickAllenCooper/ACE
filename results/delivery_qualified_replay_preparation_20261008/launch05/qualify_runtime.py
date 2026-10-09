"""Derived replay-only runtime qualifier; exact native alias; no installs/models."""
import argparse,hashlib,importlib,importlib.metadata,json,sys,os,datetime,re,_imp,importlib.machinery,types,stat
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def native_autograd_alias(name,module,prefix,pins):
 if name!='torch._C._dynamo.autograd_compiler':raise ValueError('unexpected alias name')
 canonical_pins={}
 for path,pin in pins.items():
  if not Path(path).is_absolute() or not re.fullmatch('[0-9a-f]{64}',pin):raise ValueError('invalid native pin')
  canonical=str(Path(path).resolve(strict=True))
  if canonical in canonical_pins:raise ValueError('duplicate native backing pin')
  canonical_pins[canonical]=pin
 pins=canonical_pins
 torch=sys.modules.get('torch');parent=sys.modules.get('torch._C')
 if type(module) is not types.ModuleType or vars(module).get('__name__')!=name or vars(module).get('__file__'):
  raise ValueError('invalid native alias object')
 if torch is None or type(parent) is not types.ModuleType or vars(torch).get('_C') is not parent:
  raise ValueError('native parent identity differs')
 dynamo=vars(parent).get('_dynamo')
 if dynamo is None or sys.modules.get('torch._C._dynamo') is not dynamo or vars(dynamo).get('compiled_autograd') is not module:
  raise ValueError('native alias identity differs')
 backing=Path(vars(parent).get('__file__','')).resolve()
 if not any(str(backing).endswith(ext) for ext in importlib.machinery.EXTENSION_SUFFIXES):
  raise ValueError('native parent extension required')
 implementation=prefix/'lib/python3.11/site-packages/torch/lib/libtorch_python.so'
 expected={str(backing),str(implementation.resolve())}
 if set(pins)!=expected:raise ValueError('exact native backing pin set required')
 witnesses=[]
 for path in sorted(expected):
  f=Path(path)
  if not f.is_relative_to(prefix) or not f.is_file() or f.is_symlink() or sha(f)!=pins[path]:
   raise ValueError('native backing bytes/origin differ')
  witnesses.append({'creator':'torch._C','backing_file':str(f),'sha256':pins[path]})
 return {'module':name,'kind':'native_alias','attribute_path':'torch._C._dynamo.compiled_autograd','witnesses':witnesses}

def allowed_search_paths(prefix,base):
 return {str(p.resolve()) for p in [base/'lib/python311.zip',base/'lib/python3.11',base/'lib/python3.11/lib-dynload',prefix/'lib/python3.11/site-packages']}
def search_path_check(prefix,base):
 allowed=allowed_search_paths(prefix,base)
 if not all(Path(s).is_absolute() and str(Path(s).resolve()) in allowed for s in sys.path):
  raise ValueError('unapproved current sys.path: '+repr(sys.path))

def dependency_records(reg,reject_cached=True):
 prefix=Path(reg['environment']).resolve();base=Path(reg['approved_base_prefix']).resolve()
 search_path_check(prefix,base)
 roots=['yaml' if n=='PyYAML' else n for n in reg['exact_dependencies']]
 if reject_cached and any(n==r or n.startswith(r+'.') for n in sys.modules for r in roots):
  raise ValueError('fresh dependency imports required')
 records={}
 for name,v in reg['exact_dependencies'].items():
  ds=[d for d in importlib.metadata.distributions() if d.metadata.get('Name','').lower()==name.lower()]
  if len(ds)!=1 or ds[0].version!=v:raise ValueError('metadata ambiguity/mismatch: '+name)
  if not Path(ds[0]._path).resolve().is_relative_to(prefix):raise ValueError('external metadata directory')
  module_name='yaml' if name=='PyYAML' else name;m=importlib.import_module(module_name)
  origin=Path(m.__file__).resolve();expected=Path(ds[0].locate_file(module_name+'/__init__.py')).resolve()
  if origin!=expected or not origin.is_relative_to(prefix) or m.__version__.split('+',1)[0]!=v:
   raise ValueError('actual module mismatch: '+name)
  records[name]={'distribution':ds[0].version,'imported':m.__version__,'origin':str(origin),'metadata_path':str(ds[0]._path),'module_file_sha256':sha(origin)}
 search_path_check(prefix,base)
 return records

def loaded_module_records(prefix,base,native_pins,allowed_sources=None):
 allowed_sources=allowed_sources or {}
 loaded=[]
 for name,module in list(sys.modules.items()):
  if module is None:continue
  attrs=vars(module)
  if name=='torch._C._dynamo.autograd_compiler':
   loaded.append(native_autograd_alias(name,module,prefix,native_pins));continue
  if '__path__' in attrs:
   package_file=Path(attrs['__file__']).resolve() if attrs.get('__file__') else None
   runtime_package=(package_file is not None and (package_file.is_relative_to(prefix) or package_file.is_relative_to(base/'lib/python3.11') and not package_file.is_relative_to(base/'lib/python3.11/site-packages')))
   external_package=(package_file is not None and not runtime_package)
   if external_package:
    assert package_file.name=='__init__.py' and str(package_file) in allowed_sources and sha(package_file)==allowed_sources[str(package_file)],('unapproved external package',name)
   for location in attrs['__path__']:
    location=Path(location).resolve()
    if external_package:assert location==package_file.parent,('external namespace path',name,str(location))
    else:assert location.is_relative_to(prefix) or location.is_relative_to(base/'lib/python3.11') and not location.is_relative_to(base/'lib/python3.11/site-packages'),('external namespace path',name,str(location))
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
  if str(origin) in allowed_sources:
   assert origin.is_file() and sha(origin)==allowed_sources[str(origin)],('authenticated source changed',name)
   loaded.append({'module':name,'path':str(origin),'sha256':allowed_sources[str(origin)]});continue
  assert origin.is_file() and (origin.is_relative_to(prefix) or origin.is_relative_to(base/'lib/python3.11') and not origin.is_relative_to(base/'lib/python3.11/site-packages') or origin==Path(__file__).resolve()),('external loaded module',name,str(origin))
  loaded.append({'module':name,'path':str(origin),'sha256':sha(origin)})
 return loaded

def verify_environment_inventory(prefix,inventory):
 prefix=prefix.resolve()
 if Path(inventory['prefix']).resolve()!=prefix:raise ValueError('inventory prefix differs')
 entries={e['path']:e for e in inventory['files']}
 if len(entries)!=len(inventory['files']):raise ValueError('duplicate inventory path')
 names=set()
 for f in prefix.rglob('*'):
  if f.is_symlink() or f.is_file():names.add(f.relative_to(prefix).as_posix())
  elif not f.is_dir():raise ValueError('nonregular runtime object')
 if names!=set(entries):raise ValueError('live environment membership differs')
 for name,e in entries.items():
  if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('inventory path escapes')
  f=prefix/name
  if e['kind']=='file':
   if f.is_symlink() or not f.is_file() or f.stat().st_size!=e['bytes'] or sha(f)!=e['sha256']:raise ValueError('live environment file changed: '+name)
  elif e['kind']=='symlink':
   if not f.is_symlink() or os.readlink(f)!=e['link'] or f.resolve(strict=True)!=Path(e['resolved']).resolve(strict=True):raise ValueError('live environment link changed: '+name)
   if e['target_sha256'] is None:
    if not f.resolve().is_dir():raise ValueError('live link target kind differs')
   elif not f.resolve().is_file() or sha(f)!=e['target_sha256']:raise ValueError('live link target bytes differ')
  else:raise ValueError('unknown inventory object kind')
 return entries

def bind_loaded_inventory(loaded,prefix,entries):
 prefix=prefix.resolve()
 for record in loaded:
  for witness in [record,*record.get('witnesses',[])]:
   name=witness.get('backing_file',witness.get('path'))
   if name and Path(name).resolve().is_relative_to(prefix):
    relative=Path(name).resolve().relative_to(prefix).as_posix();e=entries.get(relative)
    if e is None or e['kind']!='file' or witness['sha256']!=e['sha256']:raise ValueError('loaded backing differs from inventory: '+relative)

def read_fd(fd,name):
 if Path(name).name!=name:raise ValueError('basename required')
 h=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
 with os.fdopen(h,'rb') as stream:
  if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):raise ValueError('regular qualifier input required')
  return stream.read()
def write_buffer(fd,name,raw):
 if Path(name).name!=name:raise ValueError('basename required')
 h=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
 with os.fdopen(h,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())

def main():
 if sys.flags.optimize:raise ValueError('optimized qualification forbidden')
 p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--freeze-sha256',required=True);a=p.parse_args()
 fd=globals().get('__output_dir_fd__');assert type(fd) is int
 raw=read_fd(fd,'freeze.json');assert hashlib.sha256(raw).hexdigest()==a.freeze_sha256
 reg=json.loads(raw);assert globals().get('__executed_source_sha256__')==reg['qualification_worker_sha256'], 'captured qualifier source differs'
 assert reg['exact_dependencies']=={'torch':'2.9.1','numpy':'2.2.6','scipy':'1.15.3','pandas':'2.3.3','sympy':'1.14.0','PyYAML':'6.0.3'}
 assert reg['rss_bytes']==3*2**30 and reg['wall_seconds']==900 and reg['slurm_wall_limit']=='00:15:00'
 prefix=Path(reg['environment']).resolve();fd=globals().get('__output_dir_fd__');assert type(fd) is int and os.fstat(fd).st_ino==reg['output_inode'];out=Path('/proc/self/fd/'+str(fd));assert Path(sys.prefix).resolve()==prefix and sys.version_info[:2]==(3,11)
 assert Path(reg['base_python']).resolve(strict=True)==Path(reg['base_python_resolved']).resolve(strict=True) and sha(reg['base_python'])==reg['base_python_sha256'] and sha('/proc/self/exe')==reg['base_python_sha256']
 base=Path(reg['approved_base_prefix']).resolve();assert Path(sys.base_prefix).resolve()==base
 records=dependency_records(reg)
 allowed_sources={str((out/name).resolve()):pin for name,pin in reg['execution_source_pins'].items()}
 loaded=loaded_module_records(prefix,base,reg['native_backing_pins'],allowed_sources)
 files=[]
 for f in sorted(prefix.rglob('*')):
  if f.is_symlink():
   target=f.resolve(strict=True);assert target.is_relative_to(prefix) or target==Path(reg['base_python_resolved']).resolve(strict=True),'unexpected external environment symlink'
   files.append({'path':str(f.relative_to(prefix)),'kind':'symlink','link':os.readlink(f),'resolved':str(target),'target_sha256':sha(target) if target.is_file() else None})
  elif f.is_file():files.append({'path':str(f.relative_to(prefix)),'kind':'file','bytes':f.stat().st_size,'sha256':sha(f)})
  else:assert f.is_dir(),'nonregular environment object'
 inventory_raw=(json.dumps({'prefix':str(prefix),'files':files},indent=2,allow_nan=False)+'\n').encode()
 write_buffer(fd,'runtime_inventory.json',inventory_raw)
 result={'qualified':True,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.executable,'python_version':sys.version,'python_sha256':sha(sys.executable),'base_python':reg['base_python'],'base_python_sha256':reg['base_python_sha256'],'base_prefix':sys.base_prefix,'sys_path':sys.path,'dependencies':records,'loaded_modules':loaded,'executed_source_sha256':__executed_source_sha256__,'runtime_inventory_sha256':hashlib.sha256(inventory_raw).hexdigest(),'objects':len(files),'freeze_sha256':a.freeze_sha256,'new_fits':0,'new_responses':0}
 write_buffer(fd,'runtime_qualification.json',(json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
if __name__=='__main__':main()

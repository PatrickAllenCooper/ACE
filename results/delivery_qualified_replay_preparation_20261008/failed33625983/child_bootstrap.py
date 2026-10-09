import base64,hashlib,json,os,stat,sys,types
from pathlib import Path
package,pin,fdtext,cli64,verifier64,qual64,guard64,manifest64,freeze_pin=sys.argv[1:]
fd=int(fdtext);root=Path('/proc/self/fd/'+str(fd))
def read_leaf(name):
 handle=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
 with os.fdopen(handle,'rb') as stream:
  if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):raise ValueError('regular child descriptor input required')
  return stream.read()
regraw=read_leaf('freeze.json')
if hashlib.sha256(regraw).hexdigest()!=freeze_pin:raise ValueError('child freeze differs')
reg=json.loads(regraw);info=os.fstat(fd)
if info.st_ino!=reg['output_inode'] or info.st_uid!=os.geteuid() or info.st_mode&0o077:raise ValueError('child output differs')
if vars(sys.modules['__main__']).get('__executed_source_sha256__')!=reg['execution_source_pins']['child_bootstrap.py']:raise ValueError('actual bootstrap source identity differs')
qraw=base64.b64decode(qual64);qpath=str(root/'qualify_runtime.py')
if hashlib.sha256(qraw).hexdigest()!=reg['qualification_worker_sha256']:raise ValueError('captured qualifier differs')
q=types.ModuleType('ace_qualified_runtime');q.__file__=qpath;sys.modules[q.__name__]=q
q.__dict__.update(__executed_source_sha256__=hashlib.sha256(qraw).hexdigest(),__output_dir_fd__=fd)
exec(compile(qraw,qpath,'exec',dont_inherit=True),q.__dict__)
sys.argv=[qpath,'--freeze',str(root/'freeze.json'),'--freeze-sha256',freeze_pin]
q.main()
qualified=json.loads(read_leaf('runtime_qualification.json'))
if qualified['qualified'] is not True or qualified['freeze_sha256']!=freeze_pin:raise ValueError('runtime not qualified')
manifest_raw=base64.b64decode(manifest64)
if hashlib.sha256(manifest_raw).hexdigest()!=pin:raise ValueError('captured child manifest differs')
manifest=json.loads(manifest_raw)
graw=base64.b64decode(guard64);gpath=str(root/'checkpoint_guard.py')
if hashlib.sha256(graw).hexdigest()!=reg['execution_source_pins']['checkpoint_guard.py']:raise ValueError('checkpoint guard differs')
g=types.ModuleType('ace_runtime_checkpoint_guard');g.__file__=gpath;sys.modules[g.__name__]=g
exec(compile(graw,gpath,'exec',dont_inherit=True),g.__dict__)
g.state=g.install(q,reg,fd,manifest,package,reg['execution_source_pins'])
verifier_path=str(Path(package)/'verify_delivery_release.py')
v=types.ModuleType('verify_delivery_release');v.__file__=verifier_path;sys.modules[v.__name__]=v
exec(compile(base64.b64decode(verifier64),verifier_path,'exec',dont_inherit=True),v.__dict__)
cli_raw=base64.b64decode(cli64);cli_path=str(Path(package)/'replay_delivery_prospective_release.py')
main_context=vars(sys.modules['__main__'])
main_context.update(__name__='__main__',__file__=cli_path,__executed_source_sha256__=hashlib.sha256(cli_raw).hexdigest())
sys.argv=[cli_path,'--root',package,'--expected-manifest-sha256',pin,'--receipt',str(root/'replay_receipt.json')]
exec(compile(cli_raw,cli_path,'exec',dont_inherit=True),main_context)
if sys.modules['ace_runtime_checkpoint_guard'].state[0]:raise ValueError('checkpoint runtime guard never executed')

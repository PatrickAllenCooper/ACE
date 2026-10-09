"""One derived descriptor-bound supervised qualification/replay; no installs."""
import argparse,base64,hashlib,json,os,re,stat,subprocess,sys,time,types
from datetime import datetime,timezone
from pathlib import Path

def utc():return datetime.now(timezone.utc).isoformat()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path,expected):
 p=Path(path);require(p.is_file() and not p.is_symlink(),'regular input required')
 raw=p.read_bytes();require(digest(raw)==expected,'input digest differs: '+str(p));return raw
def read_fd(fd,name):
 require(Path(name).name==name,'basename required')
 h=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
 with os.fdopen(h,'rb') as f:
  require(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'regular descriptor input required');return f.read()
def write_fd(fd,name,value):
 require(Path(name).name==name,'basename required')
 h=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
 with os.fdopen(h,'w') as f:
  json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def mem_bytes(v):
    m=re.fullmatch(r'([0-9]+)([MG])',v);require(m is not None,'unsupported scheduler memory format')
    return int(m[1])*2**(20 if m[2]=='M' else 30)

def allocation(job,reg):
    require(job.isdigit(),'numeric Slurm job ID required')
    raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=10)
    fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S+)',raw))
    require(fields.get('JobId')==job,'scheduler identity mismatch')
    for key,want in [('Account',reg['account']),('Partition',reg['partition']),('QOS',reg['qos']),('NumNodes','1'),('NumCPUs','1'),('NumTasks','1'),('CPUs/Task','1')]:
        require(fields.get(key)==want,'allocated '+key+' differs')
    require(fields.get('TimeLimit')=='00:15:00','allocated wall limit differs')
    require(mem_bytes(fields.get('MinMemoryNode',''))==reg['rss_bytes'],'allocated node memory differs')
    tres_raw=fields.get('AllocTRES','')
    require(bool(tres_raw) and tres_raw not in ('(null)','N/A','Unknown'),'complete allocated-resource evidence required')
    tres={}
    for component in tres_raw.split(','):
        parts=component.split('=')
        require(len(parts)==2 and bool(parts[0]) and bool(parts[1]) and parts[0] not in tres,'ambiguous allocated-resource evidence')
        tres[parts[0]]=parts[1]
    require(tres.get('cpu')=='1' and tres.get('node')=='1','allocated TRES CPU/node differs')
    require(mem_bytes(tres.get('mem',''))==reg['rss_bytes'],'allocated TRES memory differs')
    require(not any(k.lower().startswith('gres') for k in tres),'generic accelerator allocation forbidden')
    require('gpu' not in ' '.join(fields.get(k,'').lower() for k in ('TresPerNode','TresPerTask','Gres')),'GPU allocation forbidden')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==1 and int(os.environ.get('SLURM_NTASKS','0'))==1 and int(os.environ.get('SLURM_JOB_NUM_NODES','0'))==1,'task/node environment differs')
    require(fields.get('JobState')=='RUNNING','allocation not running')
    return {'raw_scontrol':raw,'raw_scontrol_sha256':digest(raw.encode()),'verified_fields':{k:fields.get(k) for k in ('JobId','Account','Partition','QOS','NumNodes','NumCPUs','NumTasks','CPUs/Task','TimeLimit','MinMemoryNode','AllocTRES','TresPerNode','TresPerTask','Gres')}}

CHILD_BOOTSTRAP=r"""import base64,hashlib,json,os,stat,sys,types
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
"""

def validate_receipt(raw,reg):
    d=json.loads(raw)
    require(d.get('full_supplemental_replay') is True,'full replay not qualified')
    for k,w in [('checkpoints_replayed',640),('primary_cells_recomputed',240),('cached_responses_checked',48000),('new_optimizer_updates',0),('new_responses',0)]:
        require(type(d.get(k)) is int and d[k]==w,'invalid receipt '+k)
    require(d.get('runtime')==reg['exact_dependencies'],'actual runtime differs')
    require(d.get('integrity',{}).get('manifest_sha256')==reg['manifest_sha256'],'receipt manifest differs')
    require(d['integrity'].get('files_verified')==reg['package_files'] and d['integrity'].get('bindings_verified')==reg['package_bindings'],'receipt inventory differs')
    return digest(raw)

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def validate_qualification(fd,reg,pin):
 raw=read_fd(fd,'runtime_qualification.json');q=json.loads(raw)
 require(q['qualified'] is True and q['freeze_sha256']==pin and q['executed_source_sha256']==reg['qualification_worker_sha256'],'new supervised runtime qualification required')
 require(q['python_sha256']==reg['base_python_sha256'] and Path(q['python']).resolve()==Path(reg['python']).resolve(),'qualified interpreter differs')
 invraw=read_fd(fd,'runtime_inventory.json')
 require(digest(invraw)==q['runtime_inventory_sha256'],'qualified inventory differs')
 inventory=json.loads(invraw);require(Path(inventory['prefix']).resolve()==Path(reg['environment']).resolve() and q['objects']==len(inventory['files']),'qualified inventory membership differs')
 for name,version in reg['exact_dependencies'].items():
  d=q['dependencies'][name];require(d['distribution']==version and d['imported'].split('+',1)[0]==version and Path(d['origin']).resolve().is_relative_to(Path(reg['environment']).resolve()),'actual dependency differs')
 preraw=read_fd(fd,'pre_checkpoint_runtime.json');pre=json.loads(preraw)
 require(pre['passed'] is True and pre['runtime_qualification_sha256']==digest(raw) and pre['runtime_inventory_sha256']==digest(invraw),'current pre-checkpoint runtime closure required')
 return {'runtime_qualification_sha256':digest(raw),'runtime_inventory_sha256':digest(invraw),'pre_checkpoint_runtime_sha256':digest(preraw),'environment_objects_verified':q['objects'],'python_sha256':q['python_sha256']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--freeze-sha256',required=True);a=p.parse_args()
 started=time.monotonic();trusted=False;fd=globals().get('__output_dir_fd__');job=os.environ.get('SLURM_JOB_ID','')
 result={'status':'wrapper_failure','exit_code':1,'job_id':job,'freeze_sha256':a.freeze_sha256,'stage':'output_authentication'}
 try:
  require(type(fd) is int,'verified output descriptor required')
  info=os.fstat(fd);require(stat.S_ISDIR(info.st_mode) and info.st_uid==os.geteuid() and info.st_mode&0o077==0,'trusted output descriptor required')
  raw=read_fd(fd,'freeze.json');require(digest(raw)==a.freeze_sha256,'freeze pin differs');reg=json.loads(raw)
  require(info.st_ino==reg['output_inode'] and digest(read_fd(fd,'output_claim.json'))==reg['output_claim_sha256'],'output identity/claim differs')
  trusted=True
  require(not sys.flags.optimize and job.isdigit(),'unoptimized Slurm execution required')
  require(globals().get('__executed_source_sha256__')==reg['supervision_worker_sha256'],'captured wrapper bytes differ')
  require(reg['account']=='ucb736_asc1' and reg['partition']=='acpu' and reg['qos']=='cpu-normal' and reg['cpus']==1 and reg['rss_bytes']==3*2**30 and reg['wall_seconds']==900 and reg['slurm_wall_limit']=='00:15:00','resource freeze differs')
  require(reg['new_fits']==reg['new_responses']==0,'inference-only freeze required')
  root=Path('/proc/self/fd/'+str(fd))
  for name in ('replay_started.json','replay_execution.json','replay_receipt.json','replay.log','runtime_inventory.json','runtime_qualification.json','pre_checkpoint_runtime.json'):
   try:os.stat(name,dir_fd=fd,follow_symlinks=False)
   except FileNotFoundError:continue
   raise ValueError('existing attempt artifact: '+name)
  write_fd(fd,'replay_started.json',{'at':utc(),'job_id':job,'freeze_sha256':a.freeze_sha256})
  result.update(manifest_sha256=reg['manifest_sha256'],stage='allocation_authentication')
  result['verified_allocation']=allocation(job,reg)
  require(sha_file(sys.executable)==reg['base_python_sha256'] and sha_file('/proc/self/exe')==reg['base_python_sha256'] and Path(sys.prefix).resolve()==Path(reg['environment']).resolve() and sys.version_info[:2]==(3,11),'wrapper interpreter differs')
  captures={}
  for name,pin in reg['execution_source_pins'].items():
   rawsource=read_fd(fd,name);require(digest(rawsource)==pin,'execution source differs: '+name);captures[name]=rawsource
  require(captures['child_bootstrap.py']==CHILD_BOOTSTRAP.encode(),'bootstrap constant/source differs')
  package=Path(reg['package']);manifest_raw=read(package/'manifest.json',reg['manifest_sha256']);manifest=json.loads(manifest_raw)
  entries={f['path']:f for f in manifest['files']}
  require(len(entries)==len(manifest['files'])==reg['package_files'],'unique package membership differs')
  cli=read(package/'replay_delivery_prospective_release.py',entries['replay_delivery_prospective_release.py']['sha256'])
  verifier=read(package/'verify_delivery_release.py',entries['verify_delivery_release.py']['sha256'])
  original=read(reg['original_supervisor'],reg['original_supervisor_sha256'])
  sampler=types.ModuleType('authenticated_original_sampler');sampler.__file__=reg['original_supervisor']
  exec(compile(original,sampler.__file__,'exec',dont_inherit=True),sampler.__dict__)
  supervision=types.ModuleType('derived_owned_supervision');supervision.__file__=str(root/'owned_supervision.py')
  exec(compile(captures['owned_supervision.py'],supervision.__file__,'exec',dont_inherit=True),supervision.__dict__)
  loader="import base64,hashlib,sys;source,path,*args=sys.argv[1:];raw=base64.b64decode(source);context=vars(sys.modules['__main__']);context.update(__name__='__main__',__file__=path,__executed_source_sha256__=hashlib.sha256(raw).hexdigest());sys.argv=[path,*args];exec(compile(raw,path,'exec',dont_inherit=True),context)"
  encode=lambda b:base64.b64encode(b).decode()
  command=[sys.executable,'-I','-B','-c',loader,encode(captures['child_bootstrap.py']),str(root/'child_bootstrap.py'),str(package),reg['manifest_sha256'],str(fd),encode(cli),encode(verifier),encode(captures['qualify_runtime.py']),encode(captures['checkpoint_guard.py']),encode(manifest_raw),a.freeze_sha256]
  result['stage']='supervised_runtime_then_replay'
  child=supervision.supervise(command,started+850,reg['rss_bytes'],fd,'replay.log',dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),sampler.process_rss)
  result['child_execution']=child
  require(child['status']=='complete' and child['exit_code']==0 and child['cleanup']['complete'] is True,'child qualification/replay/cleanup failed')
  result['stage']='receipt_authentication'
  result['authenticated_runtime']=validate_qualification(fd,reg,a.freeze_sha256)
  result['replay_receipt_sha256']=validate_receipt(read_fd(fd,'replay_receipt.json'),reg)
  result.update(status='complete',exit_code=0,stage='qualified')
 except BaseException as exc:
  result.update(status='wrapper_failure',exit_code=1,failure_type=type(exc).__name__,failure_reason=str(exc))
 result.update(finished_at=utc(),wrapper_elapsed_seconds=time.monotonic()-started,new_fits=0,new_responses=0)
 if trusted:
  try:write_fd(fd,'replay_execution.json',result)
  except BaseException as exc:
   result['terminal_custody_failure']={'type':type(exc).__name__,'reason':str(exc)};result['status']='custody_failure';result['exit_code']=1
   print(json.dumps(result),file=sys.stderr)
 else:print(json.dumps(result),file=sys.stderr)
 if result['exit_code']!=0:raise SystemExit(1)

if __name__=='__main__':main()

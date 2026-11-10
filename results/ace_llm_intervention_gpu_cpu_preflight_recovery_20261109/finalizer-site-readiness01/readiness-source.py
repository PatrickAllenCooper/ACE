from pathlib import Path
import datetime,hashlib,json,os,signal,stat,subprocess,tempfile,time

def need(ok,message):
 if not ok:raise ValueError(message)
def expired(*_):raise TimeoutError('bounded150s metadata readiness expired')
signal.signal(signal.SIGALRM,expired);signal.alarm(150)
start=time.monotonic();out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':{},'scope':'CPU finalizer site readiness; no execution qualification','profile':{'account':'ucb736_asc2','partition':'acpu','qos':'cpu-normal','cpus':2,'memory_mib':7680,'wall_seconds':4200,'gpus':0,'requeue':False}}
commands={
 'hostname':['hostname'],
 'partition':['scontrol','show','partition','acpu','-o'],
 'nodes':['sinfo','-p','acpu','-h','-o','%P|%a|%l|%c|%m'],
 'qos':['sacctmgr','-n','-P','show','qos','where','name=cpu-normal','format=Name,MaxWall,MaxTRESPU,MaxTRESPJ,MaxJobsPU,MaxSubmitPU,Flags'],
 'association':['sacctmgr','-n','-P','show','assoc','where','user=paco0228','account=ucb736_asc2','format=User,Account,Partition,QOS,MaxJobs,MaxSubmitJobs'],
 'quota':['curc-quota'],
 'df':['df','-Pk','/projects/paco0228','/scratch/alpine/paco0228'],
 'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],
 'accounting':['sacct','-X','-u','paco0228','-S','2026-11-02','--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P']}
try:
 for n,c in commands.items():
  t=time.monotonic()
  try:
   r=subprocess.run(c,capture_output=True,text=True,timeout=25)
   out['commands'][n]={'command':c,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'elapsed_seconds':time.monotonic()-t}
  except subprocess.TimeoutExpired as exc:
   txt=lambda v:v.decode(errors='replace') if isinstance(v,bytes) else v
   out['commands'][n]={'command':c,'returncode':None,'timeout':True,'stdout':txt(exc.stdout),'stderr':txt(exc.stderr),'elapsed_seconds':time.monotonic()-t};raise
  need(r.returncode==0,'metadata command failed: '+n)
 need(not any('ace_llm' in l.lower() for l in out['commands']['queue']['stdout'].splitlines()),'active ACE job requires coordination')
 parent=Path('/scratch/alpine/paco0228/ACE/results');need(parent.resolve()==parent and parent.is_dir() and parent.stat().st_uid==os.getuid(),'owned canonical scratch parent required')
 target=parent/'ace_llm_intervention_scientific_finalization_20261108_01'
 out['finalizer_root']={'path':str(target),'lexists':os.path.lexists(target)}
 need(not out['finalizer_root']['lexists'],'prospective finalizer root already exists')
 out['namespaces']={}
 for p,h in [(parent/'.ace-scientific-screen-20261107.namespace.json','58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'),(parent/'.ace-scientific-screen-20261107.namespace-recovery02.json','40c0937c623d9473a413347125d3bc35648955132c4cf2394f055d4b228fed6a')]:
  need(p.parent.resolve()==p.parent and not p.is_symlink(),'namespace parent/file alias')
  with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
   a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode) and a.st_nlink==1 and a.st_size<=65536,'ordinary bounded namespace required');data=f.read(65537);b=os.fstat(f.fileno())
  ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
  digest=hashlib.sha256(data).hexdigest();need(ident(a)==ident(b)==ident(p.lstat()) and digest==h,'original namespace bytes/identity differ')
  out['namespaces'][str(p)]={'bytes':len(data),'sha256':digest,'unchanged':True}
 for n in ['/projects/paco0228','/scratch/alpine/paco0228/ACE/results']:
  p=Path(n);s=p.stat();v=os.statvfs(n)
  out[n]={'uid':s.st_uid,'mode':oct(stat.S_IMODE(s.st_mode)),'canonical':str(p.resolve()),'available_bytes':v.f_bavail*v.f_frsize,'available_inodes':v.f_favail}
  need(p.resolve()==p and s.st_uid==os.getuid(),'owned canonical storage parent required')
 probe=Path(tempfile.mkdtemp(prefix='ace-finalizer-site-readiness-',dir=parent));out['scratch_probe']={'path':str(probe),'dedicated':True,'removed':False}
 f=probe/'probe';payload=b'ACE CPU finalizer readiness; no scientific data\n'
 with f.open('xb') as s:s.write(payload);s.flush();os.fsync(s.fileno())
 need(f.read_bytes()==payload,'dedicated scratch readback differs')
 f.unlink();probe.rmdir();out['scratch_probe']['removed']=True;out['scratch_write_read_cleanup']=True
 need(not os.path.lexists(target),'prospective output root changed during readiness')
 out['status']='metadata_observation_complete'
except BaseException as exc:
 out['failure']={'type':type(exc).__name__,'message':str(exc)};out['status']='failed_metadata_observation';raise
finally:
 out['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();out['elapsed_seconds']=time.monotonic()-start
 print(json.dumps(out,sort_keys=True),flush=True)

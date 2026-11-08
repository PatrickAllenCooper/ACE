ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_training_gpu_20261108_01'
JOB='35393751'

from pathlib import Path
import datetime,hashlib,json,os,stat,subprocess,time
root=Path(ROOT);deadline=time.monotonic()+150
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':{},'files':{},'submission_journals':{}}
def read(path,cap):
 if time.monotonic()>deadline:raise TimeoutError('inventory deadline')
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  a=os.fstat(f.fileno())
  if not stat.S_ISREG(a.st_mode) or a.st_size>cap:raise ValueError('bounded regular artifact required: '+str(path))
  raw=f.read(cap+1);b=os.fstat(f.fileno())
 if len(raw)>cap or (a.st_ino,a.st_size,a.st_mtime_ns)!=(b.st_ino,b.st_size,b.st_mtime_ns):raise ValueError('changing artifact')
 return raw
commands={
 'accounting':['sacct','-j',JOB,'-n','-P','--format=JobIDRaw,JobID,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180,MaxRSS,TotalCPU'],
 'details':['sacct','-X','-j',JOB,'-n','-P','--format=JobIDRaw,Partition,QOS,TimelimitRaw,ReqCPUS,ReqTRES%180,User'],
 'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],
 'census':['sacct','-X','-u','paco0228','-S','2026-11-02','-n','-P','--format=JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180'],
 'job':['scontrol','show','job','-o',JOB]}
try:
 for name,cmd in commands.items():
  try:
   p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
   v['results'][name]={'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
  except subprocess.TimeoutExpired as e:
   decode=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x
   v['results'][name]={'command':cmd,'returncode':None,'stdout':decode(e.stdout),'stderr':decode(e.stderr),'timed_out':True}
 if not root.is_dir() or root.is_symlink():raise ValueError('ordinary job root required')
 size=0
 for p in sorted(root.rglob('*')):
  if p.is_symlink():raise ValueError('output symlink: '+str(p))
  if p.is_dir():continue
  raw=read(p,64*1024**2);size+=len(raw)
  if size>512*1024**2 or len(v['files'])>=4096:raise ValueError('complete raw custody exceeds bound')
  v['files'][p.relative_to(root).as_posix()]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 for folder in sorted(root.parent.glob('ace_llm_intervention*')):
  if folder.is_symlink():raise ValueError('journal root symlink')
  for p in sorted(folder.glob('submission.*.json')):
   if len(v['submission_journals'])>=128:raise ValueError('journal bound exceeded')
   v['submission_journals'][str(p)]=hashlib.sha256(read(p,4*1024**2)).hexdigest()
except BaseException as e:v['inventory_error']={'type':type(e).__name__,'message':str(e)}
print(json.dumps(v))

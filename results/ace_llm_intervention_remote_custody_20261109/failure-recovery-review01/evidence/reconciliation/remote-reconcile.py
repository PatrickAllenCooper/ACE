import os,json,hashlib,stat,signal,time,subprocess
from pathlib import Path
start=time.monotonic()
def expired(*args):raise TimeoutError('failure-custody reconciliation deadline')
signal.signal(signal.SIGALRM,expired);signal.alarm(300)
proc=Path('/proc/1782538')
assert not proc.exists(), 'prior remote backup process still present'
base=Path('/projects/paco0228/ACE_evidence')
assert base.resolve()==base and base.stat().st_uid==os.getuid()
out=base/'ace-llm-campaign-cpu-failed-backup-20261109-01'
assert not os.path.lexists(out)
roots={'scratch':Path('/scratch/alpine/paco0228/ACE_custody_transfer/ace-llm-campaign-cpu-20261109-01'),'projects':base/'ace-llm-campaign-cpu-20261109-01'}
expected={'scratch':{'intent.json','capacity.json','failure.json','raw.tar'},'projects':{'intent.json','capacity.json','failure.json'}}
for label,p in roots.items():
 assert p.resolve()==p and p.stat().st_uid==os.getuid()
 assert {x.name for x in p.iterdir()}==expected[label], 'unreconciled prior backup output set'
for p in roots.values():
 f=json.loads((p/'failure.json').read_text());assert f['type']=='TimeoutError' and f['message']=='bounded remote custody deadline'
out.mkdir(mode=0o700)
records={}
def identity(s):return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
try:
 for label,p in roots.items():
  dest=out/label;dest.mkdir(mode=0o700)
  for src in sorted(p.iterdir()):
   assert not src.is_symlink()
   with os.fdopen(os.open(src,os.O_RDONLY|os.O_NOFOLLOW),'rb') as a:
    before=os.fstat(a.fileno());assert stat.S_ISREG(before.st_mode) and before.st_nlink==1
    assert before.st_size<=2*1024**3
    h=hashlib.sha256();n=0
    with os.fdopen(os.open(dest/src.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as b:
     for block in iter(lambda:a.read(1024**2),b''):b.write(block);h.update(block);n+=len(block)
     b.flush();os.fsync(b.fileno())
    assert identity(before)==identity(os.fstat(a.fileno()))==identity(src.lstat()) and n==before.st_size
   second=hashlib.sha256()
   with (dest/src.name).open('rb') as b:
    for block in iter(lambda:b.read(1024**2),b''):second.update(block)
   assert second.hexdigest()==h.hexdigest()
   records[label+'/'+src.name]={'bytes':n,'sha256':h.hexdigest(),'original':str(src),'copy':str(dest/src.name)}
 observations={}
 for name,cmd in {'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%a|%R'],'job':['sacct','-j','35482135','-n','-P','--format','JobIDRaw,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180']}.items():
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=20);observations[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};assert p.returncode==0
 result={'schema':'ace-failed-collector-reconciliation-v1','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'prior_process_absent':True,'records':records,'observations':observations,'elapsed_seconds':time.monotonic()-start,'copy_only':True,'partial_archive_is_not_complete_backup':True,'scientific_acceptance':False,'jobs_submitted':0,'originals_changed':False}
 raw=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
 with (out/'reconciliation.json').open('xb') as f:f.write(raw)
 print(raw.decode())
except BaseException as e:
 with (out/'reconciliation-failure.json').open('x') as f:json.dump({'type':type(e).__name__,'message':str(e),'retry_allowed':False},f)
 raise

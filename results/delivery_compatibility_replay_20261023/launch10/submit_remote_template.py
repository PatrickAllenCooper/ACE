import os,stat,hashlib,json,subprocess,datetime,re
out='/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261023_10'
inode=1552757822
pins={'freeze.json': '831cebd8dce8d127d90b713343643b5ab6fd107c8198d2311dd867cf80481b2b', 'output_claim.json': 'aba56a536675c6e0471417ce635f8a51c6b5e213483ceefbaebb95f4c346283a', 'run.sbatch': '57d165c8ccaba423598f66e813ccb16d808d6af3bb994fee93317c39a933f44a', 'authorization.json': 'abd4c5125a5e7a8523e46839ece8c3aae6cdee758dc08dcb89d6ffc8b55a5c8f', 'qualify_runtime.py': '11a27cd0e1b0e685c24aeef74547a38976b6bebe43cc190147104f2b3cdaabee', 'supervise_replay.py': 'fce27521fdb17bb94384263663c40807ccd1e3e0a988e0081605ea138ea075dc', 'checkpoint_guard.py': 'e6a65dbe736ad93b07aee897e5713356e3d4c6ccd59305338eccd83034f91546', 'owned_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'child_bootstrap.py': 'e2ddf8fe650de97f54278eb8f5f1253e11251c44dc9403af215f1888dc6be739'}
intent=json.loads(os.environ['ACE_APPROVED_SUBMISSION_INTENT'])
fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);info=os.fstat(fd)
if info.st_ino!=inode or info.st_uid!=os.geteuid() or info.st_mode&0o077:raise ValueError('output identity differs')
def read(name):
 h=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
 with os.fdopen(h,'rb') as f:
  if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('regular input required')
  return f.read()
def write(name,obj):
 h=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
 with os.fdopen(h,'w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.fsync(fd)
if set(os.listdir(fd))!=set(pins):raise ValueError('attempt artifacts already exist; reconcile without retry')
for name,pin in pins.items():
 if hashlib.sha256(read(name)).hexdigest()!=pin:raise ValueError('input differs '+name)
queue=subprocess.check_output(['squeue','-h','-u','paco0228','-o','%i|%j|%T|%Z'],text=True,timeout=20)
ace=[r for r in queue.splitlines() if r.split('|')[1].lower().startswith('ace_') or '/ACE/' in r.split('|')[-1] or r.split('|')[-1].endswith('/ACE')]
if ace:raise ValueError('ACE queue now nonempty')
fs=os.statvfs(out)
if fs.f_bavail*fs.f_frsize<2*1024**3:raise ValueError('insufficient scratch')
h=os.open('.pre-submit-readiness',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
with os.fdopen(h,'wb') as f:f.write(b'ACE single approved replay10');f.flush();os.fsync(f.fileno())
if read('.pre-submit-readiness')!=b'ACE single approved replay10':raise ValueError('scratch readback differs')
os.unlink('.pre-submit-readiness',dir_fd=fd)
intent.update(queue_before=ace,scratch_available_bytes=fs.f_bavail*fs.f_frsize,write_read_cleanup=True,remote_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
write('submission_started.json',intent)
try:
 proc=subprocess.run(['sbatch','--parsable','--chdir',out,out+'/run.sbatch'],capture_output=True,text=True,timeout=30)
 result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,'retry_allowed':False}
 if proc.returncode==0 and re.fullmatch(r'[0-9]+(?:;[^\s]+)?',proc.stdout.strip()):result['job_id']=int(proc.stdout.strip().split(';')[0])
 else:result['submission_status']='rejected_or_ambiguous; reconcile; never retry automatically'
except BaseException as exc:
 result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'submission_status':'ambiguous','failure_type':type(exc).__name__,'failure_reason':str(exc),'retry_allowed':False}
write('submission_result.json',result)
print(json.dumps(result,indent=2))

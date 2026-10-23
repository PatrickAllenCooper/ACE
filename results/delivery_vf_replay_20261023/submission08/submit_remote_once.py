import os,stat,hashlib,json,subprocess,datetime,re
out='/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261023_08'
inode=384996661
pins={'freeze.json': 'c22109e41253d2b60c7e35da46c36292fc12bcad0c4bf04a2f235f96b7c20bd6', 'output_claim.json': '5031c5a7e18ea88f06e1e3f12857e94799c498ba9eefdab9d4a99f2312d6e0c7', 'run.sbatch': '008f87d944d95b747d28cfc676963b3dd98db00182a5fd999cfa964fba2c8a87', 'authorization.json': 'd9459fca1fc0c9aeb0ba332ceb6797f341125088774744ed3697b2e5c2d90208', 'qualify_runtime.py': 'e1acd6380b70c77a05b07a13ba604ebb023bd190a1e3be3cbecb5042c673d01e', 'supervise_replay.py': 'fce27521fdb17bb94384263663c40807ccd1e3e0a988e0081605ea138ea075dc', 'checkpoint_guard.py': 'dc97ed8ed3ac006140856f57b10744668f96684946b912235985a2ada3412fd8', 'owned_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'child_bootstrap.py': 'e2ddf8fe650de97f54278eb8f5f1253e11251c44dc9403af215f1888dc6be739'}
intent={'at': '2026-10-23T17:43:04.993074+00:00', 'packet_commit': 'f0c21e66a0c525828d4db1dc9707363c979484b3', 'review_sha256': 'cd80bddf1a5f0d1f0567352fc9899aceede4019c6cc11b4783ccd08cad7a927a', 'review_required_unresolved': 0, 'retry_allowed': False, 'pins': {'freeze.json': 'c22109e41253d2b60c7e35da46c36292fc12bcad0c4bf04a2f235f96b7c20bd6', 'output_claim.json': '5031c5a7e18ea88f06e1e3f12857e94799c498ba9eefdab9d4a99f2312d6e0c7', 'run.sbatch': '008f87d944d95b747d28cfc676963b3dd98db00182a5fd999cfa964fba2c8a87', 'authorization.json': 'd9459fca1fc0c9aeb0ba332ceb6797f341125088774744ed3697b2e5c2d90208', 'qualify_runtime.py': 'e1acd6380b70c77a05b07a13ba604ebb023bd190a1e3be3cbecb5042c673d01e', 'supervise_replay.py': 'fce27521fdb17bb94384263663c40807ccd1e3e0a988e0081605ea138ea075dc', 'checkpoint_guard.py': 'dc97ed8ed3ac006140856f57b10744668f96684946b912235985a2ada3412fd8', 'owned_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'child_bootstrap.py': 'e2ddf8fe650de97f54278eb8f5f1253e11251c44dc9403af215f1888dc6be739'}, 'authorization_sha256': 'd9459fca1fc0c9aeb0ba332ceb6797f341125088774744ed3697b2e5c2d90208', 'controller_pid': 27258}
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
with os.fdopen(h,'wb') as f:f.write(b'ACE single approved replay08');f.flush();os.fsync(f.fileno())
if read('.pre-submit-readiness')!=b'ACE single approved replay08':raise ValueError('scratch readback differs')
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

import os,stat,hashlib,json,subprocess,datetime,re
out='/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261023_07'
inode=384569057
pins={'freeze.json': 'e2b79a855a86eca8f96ebbc9753b01a75c921a77b80262b652ab593460f4c227', 'output_claim.json': '9e7557a9a035478cb5e26520a7106629e17ed4c5b40a4424231f5916eb7a2130', 'run.sbatch': '00b86c010da2e0ed9efea86d71d5985ecd0823136ef4d30b4c9c3b07f0a43da7', 'authorization.json': '350372308cc93fc199ea5c00a1c838c80b94594b0273de7443fa3ddef3aabe91', 'supervise_replay.py': '3e09f8e40cf3613a5f5642fcd8275039e048724bcc3441108634dce56610cf44', 'qualify_runtime.py': 'caec9560f5711de478c56b854d40506d7c493df0dad76c578733186a6bcf1ac3', 'checkpoint_guard.py': 'dc97ed8ed3ac006140856f57b10744668f96684946b912235985a2ada3412fd8', 'owned_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'child_bootstrap.py': 'e2ddf8fe650de97f54278eb8f5f1253e11251c44dc9403af215f1888dc6be739'}
intent={'at': '2026-10-23T16:54:22.988696+00:00', 'packet_commit': '2ffe5526da51d203947ef7546f7f6d1d11e91df5', 'review_sha256': '99596ae09a84fc0072f822e07ccd6d85bf85445f97b41dc4f252c7a4c951faf5', 'review_required_unresolved': 0, 'retry_allowed': False, 'pins': {'freeze.json': 'e2b79a855a86eca8f96ebbc9753b01a75c921a77b80262b652ab593460f4c227', 'output_claim.json': '9e7557a9a035478cb5e26520a7106629e17ed4c5b40a4424231f5916eb7a2130', 'run.sbatch': '00b86c010da2e0ed9efea86d71d5985ecd0823136ef4d30b4c9c3b07f0a43da7', 'authorization.json': '350372308cc93fc199ea5c00a1c838c80b94594b0273de7443fa3ddef3aabe91', 'supervise_replay.py': '3e09f8e40cf3613a5f5642fcd8275039e048724bcc3441108634dce56610cf44', 'qualify_runtime.py': 'caec9560f5711de478c56b854d40506d7c493df0dad76c578733186a6bcf1ac3', 'checkpoint_guard.py': 'dc97ed8ed3ac006140856f57b10744668f96684946b912235985a2ada3412fd8', 'owned_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'child_bootstrap.py': 'e2ddf8fe650de97f54278eb8f5f1253e11251c44dc9403af215f1888dc6be739'}, 'authorization_sha256': '350372308cc93fc199ea5c00a1c838c80b94594b0273de7443fa3ddef3aabe91', 'controller_pid': 59431}
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
with os.fdopen(h,'wb') as f:f.write(b'ACE single approved replay07');f.flush();os.fsync(f.fileno())
if read('.pre-submit-readiness')!=b'ACE single approved replay07':raise ValueError('scratch readback differs')
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

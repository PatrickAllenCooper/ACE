import os,stat,hashlib,json,subprocess,datetime,re
out='/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261023_09'
inode=1554896029
pins={'freeze.json': '5825290ce5cb061baadfa8ff2a24f6038bcc8cb6f1cce64dc451fcc9d31909ca', 'output_claim.json': '05e3eaa14e96d5719d9013dac4f143895516a881dd1e7f72e6718136b73262ef', 'run.sbatch': '9559bf0595d2cbc6db94fac487618d2fefc3a62ececab54eccfba878e91f7411', 'authorization.json': '192dab9c468f6ccb0b23c45cbf606085a5a21a5a5f4e7b5f2158b09246066f82', 'qualify_runtime.py': 'e1bfc6ef06a892461b8310975a2b7e5a0ad323256c8d8e402480ac914ffb5318', 'supervise_replay.py': 'fce27521fdb17bb94384263663c40807ccd1e3e0a988e0081605ea138ea075dc', 'checkpoint_guard.py': 'd797aceb25fa66a2d781f3da052f699ac60abdeadc5b7dea1f82ebb43e21ab80', 'owned_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'child_bootstrap.py': 'e2ddf8fe650de97f54278eb8f5f1253e11251c44dc9403af215f1888dc6be739'}
intent={'at': '2026-10-23T22:49:12.214927+00:00', 'packet_commit': 'df963b546c9cbae0bf156a091a3a9f2fae610eb2', 'review_sha256': '2c928f9ff8afab7c31724c5decb9f5076743498cc5737cc6ce86c57c4de2c03c', 'review_required_unresolved': 0, 'retry_allowed': False, 'pins': {'freeze.json': '5825290ce5cb061baadfa8ff2a24f6038bcc8cb6f1cce64dc451fcc9d31909ca', 'output_claim.json': '05e3eaa14e96d5719d9013dac4f143895516a881dd1e7f72e6718136b73262ef', 'run.sbatch': '9559bf0595d2cbc6db94fac487618d2fefc3a62ececab54eccfba878e91f7411', 'authorization.json': '192dab9c468f6ccb0b23c45cbf606085a5a21a5a5f4e7b5f2158b09246066f82', 'qualify_runtime.py': 'e1bfc6ef06a892461b8310975a2b7e5a0ad323256c8d8e402480ac914ffb5318', 'supervise_replay.py': 'fce27521fdb17bb94384263663c40807ccd1e3e0a988e0081605ea138ea075dc', 'checkpoint_guard.py': 'd797aceb25fa66a2d781f3da052f699ac60abdeadc5b7dea1f82ebb43e21ab80', 'owned_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'child_bootstrap.py': 'e2ddf8fe650de97f54278eb8f5f1253e11251c44dc9403af215f1888dc6be739'}, 'authorization_sha256': '192dab9c468f6ccb0b23c45cbf606085a5a21a5a5f4e7b5f2158b09246066f82', 'predecessor_job': 34824953, 'single_attempt': True}
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
with os.fdopen(h,'wb') as f:f.write(b'ACE single approved replay09');f.flush();os.fsync(f.fileno())
if read('.pre-submit-readiness')!=b'ACE single approved replay09':raise ValueError('scratch readback differs')
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

import os,stat,hashlib,json,subprocess,datetime,re
out='/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261024_12'
inode=685041003
pins={'freeze.json': '3ada98c1af9645b21b1e19bb09e8f8efd1450b12243c649a9851d122b53dd667', 'output_claim.json': '060db7c66d56aeeec02aa92cf491a681099fed1328ce9f0b414a28e8fca1da1b', 'run.sbatch': 'faf62778e8b045708ee0c5cfdaff69c454ddfc6efa340f4989702a13ce130112', 'authorization.json': '5a30ada2f990701c415ad51f97c2299e46226dbc839263fd1069d93cae592a5d', 'qualify_runtime.py': '117e6bb3b9d46823c95e1dc925854a2fdd155a192b167d6eee04e0cdbd9f3a69', 'supervise_replay.py': 'fa753779b6878e3c4cf2c1a3cd481afb467c8b3e00990ae1854ec420f9ef9ea6', 'checkpoint_guard.py': '6f7e15e3ef6a7263d266a534d1093df37025535b46b2eb2295c19a6d8a139889', 'owned_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'child_bootstrap.py': 'bb625984266116811e9c7045b09980c005e2f2fffeea8ab65059e59ba6d50009'}
intent={'at': '2026-10-24T19:31:44.798580+00:00', 'packet_commit': 'fc724aec60b6f259627a4f593f787f5e90af9152', 'review_sha256': '1723795bdbc3bc17848c17a76475b91215e14eb7fea0c2e70e8c19f6fb92bcf7', 'review_required_unresolved': 0, 'retry_allowed': False, 'pins': {'freeze.json': '3ada98c1af9645b21b1e19bb09e8f8efd1450b12243c649a9851d122b53dd667', 'output_claim.json': '060db7c66d56aeeec02aa92cf491a681099fed1328ce9f0b414a28e8fca1da1b', 'run.sbatch': 'faf62778e8b045708ee0c5cfdaff69c454ddfc6efa340f4989702a13ce130112', 'authorization.json': '5a30ada2f990701c415ad51f97c2299e46226dbc839263fd1069d93cae592a5d', 'qualify_runtime.py': '117e6bb3b9d46823c95e1dc925854a2fdd155a192b167d6eee04e0cdbd9f3a69', 'supervise_replay.py': 'fa753779b6878e3c4cf2c1a3cd481afb467c8b3e00990ae1854ec420f9ef9ea6', 'checkpoint_guard.py': '6f7e15e3ef6a7263d266a534d1093df37025535b46b2eb2295c19a6d8a139889', 'owned_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'child_bootstrap.py': 'bb625984266116811e9c7045b09980c005e2f2fffeea8ab65059e59ba6d50009'}, 'authorization_sha256': '5a30ada2f990701c415ad51f97c2299e46226dbc839263fd1069d93cae592a5d', 'predecessor_job': 34861753, 'single_attempt': True}
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
with os.fdopen(h,'wb') as f:f.write(b'ACE single approved replay12');f.flush();os.fsync(f.fileno())
if read('.pre-submit-readiness')!=b'ACE single approved replay12':raise ValueError('scratch readback differs')
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

import os,stat,hashlib,json,subprocess,datetime,re
out='/scratch/alpine/paco0228/ACE/results/delivery_prospective_reporting_20261024_02'
inode=686521326
pins={'run_delivery_prospective_reporting.py': '7b563c5846099a663490de3df70973e650473643b862375a78efdab331646176', 'qualify_delivery_replay_runtime.py': '117e6bb3b9d46823c95e1dc925854a2fdd155a192b167d6eee04e0cdbd9f3a69', 'supervise_delivery_qualified_replay.py': 'fa753779b6878e3c4cf2c1a3cd481afb467c8b3e00990ae1854ec420f9ef9ea6', 'delivery_replay_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'runner_delivery_confirmation.py': '2f93bacbd6c14c66a5aeac5daabcff50d47886eeec38885e6e83ab40d141baee', 'audit_delivery_prospective_results.py': '400bc49f4832b37231e12f803d240040f6d8a12255d7579cbf38da3cf5b1335a', 'verify_delivery_release.py': '9f80286faeb3266c53bba70a7cf2abc26c4129c61a56c265c263cb09a8f00d52', 'replay_delivery_prospective_release.py': '7ef3ef6f0702b4e9f836fcce00ac866703160d5e41bd657375cd543248ab08bc', 'export_delivery_prospective_claims.py': '090072f9f7ee08285a59f60352152e3bc7665c0ffea666db081d978f92cafb8e', 'prepare_delivery_prospective_supplement.py': 'd7936cbdda185a1996327438acbbe3499155481528f38113aeacf9f5c4e8db24', 'runtime_inventory.json': '27ca032d9106b9684138b3e9363a667557fcd80788629b221c07e0eb9c2801c0', 'replay_receipt.json': 'd824cbbe125ae7ed50222298bba227c84ab4384685c4b625f761801e28f5b98b', 'replay_acceptance.json': '57da98e2bbce82faad642442daa491b2cf3722df7524bfd4c881a188a36e3232', 'report_freeze.json': '00975b8ee41b5eca0cfb23a263bfdb4cdbe874eee24b68ac476ab662f92b2ca8', 'authorization.json': 'd0f73f7a5aee258117ee61d8ce5723f3d0e74168314298c686937a84c7811c23', 'output_claim.json': '6552620002b5066a328406ed746e8776362b239cb7c18be5cc2df325eba2d4e5', 'run.sbatch': 'a90f4eb5805cbcfe8ceabb69585130fb9cf0caf888364a5b31e81c11fb623ee6'}
intent={'at': '2026-10-24T21:27:22.196442+00:00', 'packet_commit': '1848d6ebac4d0dd0a48a8d6e5b5d6fb0a201f823', 'review_sha256': 'c44c2a669cbeaf236d4223027d7a052b4a914351255ad1f963a55242f7b41f38', 'review_required_unresolved': 0, 'retry_allowed': False, 'pins': {'run_delivery_prospective_reporting.py': '7b563c5846099a663490de3df70973e650473643b862375a78efdab331646176', 'qualify_delivery_replay_runtime.py': '117e6bb3b9d46823c95e1dc925854a2fdd155a192b167d6eee04e0cdbd9f3a69', 'supervise_delivery_qualified_replay.py': 'fa753779b6878e3c4cf2c1a3cd481afb467c8b3e00990ae1854ec420f9ef9ea6', 'delivery_replay_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'runner_delivery_confirmation.py': '2f93bacbd6c14c66a5aeac5daabcff50d47886eeec38885e6e83ab40d141baee', 'audit_delivery_prospective_results.py': '400bc49f4832b37231e12f803d240040f6d8a12255d7579cbf38da3cf5b1335a', 'verify_delivery_release.py': '9f80286faeb3266c53bba70a7cf2abc26c4129c61a56c265c263cb09a8f00d52', 'replay_delivery_prospective_release.py': '7ef3ef6f0702b4e9f836fcce00ac866703160d5e41bd657375cd543248ab08bc', 'export_delivery_prospective_claims.py': '090072f9f7ee08285a59f60352152e3bc7665c0ffea666db081d978f92cafb8e', 'prepare_delivery_prospective_supplement.py': 'd7936cbdda185a1996327438acbbe3499155481528f38113aeacf9f5c4e8db24', 'runtime_inventory.json': '27ca032d9106b9684138b3e9363a667557fcd80788629b221c07e0eb9c2801c0', 'replay_receipt.json': 'd824cbbe125ae7ed50222298bba227c84ab4384685c4b625f761801e28f5b98b', 'replay_acceptance.json': '57da98e2bbce82faad642442daa491b2cf3722df7524bfd4c881a188a36e3232', 'report_freeze.json': '00975b8ee41b5eca0cfb23a263bfdb4cdbe874eee24b68ac476ab662f92b2ca8', 'authorization.json': 'd0f73f7a5aee258117ee61d8ce5723f3d0e74168314298c686937a84c7811c23', 'output_claim.json': '6552620002b5066a328406ed746e8776362b239cb7c18be5cc2df325eba2d4e5', 'run.sbatch': 'a90f4eb5805cbcfe8ceabb69585130fb9cf0caf888364a5b31e81c11fb623ee6'}, 'authorization_sha256': 'd0f73f7a5aee258117ee61d8ce5723f3d0e74168314298c686937a84c7811c23', 'predecessor_job': 34864950, 'single_attempt': True, 'purpose': 'both reporters, no replay repetition'}
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
with os.fdopen(h,'wb') as f:f.write(b'ACE accepted-replay report02');f.flush();os.fsync(f.fileno())
if read('.pre-submit-readiness')!=b'ACE accepted-replay report02':raise ValueError('scratch readback differs')
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

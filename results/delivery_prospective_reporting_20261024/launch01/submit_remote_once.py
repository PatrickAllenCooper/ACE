import os,stat,hashlib,json,subprocess,datetime,re
out='/scratch/alpine/paco0228/ACE/results/delivery_prospective_reporting_20261024_01'
inode=685172495
pins={'run_delivery_prospective_reporting.py': '00b4c8c05a6b146b2213efd4b5f098163fa9620a8f30fb9b8ebdb94e4526f2f4', 'qualify_delivery_replay_runtime.py': '117e6bb3b9d46823c95e1dc925854a2fdd155a192b167d6eee04e0cdbd9f3a69', 'supervise_delivery_qualified_replay.py': 'fa753779b6878e3c4cf2c1a3cd481afb467c8b3e00990ae1854ec420f9ef9ea6', 'delivery_replay_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'runner_delivery_confirmation.py': '2f93bacbd6c14c66a5aeac5daabcff50d47886eeec38885e6e83ab40d141baee', 'audit_delivery_prospective_results.py': '400bc49f4832b37231e12f803d240040f6d8a12255d7579cbf38da3cf5b1335a', 'verify_delivery_release.py': '9f80286faeb3266c53bba70a7cf2abc26c4129c61a56c265c263cb09a8f00d52', 'replay_delivery_prospective_release.py': '7ef3ef6f0702b4e9f836fcce00ac866703160d5e41bd657375cd543248ab08bc', 'export_delivery_prospective_claims.py': '090072f9f7ee08285a59f60352152e3bc7665c0ffea666db081d978f92cafb8e', 'prepare_delivery_prospective_supplement.py': 'd7936cbdda185a1996327438acbbe3499155481528f38113aeacf9f5c4e8db24', 'runtime_inventory.json': '27ca032d9106b9684138b3e9363a667557fcd80788629b221c07e0eb9c2801c0', 'replay_receipt.json': 'd824cbbe125ae7ed50222298bba227c84ab4384685c4b625f761801e28f5b98b', 'replay_acceptance.json': '57da98e2bbce82faad642442daa491b2cf3722df7524bfd4c881a188a36e3232', 'report_freeze.json': '256887bcbcbd989e0bab93655b68f371c0615d648a83d816ffeddac1f0a65686', 'authorization.json': 'deef7cd3d511a811a9e82d1a448d9fb29e556da27cbfbf13624f75f1febef3f3', 'output_claim.json': 'c48ecb1fa66409dabe40c24cd812486a3c04a393e50f2b84a4ec2723c6443a78', 'run.sbatch': 'd2374eca226cf7bd7ad452af06f87d9287d4a7e58d1a7628fd4c92007eecf3c7'}
intent={'at': '2026-10-24T20:44:08.083888+00:00', 'packet_commit': 'f2ec4f9d05776dd4c99658c1639bb2b7014dc367', 'review_sha256': '3890f40a8076d6ced028de5686744cf5247403a0d0cd5b89fd34e69615f6d966', 'review_required_unresolved': 0, 'retry_allowed': False, 'pins': {'run_delivery_prospective_reporting.py': '00b4c8c05a6b146b2213efd4b5f098163fa9620a8f30fb9b8ebdb94e4526f2f4', 'qualify_delivery_replay_runtime.py': '117e6bb3b9d46823c95e1dc925854a2fdd155a192b167d6eee04e0cdbd9f3a69', 'supervise_delivery_qualified_replay.py': 'fa753779b6878e3c4cf2c1a3cd481afb467c8b3e00990ae1854ec420f9ef9ea6', 'delivery_replay_supervision.py': '8e8e93bf920241169b5ab728ef28c8ae93b6978923d53b31e4f63f31cec5dd63', 'runner_delivery_confirmation.py': '2f93bacbd6c14c66a5aeac5daabcff50d47886eeec38885e6e83ab40d141baee', 'audit_delivery_prospective_results.py': '400bc49f4832b37231e12f803d240040f6d8a12255d7579cbf38da3cf5b1335a', 'verify_delivery_release.py': '9f80286faeb3266c53bba70a7cf2abc26c4129c61a56c265c263cb09a8f00d52', 'replay_delivery_prospective_release.py': '7ef3ef6f0702b4e9f836fcce00ac866703160d5e41bd657375cd543248ab08bc', 'export_delivery_prospective_claims.py': '090072f9f7ee08285a59f60352152e3bc7665c0ffea666db081d978f92cafb8e', 'prepare_delivery_prospective_supplement.py': 'd7936cbdda185a1996327438acbbe3499155481528f38113aeacf9f5c4e8db24', 'runtime_inventory.json': '27ca032d9106b9684138b3e9363a667557fcd80788629b221c07e0eb9c2801c0', 'replay_receipt.json': 'd824cbbe125ae7ed50222298bba227c84ab4384685c4b625f761801e28f5b98b', 'replay_acceptance.json': '57da98e2bbce82faad642442daa491b2cf3722df7524bfd4c881a188a36e3232', 'report_freeze.json': '256887bcbcbd989e0bab93655b68f371c0615d648a83d816ffeddac1f0a65686', 'authorization.json': 'deef7cd3d511a811a9e82d1a448d9fb29e556da27cbfbf13624f75f1febef3f3', 'output_claim.json': 'c48ecb1fa66409dabe40c24cd812486a3c04a393e50f2b84a4ec2723c6443a78', 'run.sbatch': 'd2374eca226cf7bd7ad452af06f87d9287d4a7e58d1a7628fd4c92007eecf3c7'}, 'authorization_sha256': 'deef7cd3d511a811a9e82d1a448d9fb29e556da27cbfbf13624f75f1febef3f3', 'predecessor_job': 34863006, 'single_attempt': True, 'purpose': 'both reporters, no replay repetition'}
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
with os.fdopen(h,'wb') as f:f.write(b'ACE accepted-replay report01');f.flush();os.fsync(f.fileno())
if read('.pre-submit-readiness')!=b'ACE accepted-replay report01':raise ValueError('scratch readback differs')
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

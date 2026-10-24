import hashlib,json,os,pathlib,stat,subprocess,tempfile
from datetime import datetime,timezone
root=pathlib.Path('/scratch/alpine/paco0228/ACE/results')
package=root/'delivery_qualified_replay_20261008_05/package'
out=root/'delivery_qualified_replay_20261023_10'
if out.exists():raise ValueError('proposed output10 exists; reconcile')
queue=subprocess.check_output(['squeue','-h','-u','paco0228','-o','%i|%j|%T|%Z'],text=True,timeout=20)
ace_queue=[row for row in queue.splitlines() if row.split('|')[1].lower().startswith('ace_') or '/ACE/' in row.split('|')[-1] or row.split('|')[-1].endswith('/ACE')]
if ace_queue:raise ValueError('ACE queued work requires reconciliation: '+repr(ace_queue))
journals=[]
for d in sorted(root.glob('delivery*replay*')):
 if not d.is_dir() or d.is_symlink():continue
 row={'directory':str(d),'files':{}}
 for name in ['submission_started.json','submission_result.json','replay_started.json','replay_execution.json','launcher_failure.json']:
  p=d/name
  if not p.exists():continue
  if p.is_symlink() or not p.is_file():raise ValueError('nonregular journal')
  b=p.read_bytes();obj=json.loads(b)
  row['files'][name]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),
    'status':obj.get('status'),'job_id':obj.get('job_id'),'exit_code':obj.get('exit_code'),
    'returncode':obj.get('returncode'),'submission_stdout':obj.get('stdout') if name=='submission_result.json' else None}
 if row['files']:journals.append(row)
raw=(package/'manifest.json').read_bytes();pin=hashlib.sha256(raw).hexdigest()
if pin!='b30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9':raise ValueError('manifest changed')
m=json.loads(raw);names=[e['path'] for e in m['files']]
if len(names)!=3893 or len(set(names))!=3893:raise ValueError('manifest membership')
size=0
for e in m['files']:
 rel=pathlib.PurePosixPath(e['path'])
 if rel.is_absolute() or '..' in rel.parts:raise ValueError('invalid member')
 q=package/e['path']
 if q.is_symlink() or not q.is_file() or not q.resolve().is_relative_to(package.resolve()):raise ValueError('invalid object')
 h=hashlib.sha256()
 with q.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b);size+=len(b)
 if h.hexdigest()!=e['sha256']:raise ValueError('changed member: '+e['path'])
allpaths=list(package.rglob('*'))
if any(p.is_symlink() for p in allpaths):raise ValueError('package symlink')
actual={str(p.relative_to(package)) for p in allpaths if p.is_file()}
if actual!=set(names)|{'manifest.json','manifest.sha256'}:raise ValueError('unexpected package membership')
sidecar=(package/'manifest.sha256').read_text().split()[0]
if sidecar!=pin:raise ValueError('manifest sidecar changed')
original=pathlib.Path('/scratch/alpine/paco0228/ACE/results/delivery_prospective_release_20261006_45ebeb89/bundle/project/scripts/research/runner_delivery_confirmation.py')
original_pin=hashlib.sha256(original.read_bytes()).hexdigest()
if original_pin!='2f93bacbd6c14c66a5aeac5daabcff50d47886eeec38885e6e83ab40d141baee':raise ValueError('original sampler changed')
fs=os.statvfs(root)
if fs.f_bavail*fs.f_frsize<2*1024**3:raise ValueError('insufficient scratch')
fd,tmp=tempfile.mkstemp(prefix='.ace-backend09-readiness-',dir=root)
try:
 with os.fdopen(fd,'wb') as f:f.write(b'ACE bounded replay readiness');f.flush();os.fsync(f.fileno())
 if pathlib.Path(tmp).read_bytes()!=b'ACE bounded replay readiness':raise ValueError('scratch readback')
finally:pathlib.Path(tmp).unlink()

installed_pins = {'/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/lib/python3.11/site-packages/torch/_C.cpython-311-x86_64-linux-gnu.so': '4349b125953ce9e0f7e0e94f5a78519107f15a234e1fb3703e86c12c405b09cd', '/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/lib/python3.11/site-packages/torch/lib/libtorch_python.so': 'a5863ec0a360320fff1932893266a86980d100060ba03b2c15815b8d3e2a85f9', '/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/lib/python3.11/site-packages/torch/_VF.py': '5c87e1f29223cef1b4c9259c392733e04d3aec31684fef1391c9801b9a189fde', '/scratch/alpine/paco0228/ACE/envs/neuronbench_py311/bin/python3.11': '89ed4dcb070e52f6023682e0295a0e8a402ecce708ebda33c9c6e91b8ab2b060'}
installed_pins.update({'/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/lib/python3.11/site-packages/torch/backends/__init__.py': '1667b68264dd4e6e2905d7845b1325819973a0b0678b31dc71d85766e7385de0', '/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/lib/python3.11/site-packages/torch/backends/cudnn/__init__.py': '74a6a25942eb32b576b9ca3547393a8236b59164d8c2eb5b872dcd862745ddac', '/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/lib/python3.11/site-packages/torch/backends/miopen/__init__.py': '445f5081a3b82c21f24ddc01a2a7292672ba0beb61c6ef7a3bb7d7e89702853b', '/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/lib/python3.11/site-packages/torch/backends/mkldnn/__init__.py': '44384600fe467a4f7d5695cb9285ab6846897e3ed14d4d13754b5824c248a2b3', '/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/lib/python3.11/site-packages/torch/backends/opt_einsum/__init__.py': '969d5271687e5540861b9e3605eb8fccc0a00eca48570625e12fdd5e40cdd05d', '/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/lib/python3.11/site-packages/torch/backends/quantized/__init__.py': 'bc3d83b3cd662ad406accac0632d45028fce3dfcb3be10ce6c5dbbea44ffb7e5', '/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/lib/python3.11/site-packages/torch/backends/xnnpack/__init__.py': '04d2fb08dc9395a4cdc6ee0954313c559822bb5ea644ee89a59790d0db3736bf'})
installed_pins['/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02/lib/python3.11/site-packages/six.py']='c51c91f703d3d4b3696c923cb5fec213e05e75d9215393befac7f2fa6a3904df'
for file,pinned in installed_pins.items():
 q=pathlib.Path(file)
 if q.is_symlink() or not q.is_file():raise ValueError('installed backing file is not regular')
 if hashlib.sha256(q.read_bytes()).hexdigest()!=pinned:raise ValueError('installed backing mismatch '+file)
accounting=subprocess.check_output(['sacct','-X','-j','33640933,34821094,34824953,34835547','--noheader','--parsable2','--format=JobIDRaw,JobID,JobName,State,ExitCode,AllocCPUS,ElapsedRaw,CPUTimeRAW,ReqMem,AllocTRES,Account,Partition'],text=True,timeout=20)

print(json.dumps({'installed_pins':installed_pins,'predecessor_sacct':accounting,'at':datetime.now(timezone.utc).isoformat(),'hostname':os.uname().nodename,'ace_queue':ace_queue,'journals':journals,'proposed_output_absent':str(out),'scratch_inode':root.stat().st_ino,'available_bytes':fs.f_bavail*fs.f_frsize,'df':subprocess.check_output(['df','-Pk',str(root)],text=True,timeout=10),'write_read_cleanup':True,'manifest_sha256':pin,'files':len(names),'bytes':size,'membership':len(actual),'original_sampler_sha256':original_pin,'actual_runtime_import_or_replay_performed':False},indent=2))

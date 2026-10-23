import hashlib,json,os,pathlib,stat,subprocess,tempfile
from datetime import datetime,timezone
root=pathlib.Path('/scratch/alpine/paco0228/ACE/results')
package=root/'delivery_qualified_replay_20261008_05/package'
out=root/'delivery_qualified_replay_20261023_07'
if out.exists():raise ValueError('proposed output07 exists; reconcile')
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
fd,tmp=tempfile.mkstemp(prefix='.ace-vf07-readiness-',dir=root)
try:
 with os.fdopen(fd,'wb') as f:f.write(b'ACE bounded replay readiness');f.flush();os.fsync(f.fileno())
 if pathlib.Path(tmp).read_bytes()!=b'ACE bounded replay readiness':raise ValueError('scratch readback')
finally:pathlib.Path(tmp).unlink()
print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'hostname':os.uname().nodename,'ace_queue':ace_queue,'journals':journals,'proposed_output_absent':str(out),'scratch_inode':root.stat().st_ino,'available_bytes':fs.f_bavail*fs.f_frsize,'df':subprocess.check_output(['df','-Pk',str(root)],text=True,timeout=10),'write_read_cleanup':True,'manifest_sha256':pin,'files':len(names),'bytes':size,'membership':len(actual),'original_sampler_sha256':original_pin,'actual_runtime_import_or_replay_performed':False},indent=2))

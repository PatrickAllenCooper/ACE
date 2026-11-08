"""Exclusive artificial paired-campaign CPU phase after source and packet review."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shlex,subprocess,sys
REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_campaign_artificial_20261108/cpu02'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-cpu-20261108-02')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_02'
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
REVIEW=REPO/'docs/development/guidance/reviews/ace_llm_intervention_campaign_cpu_packet02_review_2026-11-08.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 raw=(json.dumps(v,indent=2,sort_keys=True)+'\n').encode()
 for root in [EXT,LOCAL]:
  with (root/n).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def remote(name,source):
 p=subprocess.run(['bash',HELPER,'run','python3 -'],input=source,capture_output=True,text=True,timeout=60)
 save(name,{'python_source':source,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 if p.returncode:raise RuntimeError(name+' failed, reconcile retained evidence before any new action')
 return json.loads(p.stdout)
if (LOCAL/'submission.intent.json').exists():raise RuntimeError('submission intent already exists; no blind retry')
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
meta=json.loads((LOCAL/'packet-metadata.json').read_bytes())
files=['inputs/'+n for n in meta['input_files']]+['job.sbatch','packet-metadata.json','intent-provenance.json','build_packet.py','launch_once.py']
for n in files:
 committed=subprocess.check_output(['git','show',revision+':'+str((LOCAL/n).relative_to(REPO))],cwd=REPO)
 if committed!=(LOCAL/n).read_bytes():raise RuntimeError('commit does not bind '+n)
if subprocess.check_output(['git','show',revision+':'+str(REVIEW.relative_to(REPO))],cwd=REPO)!=REVIEW.read_bytes():raise RuntimeError('review not committed')
pins={n:sha(LOCAL/n) for n in files}
intent={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packet_commit':revision,'source_commit':meta['source_commit'],'pins':pins,'cpu_packet_sha256':meta['cpu_packet_sha256'],'review_sha256':sha(REVIEW),'launcher_sha256':sha(Path(__file__)),'remote_root':ROOT,'controller_pid_local':os.getpid(),'resources':meta['resources'],'admission':meta['admission'],'blind_retry_forbidden':True}
locks=[]
try:
 for n in ['ace-delivery-controller.lock','ace-foundation-controller.lock']:
  f=open(EXT.parent/n,'r+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 save('controller.json',{'held':[f.name for f in locks],'pid':os.getpid(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 common='''import pathlib,json,hashlib,subprocess,datetime,os
root=pathlib.Path(ROOT)
def run(c):
 p=subprocess.run(c,capture_output=True,text=True,timeout=20)
 if p.returncode:raise RuntimeError(json.dumps({'command':c,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
 return p.stdout
def reconcile():
 queue=run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'])
 if any('|ace_llm' in l for l in queue.splitlines()):raise RuntimeError('related active job')
 a=run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'])
 rows=[l.split('|') for l in a.splitlines() if '|ace_llm' in l]
 if len(rows)!=32 or any(v[2].split()[0] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for v in rows):raise RuntimeError('prior job state changed')
 if sum(int(v[4])*int(v[5]) for v in rows)!=9232 or sum(int(v[5]) for v in rows if 'gres/gpu:' in v[7])!=1347:raise RuntimeError('accounting changed')
 journals={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for d in root.parent.glob('ace_llm_intervention*') for p in d.glob('submission.*.json')}
 if len(journals)!=66:raise RuntimeError('prior submission journal count changed')
 partition=run(['scontrol','show','partition','acpu','-o'])
 limits=dict(part.split('=',1) for part in partition.split() if '=' in part)
 cap=limits.get('MaxMemPerCPU','')
 if limits.get('PartitionName')!='acpu' or not cap.isdigit() or int(cap)<3840:raise RuntimeError('two-CPU memory ratio changed')
 assoc=run(['sacctmgr','-n','-P','show','assoc','where','account=ucb736_asc2','user=paco0228','format=Cluster,Account,User,Partition,MaxJobs,QOS'])
 ar=[l.split('|') for l in assoc.splitlines() if l.strip()]
 if len(ar)!=1 or ar[0][:3]!=['alpine','ucb736_asc2','paco0228'] or ar[0][4]=='0' or 'cpu-normal' not in ar[0][5].split(','):raise RuntimeError('renewed association changed')
 manifest=pathlib.Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_staged_runtime_20261107_01/qualification/packed/manifest.json')
 archive=manifest.with_name('runtime.tar')
 if hashlib.sha256(manifest.read_bytes()).hexdigest()!='b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a' or archive.stat().st_size!=8438650880:raise RuntimeError('predecessor archive metadata drift')
 return {'partition':partition,'max_memory_per_cpu_mib':int(cap),'queue':queue,'prior_accounting':rows,'prior_journals':journals,'association':assoc,'predecessor_manifest_sha256':'b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a','predecessor_archive_size':archive.stat().st_size,'bulk_archive_byte_authentication':'allocated stage before copied runtime executes'}
def write(n,v):
 with (root/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write(chr(10));f.flush();os.fsync(f.fileno())
'''.replace('ROOT',repr(ROOT),1)
 ready=remote('readiness_transport.json',common+'''
r=reconcile()
if root.parent.stat().st_uid!=os.getuid():raise RuntimeError('scratch owner differs')
root.mkdir();(root/'logs').mkdir();(root/'inputs').mkdir()
p=root/'write-read-probe'
with p.open('xb') as f:f.write(b'ACE paired campaign CPU readiness\\n');f.flush();os.fsync(f.fileno())
if p.read_bytes()!=b'ACE paired campaign CPU readiness\\n':raise RuntimeError('scratch readback differs')
p.unlink();fs=os.statvfs(root)
r.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scratch={'uid':root.stat().st_uid,'mode':oct(root.stat().st_mode),'free_bytes':fs.f_bavail*fs.f_frsize,'df':run(['df','-h',str(root)]),'write_read_cleanup':True})
write('readiness.json',r);print(json.dumps(r))
''')
 save('readiness.json',ready)
 for subgroup in ([n for n in pins if n.startswith('inputs/')],[n for n in pins if not n.startswith('inputs/')]):
  target=ROOT+('/inputs/' if subgroup[0].startswith('inputs/') else '/')
  p=subprocess.run(['scp','-o','ControlPath=/tmp/defab-curc.sock','-o','BatchMode=yes',*[str(LOCAL/n) for n in subgroup],'paco0228@login.rc.colorado.edu:'+target],capture_output=True,text=True,timeout=60)
  save('transfer-'+('inputs' if subgroup[0].startswith('inputs/') else 'root')+'.json',{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
  if p.returncode:raise RuntimeError('transfer failed before submission')
 verified=remote('verification_transport.json',common+'\npins='+repr(pins)+'''\nfor n,h in pins.items():
 if hashlib.sha256((root/n).read_bytes()).hexdigest()!=h:raise RuntimeError('remote pin differs '+n)
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':pins}))
''')
 save('remote_verified.json',verified)
 save('submission.intent.json',intent)
 result=remote('submission_transport.json',common+'\nintent='+repr(intent)+'''\nif (root/'submission.intent.json').exists() or (root/'submission.result.json').exists():raise RuntimeError('existing submission journal')
intent.update(reconcile())
for n,h in intent['pins'].items():
 if hashlib.sha256((root/n).read_bytes()).hexdigest()!=h:raise RuntimeError('remote input drift')
intent['remote_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write('submission.intent.json',intent)
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'intent_sha256':hashlib.sha256((root/'submission.intent.json').read_bytes()).hexdigest(),'job_id':None}
try:
 p=subprocess.run(['sbatch','--parsable',str(root/'job.sbatch')],capture_output=True,text=True,timeout=25)
 job=p.stdout.strip().split(';')[0];r.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,job_id=job if p.returncode==0 and job.isdigit() else None)
except BaseException as exc:r.update(ambiguous=True,exception_type=type(exc).__name__,message=str(exc))
write('submission.result.json',r);print(json.dumps(r))
''')
 save('submission.result.json',result)
 if result['job_id'] is None:raise RuntimeError('failed or ambiguous submission, never blind retry')
 print(json.dumps(result,indent=2))
except BaseException as exc:
 save('launcher_failure.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'type':type(exc).__name__,'message':str(exc),'blind_retry_forbidden':True});raise
finally:
 for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
 save('controller_released.json',{'pid':os.getpid(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'released':True})

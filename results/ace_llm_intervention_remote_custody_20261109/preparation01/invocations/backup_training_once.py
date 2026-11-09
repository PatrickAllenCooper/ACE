"""One prospective storage-only copy of already accepted training evidence."""
from pathlib import Path
import argparse,hashlib,json,types,sys,subprocess,fcntl,os,datetime,time,resource
REPO=Path('/Users/pat/code/ACE');BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
def need(v,m):
 if not v:raise ValueError(m)
def sha(v):return hashlib.sha256(v).hexdigest()
def raw(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def write(p,v):
 with p.open('xb') as f:f.write(v);f.flush();os.fsync(f.fileno())
ap=argparse.ArgumentParser();ap.add_argument('--helper',required=True);ap.add_argument('--helper-sha256',required=True);a=ap.parse_args()
source=Path(a.helper).read_bytes();need(sha(source)==a.helper_sha256,'helper pin differs')
b=types.ModuleType('remote_custody');b.__file__=str(Path(a.helper).resolve());sys.modules[b.__name__]=b;exec(compile(source,b.__file__,'exec'),b.__dict__);b.bind_source(source,a.helper_sha256)
old=REPO/'results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/inventory_transport.json'
data=old.read_bytes();need(sha(data)=='0e4fc693b224be3e6d91f63a824f830e3d315a794e4331b64fa8a162acbc8db3','accepted original inventory pin differs')
obs=json.loads(json.loads(data)['stdout']);need(obs['terminal_gate'] is True and not obs.get('observation_error'),'accepted terminal inventory absent')
basepath=REPO/'results/ace_llm_intervention_scientific_training_gpu_20261108/acceptance01/accounting.json'
baseraw=basepath.read_bytes();need(sha(baseraw)=='b0cf5af00e63afd56bf96fe9a20be538ef28d2bdf499fd0f0782b71deeebb5ae','accepted baseline differs');base=json.loads(baseraw)
r='/scratch/alpine/paco0228/ACE/results/'
roots={'launch':r+'ace_llm_intervention_scientific_training_gpu_20261108_01','cpu':r+'ace_llm_intervention_scientific_tokenization_20261108_02','failed-token01':r+'ace_llm_intervention_scientific_tokenization_20261108_01','common':r+'ace_llm_intervention_scientific_collection_20261108_02','failed01':r+'ace_llm_intervention_scientific_collection_20261108_01','namespace.json':r+'.ace-scientific-screen-20261107.namespace-recovery02.json','original-namespace.json':r+'.ace-scientific-screen-20261107.namespace.json'}
remote='/projects/paco0228/ACE_evidence/ace-llm-training-20261109-01'
out=BASE/'ACE-LLM-remote-training-backup-20261109-01';locks=[];started=time.monotonic();u=resource.getrusage(resource.RUSAGE_CHILDREN)
owned=False
try:
 for n in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
  f=open(BASE/n,'r+')
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BaseException:f.close();raise
  locks.append(f)
 out.mkdir(mode=0o700);owned=True;(out/'remote').mkdir(mode=0o700)
 write(out/'intent.json',raw(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=a.helper_sha256,original_inventory_sha256=sha(data),baseline_sha256=sha(baseraw),backup_root=remote,source_roots=roots,scientific_execution=False,originals_modified=False,retry_allowed=False)))
 write(out/'remote_custody.py',source)
 write(out/'invocation.py',Path(__file__).read_bytes())
 pre='BASELINE='+repr(base)+'\nROOT='+repr(r)+'''\nfrom pathlib import Path
import subprocess,hashlib,json
q=subprocess.run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],capture_output=True,text=True,timeout=25);assert q.returncode==0,q.stderr
assert not any(l.split('|')[1].lower().startswith('ace') for l in q.stdout.splitlines() if l)
a=subprocess.run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'],capture_output=True,text=True,timeout=25);assert a.returncode==0,a.stderr
rows=[l.split('|') for l in a.stdout.splitlines() if '|ace_llm' in l]
def norm(rows):
 return sorted([r[:4]+[int(r[4]),int(r[5]),r[6],sorted(r[7].split(','))] for r in rows],key=lambda r:int(r[0]))
assert norm(rows)==norm(BASELINE['related_accounting']) and len(rows)==45
j={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for d in Path(ROOT).glob('ace_llm_intervention*') for p in d.glob('submission.*.json')}
assert j==BASELINE['journal_pins'] and len(j)==92
print(json.dumps({'queue':q.stdout,'related_accounting':rows,'journal_pins':j,'no_active_ace_job':True}))
'''
 result=subprocess.run(['bash',HELPER,'run','python3 -I -B -'],input=pre,capture_output=True,text=True,timeout=90)
 write(out/'preflight.json',raw(dict(command=['bash',HELPER,'run','python3 -I -B -'],python_source=pre,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)))
 need(result.returncode==0,'original jobs/journals changed or transport failed')
 receipt=b.collect_backup(observation=obs,roots=roots,backup_root=remote,local_root=out/'remote',helper_path=HELPER,log_dir=out/'operations')
 need(receipt['full_backup_verified'] is True and receipt['full_inventory']==dict(files=obs['files'],directories=sorted(obs['directories'])),'complete backup differs')
 write(out/'remote-custody.json',raw(receipt))
 v=resource.getrusage(resource.RUSAGE_CHILDREN)
 write(out/'execution.json',raw(dict(elapsed_seconds=time.monotonic()-started,child_cpu_seconds=v.ru_utime+v.ru_stime-u.ru_utime-u.ru_stime,peak_child_rss_bytes=v.ru_maxrss,scope='Local orchestration child CPU. Remote copying/hashing and earlier preparation are excluded. Not scientific allocation.',scientific_execution=False,receipt_sha256=sha(raw(receipt)))))
 print(json.dumps(dict(output=str(out),receipt_sha256=sha(raw(receipt)),backup_root=remote)))
except BaseException as e:
 if owned:
  try:write(out/'failure.json',raw(dict(type=type(e).__name__,message=str(e),retry_allowed=False,partial_preserved=True)))
  except BaseException:pass
 raise
finally:
 for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()

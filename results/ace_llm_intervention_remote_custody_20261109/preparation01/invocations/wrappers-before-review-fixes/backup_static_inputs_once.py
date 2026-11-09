"""Opaque preservation of the already frozen runtime archive and base model."""
from pathlib import Path
import argparse,hashlib,json,types,sys,subprocess,fcntl,os,datetime
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
data=(REPO/'results/ace_llm_intervention_scientific_campaign_cpu_20261108/launch01/inputs/packet.json').read_bytes()
need(sha(data)=='ad046a726e3a3a7744e54a860f8cf1a30947a198e8be3bc25ae06ec7b2909344','frozen stage pin differs');stage=json.loads(data)
roots={'runtime.tar':stage['archive']['path'],'runtime-manifest.json':stage['manifest']['path'],'model':stage['model_directory']}
expected={'runtime.tar':stage['archive']['sha256'],'runtime-manifest.json':stage['manifest']['sha256'],**{'model/'+n:h for n,h in stage['model_files'].items()}}
baseraw=(REPO/'results/ace_llm_intervention_scientific_training_gpu_20261108/acceptance01/accounting.json').read_bytes()
need(sha(baseraw)=='b0cf5af00e63afd56bf96fe9a20be538ef28d2bdf499fd0f0782b71deeebb5ae','accepted baseline differs');baseline=json.loads(baseraw)
remote='/projects/paco0228/ACE_evidence/ace-llm-static-inputs-20261109-01'
out=BASE/'ACE-LLM-remote-static-inputs-backup-20261109-01';locks=[]
try:
 for n in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
  f=open(BASE/n,'r+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 out.mkdir(mode=0o700);(out/'remote').mkdir(mode=0o700)
 write(out/'intent.json',raw(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=a.helper_sha256,stage_sha256=sha(data),backup_root=remote,source_roots=roots,expected_file_pins=expected,scientific_execution=False,retry_allowed=False)))
 write(out/'remote_custody.py',source)
 write(out/'invocation.py',Path(__file__).read_bytes())
 # Sizes are observations, not authentication. The helper must verify the independently pinned bytes.
 code='ROOTS='+repr(roots)+'\nEXPECTED='+repr(expected)+'\nBASELINE='+repr(baseline)+'''\nfrom pathlib import Path
import os,stat,json,subprocess,hashlib
q=subprocess.run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],capture_output=True,text=True,timeout=25);assert q.returncode==0,q.stderr
assert not any(l.split('|')[1].lower().startswith('ace') for l in q.stdout.splitlines() if l)
a=subprocess.run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'],capture_output=True,text=True,timeout=25);assert a.returncode==0,a.stderr
rows=[l.split('|') for l in a.stdout.splitlines() if '|ace_llm' in l]
def norm(rows):
 return sorted([r[:4]+[int(r[4]),int(r[5]),r[6],sorted(r[7].split(','))] for r in rows],key=lambda r:int(r[0]))
assert norm(rows)==norm(BASELINE['related_accounting']) and len(rows)==45
j={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for d in Path('/scratch/alpine/paco0228/ACE/results').glob('ace_llm_intervention*') for p in d.glob('submission.*.json')}
assert j==BASELINE['journal_pins'] and len(j)==92
files={}
for n,h in EXPECTED.items():
 parts=n.split('/',1);p=Path(ROOTS[parts[0]])
 if len(parts)==2:p=p/parts[1]
 assert p==p.resolve() and not p.is_symlink()
 s=p.stat();assert stat.S_ISREG(s.st_mode)
 files[n]={'bytes':s.st_size,'sha256':h}
assert {p.name for p in Path(ROOTS['model']).iterdir()}=={n.split('/',1)[1] for n in EXPECTED if n.startswith('model/')}
print(json.dumps({'files':files,'directories':['model'],'expected_hashes_origin':'Independently pinned frozen stage inputs, not this stat command','queue':q.stdout,'related_accounting':rows,'journal_pins':j}))
'''
 p=subprocess.run(['bash',HELPER,'run','python3 -'],input=code,text=True,capture_output=True,timeout=90)
 write(out/'stat-transport.json',raw(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,python_source=code)))
 need(p.returncode==0,'static input stat/queue failed')
 observation=json.loads(p.stdout)
 need(observation['files']['runtime.tar']['bytes']==8438650880,'known runtime archive size differs')
 receipt=b.collect_backup(observation=observation,roots=roots,backup_root=remote,local_root=out/'remote',helper_path=HELPER,log_dir=out/'operations')
 need(receipt['full_backup_verified'] is True and receipt['full_inventory']=={k:observation[k] for k in ('files','directories')},'static archive backup differs')
 write(out/'remote-custody.json',raw(receipt));print(json.dumps(dict(output=str(out),receipt_sha256=sha(raw(receipt)),backup_root=remote)))
except BaseException as e:
 if out.exists():
  try:write(out/'failure.json',raw(dict(type=type(e).__name__,message=str(e),retry_allowed=False,partial_preserved=True)))
  except BaseException:pass
 raise
finally:
 for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()

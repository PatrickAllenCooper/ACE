"""Retain serial bounded public observations; stop on terminal or any failed observation.
No submission, collector invocation, cancellation, or private outcome decoding.
"""
from pathlib import Path
import datetime,hashlib,json,os,stat,subprocess,sys,time
C=Path(__file__).resolve().parent
EXT=Path((C/'external-path.txt').read_text().strip())
BASE=EXT.parent
TEMPLATE=json.loads((C/'monitor01.command.json').read_bytes())
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def raw(v):return (json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
def put(name,v):
 data=raw(v)
 for root in (C,EXT):
  with (root/name).open('xb') as f:f.write(data)
def copynew(src,dst):
 dst.mkdir(mode=0o700,parents=True,exist_ok=False)
 for p in sorted(src.rglob('*')):
  s=p.lstat();assert not stat.S_ISLNK(s.st_mode)
  q=dst/p.relative_to(src)
  if stat.S_ISDIR(s.st_mode):q.mkdir(mode=0o700)
  else:
   assert stat.S_ISREG(s.st_mode) and s.st_nlink==1
   data=p.read_bytes()
   with q.open('xb') as f:f.write(data)
   assert q.read_bytes()==data
previous=C/'monitor01/run.json'
for index in range(2,20):
 last=datetime.datetime.fromisoformat(json.loads(previous.read_bytes())['ended_utc']).timestamp()
 while True:
  remaining=last+300-time.time()
  if remaining<=0:break
  time.sleep(min(60,remaining))
 label=f'monitor{index:02d}'
 evidence=BASE/f'ACE-LLM-finalizer-monitor-20261110-{index:02d}'
 config=json.loads(json.dumps(TEMPLATE));config['evidence']=str(evidence)
 config['command'][config['command'].index('--evidence-directory')+1]=str(evidence)
 put(label+'.command.json',config)
 p=subprocess.run([sys.executable,'-I','-B',str(C/'execute_recorded.py'),label],cwd=TEMPLATE['cwd'])
 copynew(C/label,EXT/label)
 if evidence.exists():
  copynew(evidence,C/('evidence-'+label));copynew(C/('evidence-'+label),EXT/('evidence-'+label))
 previous=C/label/'run.json'
 if p.returncode:
  put('monitor-loop-stop.json',dict(reason='failed_observation',label=label,returncode=p.returncode,retry_submission=False))
  print(json.dumps(dict(event='monitor_failed',label=label,returncode=p.returncode)),flush=True);break
 d=json.loads((evidence/'stdout.log').read_bytes())
 rows=[line.split('|') for o in d['observations'] if o['command'][0]=='sacct' and o.get('returncode')==0 for line in o['stdout'].splitlines() if line.startswith('35533604|')]
 assert len(rows)==1,'exact parent scheduler row required'
 state=rows[0][2].split()[0]
 put('status-'+label+'.json',dict(utc=d['utc'],job_id=d['job_id'],observations=d['observations'],markers=d['evidence'],acceptance=False))
 print(json.dumps(dict(event='observation',label=label,utc=d['utc'],parent_row=rows[0],present=[n for n,v in d['evidence'].items() if v['present']])),flush=True)
 if state in TERMINAL:
  put('monitor-loop-stop.json',dict(reason='terminal',label=label,state=state,job_id='35533604',utc=d['utc'],acceptance=False))
  break
else:
 put('monitor-loop-stop.json',dict(reason='bounded_observation_count_exhausted',job_id='35533604',acceptance=False))

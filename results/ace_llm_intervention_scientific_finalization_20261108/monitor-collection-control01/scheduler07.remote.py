PATHS=['finalization/start.json', 'finalization/attempt.json', 'finalization/failure.json', 'finalization/allocation.json', 'finalization/stage.json', 'finalization/handoff.json', 'finalization/worker.started.json', 'finalization/worker.cleanup.json', 'finalization/worker.resources.json', 'finalization/worker/start.json', 'finalization/worker/work-start.json', 'finalization/worker/runtime-before.json', 'finalization/worker/runtime-ready.json', 'finalization/worker/runtime-after.json', 'finalization/worker/terminal.json', 'finalization/worker/attempt.json', 'finalization/worker/failure.json']
from pathlib import Path
import datetime,hashlib,json,os,signal,stat,subprocess
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_finalization_20261108_01'
CAMPAIGN='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01/cpu/worker/artifacts/campaign'
JOB='35533604'
INTENT='7918cd533a5b72f24143d91a251296c4dca3a701043db75bf458aed8bb21512d'
def deadline(signum,frame):raise TimeoutError('bounded scheduler and marker-stat observation')
signal.signal(signal.SIGALRM,deadline);signal.alarm(45)
p=Path(ROOT)/'submission.intent.json'
with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
 s=os.fstat(f.fileno())
 if not stat.S_ISREG(s.st_mode) or s.st_size>1024**2:raise ValueError('bounded original intent required')
 data=f.read(1024**2+1)
if hashlib.sha256(data).hexdigest()!=INTENT:raise ValueError('original intent differs')
obs=[]
for cmd in [['squeue','-j',JOB,'-h','-o','%i|%j|%T|%M|%R'],['sacct','-j',JOB,'--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,ReqMem,AllocTRES%180,MaxRSS','-n','-P'],['scontrol','show','job',JOB,'-o']]:
 try:
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=12)
  obs.append(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
 except subprocess.TimeoutExpired as e:
  txt=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x
  obs.append(dict(command=cmd,timeout=True,stdout=txt(e.stdout),stderr=txt(e.stderr)))
markers={}
for name,p in [(n,Path(ROOT)/n) for n in PATHS]+[('campaign/'+n,Path(CAMPAIGN)/n) for n in ('finalization-intent.json','finalization.candidate.json','finalization.json')]:
 try:
  st=p.lstat();markers[name]=dict(present=True,bytes=st.st_size,mtime_ns=st.st_mtime_ns,regular=stat.S_ISREG(st.st_mode),symlink=stat.S_ISLNK(st.st_mode))
 except FileNotFoundError:markers[name]=dict(present=False)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=JOB,observations=obs,evidence=markers,marker_payloads_read=False,acceptance=False)))

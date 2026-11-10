ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_finalization_20261108_01'
JOB='35533604'
INTENT_SHA='7918cd533a5b72f24143d91a251296c4dca3a701043db75bf458aed8bb21512d'
CAMPAIGN='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01/cpu/worker/artifacts/campaign'

from pathlib import Path
import os,stat,hashlib,json,subprocess,datetime,signal
root=Path(ROOT)
def need(ok,msg):
 if not ok:raise ValueError(msg)
def timeout(signum,frame):raise TimeoutError('bounded monitor remote deadline')
signal.signal(signal.SIGALRM,timeout);signal.alarm(45)
def read(p,cap=1024*1024):
 need(not p.is_symlink(),'evidence symlink forbidden')
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  s=os.fstat(f.fileno());need(stat.S_ISREG(s.st_mode) and s.st_size<=cap,'bounded ordinary evidence required')
  value=f.read(cap+1)
 need(len(value)<=cap,'oversize evidence');return value
def mark(p):
 if not p.exists():return dict(present=False)
 data=read(p);return dict(present=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),mtime_ns=p.stat().st_mtime_ns)
need(hashlib.sha256(read(root/'submission.intent.json')).hexdigest()==INTENT_SHA,'original remote intent differs')
observations=[]
for command in [['squeue','-j',JOB,'-h','-o','%i|%j|%T|%M|%R'],
 ['sacct','-j',JOB,'--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,ReqMem,AllocTRES%180,MaxRSS','-n','-P'],
 ['scontrol','show','job',JOB,'-o']]:
 try:
  p=subprocess.run(command,capture_output=True,text=True,timeout=12)
  observations.append(dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
 except subprocess.TimeoutExpired as exc:
  txt=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x
  observations.append(dict(command=command,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr)))
paths=['finalization/start.json','finalization/attempt.json','finalization/failure.json','finalization/allocation.json',
 'finalization/stage.json','finalization/handoff.json','finalization/worker.started.json','finalization/worker.cleanup.json',
 'finalization/worker.resources.json','finalization/worker/start.json','finalization/worker/work-start.json',
 'finalization/worker/runtime-before.json','finalization/worker/runtime-ready.json','finalization/worker/runtime-after.json',
 'finalization/worker/terminal.json','finalization/worker/attempt.json','finalization/worker/failure.json']
evidence={name:mark(root/name) for name in paths}
# Publication may precede a failing outer cleanup: presence is never acceptance or permission to retry.
campaign=Path(CAMPAIGN)
for name in ('finalization-intent.json','finalization.candidate.json','finalization.json'):
 evidence['campaign/'+name]=mark(campaign/name)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=JOB,
 observations=observations,evidence=evidence,acceptance=False,scientific_qualified=False,
 note='Bounded opaque marker hashes only. Preserve failed/partial reports and all original campaign outputs; terminal custody and independent audit required.')))


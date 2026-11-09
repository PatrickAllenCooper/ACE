ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'
JOB='35498218'
INTENT_SHA='46998e9b8a2d5cc1017b2540cebd533ef1a2637c7b0dd440eb81230eb99b4ca4'

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
paths=['gpu/terminal.json','gpu/failure.json','gpu/allocation.json','gpu/stage.json','gpu/handoff.json',
 'gpu/controller/admitted.json','gpu/controller/phase-result.candidate.json','gpu/controller/phase/start.json',
 'gpu/controller/phase/terminal.json','gpu/controller/phase/failure.json','gpu/controller/phase/cleanup-hold-intent.json']
evidence={name:mark(root/name) for name in paths};telemetry={}
for pid in ('frozen_llm','sft_llm','dpo_llm'):
 phase=root/'gpu/controller/phase'/pid
 for name in ('summary.json','fallback-cleanup.json','post-exit-gpu.json'):
  p=phase/name;evidence[str(p.relative_to(root))]=mark(p)
 for sub in ('worker','launch'):
  for name in ('terminal.json','failure.json'):
   p=root/'gpu/services'/pid/sub/name;evidence[str(p.relative_to(root))]=mark(p)
 samples=sorted(phase.glob('telemetry-*.json'))
 if samples:
  p=samples[-1];evidence[str(p.relative_to(root))]=mark(p)
  try:
   # This producer file contains public counts/process samples, never private values or token IDs.
   row=json.loads(read(p,256*1024));keys=('utc','outer_elapsed_seconds','service_elapsed_seconds','policy_id','registered_pid',
    'gpu_process_sample','positive_generations','first_verified_monotonic','last_gpu_monotonic','last_generation_monotonic','uncertainties')
   telemetry[pid]={k:row.get(k) for k in keys}
  except (OSError,ValueError,KeyError,TypeError) as exc:telemetry[pid]=dict(uncertainty=type(exc).__name__)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=JOB,
 observations=observations,evidence=evidence,public_telemetry=telemetry,utilization_percent=None,
 note='Process memory and recorded positive generation evidence are distinct signals; missing telemetry is uncertainty. No qualification from presence or exit0.',
 acceptance=False,scientific_qualified=False)))

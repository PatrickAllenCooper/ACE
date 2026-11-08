ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01'
JOB='35420219'
INTENT_SHA='2ee204a5bb644bacef1d033fc2fcd7df5b9728dcffa2af03308824c275f1e03f'

import pathlib,hashlib,json,subprocess,datetime
root=pathlib.Path(ROOT)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if sha(root/'submission.intent.json')!=INTENT_SHA:raise RuntimeError('remote intent differs')
commands=[['squeue','-j',JOB,'-h','-o','%i|%j|%T|%M|%R'],
 ['sacct','-j',JOB,'--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,ReqMem,AllocTRES%180,MaxRSS','-n','-P'],
 ['scontrol','show','job',JOB,'-o']]
observations=[]
for command in commands:
 try:
  p=subprocess.run(command,capture_output=True,text=True,timeout=12)
  observations.append(dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
 except subprocess.TimeoutExpired as exc:
  def txt(v):return v.decode(errors='replace') if isinstance(v,bytes) else v
  observations.append(dict(command=command,timeout=True,message=str(exc),stdout=txt(exc.stdout),stderr=txt(exc.stderr)))
paths=['training/terminal.json','training/failure.json','training/cleanup.json','training/cleanup-failure.json',
       'training/first-verified-gpu-work.json','training/post-exit-gpu.json',
       'training/session/worker/artifacts/first-gpu-work.json',
       'training/session/worker/artifacts/terminal.json','training/session/worker/artifacts/failure.json',
       'training/session/worker/artifacts/worker-summary.json','training/session/worker/artifacts/fit/summary.json']
evidence={}
for name in paths:
 p=root/name
 if p.is_symlink():raise RuntimeError('evidence symlink')
 evidence[name]=dict(present=p.exists(),sha256=sha(p) if p.is_file() else None)
for prefix in ('gpu-sample-','progress-'):
 samples=sorted((root/'training').glob(prefix+'*.json'))
 if samples:
  p=samples[-1]
  if p.is_symlink():raise RuntimeError('telemetry symlink')
  evidence[str(p.relative_to(root))]=dict(present=True,sha256=sha(p),bytes=p.stat().st_size,
      mtime_ns=p.stat().st_mtime_ns,observation_only=True)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=JOB,
    observations=observations,evidence=evidence,acceptance=False)))

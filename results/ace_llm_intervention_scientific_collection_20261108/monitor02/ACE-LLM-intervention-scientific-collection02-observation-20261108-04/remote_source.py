ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_02'
JOB='35409614'
INTENT_SHA='a9ddf90cf92515ff9670c61c904290226b4d9e1a38a00cee50f3526aa843c87c'

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
paths=['collection/terminal.json','collection/failure.json','collection/publication-interruption.json',
       'collection/session/worker/terminal.json','collection/session/worker/failure.json',
       'collection/session/worker/task/terminal.json','collection/session/worker/task/failure.json',
       'collection/session/worker/task/collection/terminal.json','collection/session/worker/task/collection/failure.json']
evidence={}
for name in paths:
 p=root/name
 if p.is_symlink():raise RuntimeError('evidence symlink')
 evidence[name]=dict(present=p.exists(),sha256=sha(p) if p.is_file() else None)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=JOB,
    observations=observations,evidence=evidence,acceptance=False)))

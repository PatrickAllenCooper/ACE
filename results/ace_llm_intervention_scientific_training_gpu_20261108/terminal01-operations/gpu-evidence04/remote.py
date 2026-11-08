from pathlib import Path
import json,hashlib,datetime
r=Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01')
paths=[r/'training/first-verified-gpu-work.json',r/'training/session/worker/artifacts/first-gpu-work.json',r/'training/child-start.json']
for prefix in ('gpu-sample-','progress-'):
 p=sorted((r/'training').glob(prefix+'*.json'))
 if p:paths.append(p[-1])
out={}
for p in paths:
 if p.is_file() and not p.is_symlink():
  assert p.stat().st_size<131072
  b=p.read_bytes();out[str(p.relative_to(r))]={'sha256':hashlib.sha256(b).hexdigest(),'data':json.loads(b)}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evidence':out,'acceptance':False}))

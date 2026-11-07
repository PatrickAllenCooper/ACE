"""New fabricated account migration controls, no imports/workers/models."""
from pathlib import Path
from unittest.mock import patch
import hashlib,json,os,signal,sys,types
root=Path(__file__).parent
source=(root/'probe.py').read_bytes()
module=types.ModuleType('renewed_probe');module.__file__=str(root/'probe.py');exec(compile(source,module.__file__,'exec'),module.__dict__)
class WorkerBoundary(RuntimeError):pass
out=root/'account_guard_fixtures';out.mkdir(exist_ok=False)
base='JobId=42 Account=ucb736_asc2 Partition=acpu QOS=cpu-normal NumCPUs=1 NumTasks=1 NumNodes=1 TimeLimit=00:08:00 Requeue=0 AllocTRES=cpu=1,mem=3G,node=1'
results=[]
for name,fields,accepted in [('renewed',base,True),('expired',base.replace('ucb736_asc2','ucb736_asc1'),False),('wrong_cpu',base.replace('NumCPUs=1','NumCPUs=2'),False),('gpu',base+',gres/gpu=1',False)]:
 handlers={s:signal.getsignal(s) for s in [signal.SIGTERM,signal.SIGINT]}
 argv=['probe.py','--service','unused','--service-sha256','0'*64,'--freeze','unused','--freeze-sha256','0'*64,'--output',str(out/name)]
 try:
  with patch.object(sys,'argv',argv),patch.dict(os.environ,{'SLURM_JOB_ID':'42'}),patch.object(module.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout=fields,stderr='')),patch.object(module.subprocess,'Popen',side_effect=WorkerBoundary('fabricated stop before worker')) as spawn:
   try:module.main();raise AssertionError('guard or worker boundary must stop')
   except WorkerBoundary:assert accepted and spawn.call_count==1
   except ValueError as exc:assert not accepted and spawn.call_count==0 and str(exc)=='actual allocation differs from fixed CPU-only bound'
   results.append({'case':name,'allocation_admitted':accepted,'actual_worker_started':False,'popen_boundary_calls':spawn.call_count})
 finally:
  for s,h in handlers.items():signal.signal(s,h)
(root/'account_guard_results.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source).hexdigest(),'cases':results,'scientific_qualification':False},indent=2)+'\n')
print(json.dumps(results))

"""Retain one exclusive local artificial attempt; never invoke remote/main/transport."""
import hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
attempt=root/sys.argv[1];attempt.mkdir(exist_ok=False)
source=attempt/'source';source.mkdir();(source/'predecessor527d').mkdir()
pins={}
for relative in ('remote_custody.py','test_traversal.py','predecessor527d/remote_custody.py'):
 data=(root/relative).read_bytes();(source/relative).write_bytes(data);pins[relative]=hashlib.sha256(data).hexdigest()
env=dict(os.environ,TRAVERSAL_FIXTURES=str(attempt/'fixtures'),PYTHONDONTWRITEBYTECODE='1')
cmd=[sys.executable,'-I','-B',str(source/'test_traversal.py')]+sys.argv[2:]
start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);error=None;rc=None
try:
 with (attempt/'stdout.txt').open('xb') as out,(attempt/'stderr.txt').open('xb') as err:
  rc=subprocess.run(cmd,cwd='/private',env=env,stdout=out,stderr=err,timeout=90,check=False).returncode
except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)}
after=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt={'schema':'ace-artificial-traversal-test-attempt-v1','source_pins':pins,'command':cmd,'cwd':'/private','returncode':rc,'error':error,'wall_seconds':time.monotonic()-start,'child_user_seconds':after.ru_utime-before.ru_utime,'child_system_seconds':after.ru_stime-before.ru_stime,'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'maxrss_platform_units':after.ru_maxrss,'platform':sys.platform,'remote_actions':0,'scheduler_allocations':0,'scientific_worlds':0,'models_tokenizers_or_scores':0,'artificial_only':True,'runtime_qualified':False,'previous_attempts_preserved':True}
for n in ('stdout.txt','stderr.txt','measurements.json'):
 if (attempt/n).exists():
  b=(attempt/n).read_bytes();receipt[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
(attempt/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(json.dumps(receipt,sort_keys=True,indent=2))
sys.exit(0 if rc==0 and error is None else 1)

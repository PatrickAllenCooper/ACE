from pathlib import Path
import hashlib,json,os,resource,subprocess,sys,time
root=Path(__file__).resolve().parent;attempt=root/sys.argv[1];attempt.mkdir(exist_ok=False)
source=attempt/'source';source.mkdir();pins={}
for directory in ('dependencies','oracles','references'):
 (source/directory).mkdir()
 for p in sorted((root/directory).iterdir()):
  data=p.read_bytes();(source/directory/p.name).write_bytes(data);pins[directory+'/'+p.name]=hashlib.sha256(data).hexdigest()
for n in ('audit_saved_gpu72.py','test_gpu72_aid.py','oracle-pins.json'):
 data=(root/n).read_bytes();(source/n).write_bytes(data);pins[n]=hashlib.sha256(data).hexdigest()
cmd=[sys.executable,'-I','-B',str(source/'test_gpu72_aid.py')]+sys.argv[2:]
env=dict(os.environ,AUDIT_FIXTURES=str(attempt/'fixtures'),PYTHONDONTWRITEBYTECODE='1')
before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();rc=None;error=None
try:
 with (attempt/'stdout.txt').open('xb') as out,(attempt/'stderr.txt').open('xb') as err:rc=subprocess.run(cmd,cwd='/private',env=env,stdout=out,stderr=err,timeout=60).returncode
except BaseException as e:error=dict(type=type(e).__name__,message=str(e))
after=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt=dict(schema='ace-new-artificial-audit-test-attempt-v1',command=cmd,source_pins=pins,returncode=rc,error=error,
 child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,elapsed_seconds=time.monotonic()-start,maxrss_bytes_darwin=after.ru_maxrss,
 scientific_responses=0,models=0,tokenizers=0,remote_actions=0,actual_acceptance=False,runtime_qualified=False)
for n in ('stdout.txt','stderr.txt'):
 data=(attempt/n).read_bytes();receipt[n]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
(attempt/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='source_pins'},indent=2));print((attempt/'stderr.txt').read_text())
raise SystemExit(0 if rc==0 and error is None else 1)

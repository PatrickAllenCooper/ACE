from pathlib import Path
import subprocess,json,resource,time,fcntl
D=Path(__file__).resolve().parent
B=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
locks=[]
try:
 for n in ['ace-delivery-controller.lock','ace-foundation-controller.lock']:
  f=open(B/n,'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert not (D/'receipt.json').exists()
 cmd=json.loads((D/'command.json').read_text());before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
 p=subprocess.run(cmd,cwd='/tmp',capture_output=True);after=resource.getrusage(resource.RUSAGE_CHILDREN)
 for n,b in [('stdout.log',p.stdout),('stderr.log',p.stderr)]:
  with (D/n).open('xb') as out:out.write(b)
 r=dict(returncode=p.returncode,child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,elapsed_seconds=time.monotonic()-start,peak_child_rss_bytes=after.ru_maxrss,locks_acquired=2,scientific_execution=False,submitted=False)
 with (D/'receipt.json').open('x') as out:json.dump(r,out,indent=2);out.write('\n')
 print(json.dumps(r));print(p.stdout.decode());print(p.stderr.decode())
finally:
 for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()

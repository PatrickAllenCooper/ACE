from pathlib import Path
import fcntl,json,os,subprocess
C=Path(__file__).resolve().parent
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
locks=[]
try:
 for n in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
  p=BASE/n
  if p!=p.resolve():raise ValueError('lock alias')
  f=os.fdopen(os.open(p,os.O_RDWR|os.O_NOFOLLOW),'r+')
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BaseException:f.close();raise
  locks.append(f)
 try:
  p=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','python3 -'],input=(C/'scheduler07.remote.py').read_bytes(),capture_output=True,timeout=50)
  result=dict(returncode=p.returncode,timeout=False);out=p.stdout;err=p.stderr
 except subprocess.TimeoutExpired as e:
  result=dict(returncode=None,timeout=True);out=e.stdout or b'';err=e.stderr or b''
 for name,data in [('scheduler07.transport.json',(json.dumps(result,indent=2)+'\n').encode()),('scheduler07.remote.stdout',out),('scheduler07.remote.stderr',err)]:
  with (C/name).open('xb') as f:f.write(data)
 print(json.dumps(result))
finally:
 for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()

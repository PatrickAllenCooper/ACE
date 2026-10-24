import fcntl,json,os,sys,datetime
from pathlib import Path
d=Path(__file__).parent
fds=[]
for name in ['ace-delivery-controller.lock','ace-foundation-controller.lock']:
 f=open(d.parent/name,'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fds.append(f)
r={'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'locks':[str(Path(f.name)) for f in fds],'scope':'one approved replay10 preparation/submission','status':'held'}
(d/'controller.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if input().strip()!='RELEASE':raise ValueError('explicit release expected')
for f in fds:fcntl.flock(f,fcntl.LOCK_UN);f.close()
(d/'controller_released.json').write_text(json.dumps({'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'released'})+'\n')

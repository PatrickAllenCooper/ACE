from pathlib import Path
import os,json,datetime
p=Path('/scratch/alpine/paco0228/ACE/results/delivery_prospective_reporting_20261024_02')
p.mkdir(mode=0o700)
f=os.open(p,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);st=os.fstat(f)
assert st.st_uid==os.geteuid() and st.st_mode&0o077==0
claim={'root':str(p),'inode':st.st_ino,'uid':st.st_uid,'mode':oct(st.st_mode&0o777),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'report02 exclusive unsubmitted output; accepted replay12 unchanged'}
h=os.open('output_claim.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=f)
with os.fdopen(h,'w') as out:json.dump(claim,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
os.close(f);print(json.dumps(claim))

from pathlib import Path
import os,json,datetime
p=Path('/scratch/alpine/paco0228/ACE/results/delivery_prospective_reporting_20261024_01')
p.mkdir(mode=0o700)
info=p.stat()
r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(p),'inode':info.st_ino,'uid':info.st_uid,'mode':oct(info.st_mode&0o777),'purpose':'qualified replay12 prospective report-only output','new_fits':0,'new_responses':0}
with (p/'output_claim.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))

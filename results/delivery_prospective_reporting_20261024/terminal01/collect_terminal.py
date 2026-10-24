from pathlib import Path
import subprocess,json,hashlib,datetime,shutil
b=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline');d=b/'delivery-prospective-reporting-terminal-20261024-01';launch=b/'delivery-prospective-reporting-launch-20261024-01'
helper=['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run'];pin=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
 with (d/name).open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
r=subprocess.run(helper+['/scratch/alpine/paco0228/ACE/envs/neuronbench_py311/bin/python -I -B -'],input=(d/'capture_remote.py').read_text(),text=True,capture_output=True,timeout=90)
for n,t in [('remote_inventory.stdout',r.stdout),('remote_inventory.stderr',r.stderr)]:
 with (d/n).open('x') as f:f.write(t)
if r.returncode:raise RuntimeError(r.stderr)
a=json.loads(r.stdout);save('remote_inventory.json',a)
rows=[x.split('|') for x in a['commands']['sacct']['stdout'].splitlines()];parent=[x for x in rows if x[0]=='34864950' and x[1]=='34864950'];assert len(parent)==1;state=parent[0][2];assert state in ['COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL'],state
assert a['commands']['squeue']['returncode']==0 and not a['commands']['squeue']['stdout'].strip()
raw=d/'raw';raw.mkdir(mode=0o700)
r=subprocess.run(['scp','-r','-p','-o','ControlPath=/tmp/defab-curc.sock','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','paco0228@login.rc.colorado.edu:'+a['root']+'/.',str(raw)+'/'],text=True,capture_output=True,timeout=180)
save('copy_transport.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
if r.returncode:raise RuntimeError('copy failure preserved')
paths=list(raw.rglob('*'));assert not any(f.is_symlink() for f in paths)
actual={f.relative_to(raw).as_posix():{'bytes':f.stat().st_size,'sha256':pin(f)} for f in paths if f.is_file()};expected={f['path']:{'bytes':f['bytes'],'sha256':f['sha256']} for f in a['files']};assert actual==expected
pins=json.loads((launch/'input_pins.json').read_text());assert all(pin(raw/n)==h for n,h in pins.items())
save('custody_verification.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':34864950,'parent_state':state,'files_verified':len(actual),'bytes_verified':sum(v['bytes'] for v in actual.values()),'remote_inventory_sha256':pin(d/'remote_inventory.json'),'frozen_inputs_verified':len(pins),'scientific_outcomes_decoded_by_collection':False})
print(json.dumps({'state':state,'files':len(actual),'bytes':sum(v['bytes'] for v in actual.values()),'inventory_sha256':pin(d/'remote_inventory.json'),'sacct':a['commands']['sacct']['stdout']},indent=2))

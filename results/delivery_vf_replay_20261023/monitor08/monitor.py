from pathlib import Path
import subprocess,json,time,datetime,re
p=Path(__file__).parent
remote="""import json,pathlib,subprocess,datetime
p=pathlib.Path('/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261023_08')
c=subprocess.check_output(['scontrol','show','job','34824953'],text=True,timeout=20)
d={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scontrol':c,'files':{x.name:x.stat().st_size for x in p.iterdir() if x.is_file()}}
for n in ['replay_execution.json','launcher_failure.json']:
 if (p/n).exists():
  r=json.loads((p/n).read_bytes());d[n]={k:r.get(k) for k in ['status','stage','exit_code','failure_type','failure_reason','finished_at','new_fits','new_responses']}
print(json.dumps(d,indent=2))
"""
for i in range(2,34):
 time.sleep(55)
 r=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','/scratch/alpine/paco0228/ACE/envs/neuronbench_py311/bin/python -I -B -'],input=remote,text=True,capture_output=True,timeout=35)
 with (p/f'snapshot{i:02d}.stdout').open('x') as f:f.write(r.stdout)
 with (p/f'snapshot{i:02d}.stderr').open('x') as f:f.write(r.stderr)
 if r.returncode:print(json.dumps({'monitor_error':r.stderr,'snapshot':i}),flush=True);break
 d=json.loads(r.stdout);state=re.search(r'JobState=(\S+)',d['scontrol']).group(1)
 print(json.dumps({'snapshot':i,'at':d['at'],'state':state,'elapsed':re.search(r'RunTime=(\S+)',d['scontrol']).group(1),'milestones':{n:d['files'].get(n) for n in ['runtime_inventory.json','runtime_qualification.json','pre_checkpoint_runtime.json','replay_receipt.json','replay_execution.json']},'terminal':d.get('replay_execution.json')}),flush=True)
 if state not in ['RUNNING','PENDING','COMPLETING','CONFIGURING']:break

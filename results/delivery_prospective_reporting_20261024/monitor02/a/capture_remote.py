from pathlib import Path
import subprocess,json,datetime
root=Path('/scratch/alpine/paco0228/ACE/results/delivery_prospective_reporting_20261024_02')
r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':34867191,'partial_live_snapshot_not_terminal_custody':True}
r['commands']={}
for k,c in {'scontrol':['scontrol','show','job','34867191','-o'],'sacct':['sacct','-j','34867191','--noheader','--parsable2','--format=JobIDRaw,JobID,State,ExitCode,ElapsedRaw,CPUTimeRAW,TotalCPU,MaxRSS,AllocTRES,Account,Partition,QOS']}.items():
 v=subprocess.run(c,capture_output=True,text=True,timeout=20);r['commands'][k]={'returncode':v.returncode,'stdout':v.stdout,'stderr':v.stderr}
r['stages']={}
for n in ['report_started.json','report_runtime_started.json','report_inventory_verified.json','report_runtime_verified.json','exports_started.json','report_execution.json','launcher_failure.json']:
 p=root/n
 if p.exists() and p.stat().st_size<100000:r['stages'][n]=json.loads(p.read_text())
r['phases']={p.name:json.loads(p.read_text()) for p in sorted(root.glob('runtime_phase_*.json')) if p.stat().st_size<10000}
p=root/'runtime_stacks.log'
if p.exists():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-14000));r['stack_tail']=f.read().decode(errors='replace')
print(json.dumps(r,indent=2))

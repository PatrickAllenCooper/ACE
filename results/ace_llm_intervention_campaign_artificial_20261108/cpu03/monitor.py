from pathlib import Path
import json,subprocess,datetime
local=Path('/Users/pat/code/ACE/results/ace_llm_intervention_campaign_artificial_20261108/cpu03')
ext=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-cpu-20261108-03')
job=json.loads((local/'submission.result.json').read_bytes())['job_id'];assert job.isdigit()
root='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03'
source='''from pathlib import Path
import json,subprocess,datetime
root=Path(ROOT);job=JOB
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{}}
for label,args in {'queue':['squeue','-j',job,'-h','-o','%i|%j|%T|%R|%a'],'accounting':['sacct','-j',job,'-n','-P','--format=JobIDRaw,State,ExitCode,AllocCPUS,ElapsedRaw,AllocTRES%180,MaxRSS,TotalCPU']}.items():
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);v[label]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for name in ['cpu03/failure.json','cpu03/supervisor.json','cpu03/worker/failure.json','cpu03/worker/artifacts/failure.json','cpu03/worker/artifacts/cpu-phase/terminal.json']:
 p=root/name
 if p.is_file():v['files'][name]=p.read_text()[:10000]
p=root/'cpu03/worker/artifacts'
if p.is_dir():
 fs=sorted(p.glob('phase-*.json'))
 for f in fs[-4:]:v['files'][str(f.relative_to(root))]=f.read_text()[:2000]
print(json.dumps(v))
'''.replace('ROOT',repr(root)).replace('JOB',repr(job))
p=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','python3 -'],input=source,capture_output=True,text=True,timeout=45)
v={'source':source,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
i=1
while (local/f'live{i:02d}.json').exists():i+=1
for d in (local,ext):
 with (d/f'live{i:02d}.json').open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
print(p.stdout if p.returncode==0 else p.stderr)

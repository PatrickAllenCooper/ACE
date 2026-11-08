from pathlib import Path
import json,subprocess,datetime
local=Path('/Users/pat/code/ACE/results/ace_llm_intervention_campaign_gpu_20261108/launch01');ext=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-gpu-20261108-01')
job=json.loads((local/'submission.result.json').read_bytes())['job_id'];assert job.isdigit()
root='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03/gpu03'
source='''from pathlib import Path
import json,subprocess,datetime
root=Path(ROOT);job=JOB
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{}}
for label,args in {'queue':['squeue','-j',job,'-h','-o','%i|%j|%T|%R|%a'],'accounting':['sacct','-j',job,'-n','-P','--format=JobIDRaw,State,ExitCode,AllocCPUS,ElapsedRaw,AllocTRES%180,MaxRSS,TotalCPU']}.items():
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);v[label]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
if root.is_dir():
 for p in sorted(root.rglob('*.json')):
  if p.name in ('terminal.json','failure.json','startup.json') and p.stat().st_size<50000:
   v['files'][str(p.relative_to(root))]=p.read_text()[:10000]
 # Bounded current progress only, never complete terminal custody.
 for p in sorted(root.glob('services/*/worker/artifacts/phase-*.json'))[-6:]:v['files'][str(p.relative_to(root))]=p.read_text()[:2000]
 for directory in {p.parent for p in root.glob('controller/**/telemetry-*.json')}:
  for p in sorted(directory.glob('telemetry-*.json'))[-2:]:v['files'][str(p.relative_to(root))]=p.read_text()[:6000]
print(json.dumps(v))
'''.replace('ROOT',repr(root)).replace('JOB',repr(job))
p=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','python3 -'],input=source,capture_output=True,text=True,timeout=45)
v={'source':source,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
i=1
while (local/f'live{i:02d}.json').exists():i+=1
for d in (local,ext):
 with (d/f'live{i:02d}.json').open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
print(p.stdout if p.returncode==0 else p.stderr)

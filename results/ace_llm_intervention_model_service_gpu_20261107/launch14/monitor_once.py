from pathlib import Path
import json,subprocess,sys,datetime
local=Path(__file__).parent
external=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-model-service-gpu-20261107-14')
name=sys.argv[1]
job=json.loads((local/'submission.result.json').read_bytes())['job_id']
if not job.isdigit():raise ValueError('invalid saved job')
if not name.startswith('live') or '/' in name:raise ValueError('bounded receipt name')
source="""from pathlib import Path
import subprocess,json,datetime
root=Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261107_14')
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name,cmd in {'job':['scontrol','show','job','-o',JOB],'accounting':['sacct','-j',JOB,'-n','-P','--format=JobIDRaw,JobID,JobName%60,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%150,MaxRSS,TotalCPU']}.items():
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=15);r[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
r['packed_files']={}
r['gpu_samples']={str(p.relative_to(root)):json.loads(p.read_bytes()) for p in sorted((root/'qualification').glob('telemetry-*.json'))[-3:]}
r['generation_records']={str(p.relative_to(root)):{k:v for k,v in json.loads(p.read_bytes()).items() if k in ('ordinal','cuda_interval_ms','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','elapsed_seconds')} for p in (root/'qualification').rglob('generation-*.json')}
r['phases']={str(p.relative_to(root)):json.loads(p.read_bytes()) for p in (root/'qualification').rglob('phase-*.json')}
r['ends']={str(p.relative_to(root)):json.loads(p.read_bytes()) for p in (root/'qualification').rglob('*.json') if p.name in ('terminal.json','failure.json','failure-detail.json','diagnostic_complete.json')}
print(json.dumps(r))
"""
source=source.replace('JOB',repr(job))
p=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','python3 -'],input=source,text=True,capture_output=True,timeout=45)
v={'local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':source,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'read_only':True}
raw=(json.dumps(v,indent=2)+'\n').encode()
for root in [local,external]:
 with (root/(name+'.json')).open('xb') as f:f.write(raw)
if p.returncode:print(p.stderr)
else:
 r=json.loads(p.stdout)
 print(json.dumps({'utc':r['utc'],'accounting':r['accounting'],'job':r['job']['stdout'],'phase_count':len(r['phases']),'last_phases':dict(sorted(r['phases'].items())[-4:]),'gpu_samples':r['gpu_samples'],'generation_records':r['generation_records'],'ends':r['ends']},indent=2))

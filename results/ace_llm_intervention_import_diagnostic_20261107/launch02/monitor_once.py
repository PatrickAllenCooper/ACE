from pathlib import Path
import json,subprocess,sys,datetime
local=Path(__file__).parent
external=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-import-diagnostic-20261107-02')
name=sys.argv[1]
if not name.startswith('live') or '/' in name:raise ValueError('bounded receipt name')
source="""from pathlib import Path
import subprocess,json,datetime
root=Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_import_diagnostic_20261107_02')
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name,cmd in {'job':['scontrol','show','job','-o','35343088'],'accounting':['sacct','-j','35343088','-n','-P','--format=JobIDRaw,JobID,JobName%60,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%150,MaxRSS,TotalCPU']}.items():
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=15);r[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
r['diagnostic_files']={str(p.relative_to(root)):p.stat().st_size for p in (root/'diagnostic').rglob('*') if p.is_file()}
r['phases']={str(p.relative_to(root)):json.loads(p.read_bytes()) for p in (root/'diagnostic').glob('*/phase-*.json')}
r['ends']={str(p.relative_to(root)):json.loads(p.read_bytes()) for p in (root/'diagnostic').rglob('*.json') if p.name in ('terminal.json','failure.json','diagnostic_complete.json')}
print(json.dumps(r))
"""
p=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','python3 -'],input=source,text=True,capture_output=True,timeout=45)
v={'local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':source,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'read_only':True}
raw=(json.dumps(v,indent=2)+'\n').encode()
for root in [local,external]:
 with (root/(name+'.json')).open('xb') as f:f.write(raw)
if p.returncode:print(p.stderr)
else:
 r=json.loads(p.stdout)
 print(json.dumps({'utc':r['utc'],'accounting':r['accounting'],'job':r['job']['stdout'],'phase_count':len(r['phases']),'last_phases':{k:v for k,v in r['phases'].items() if 'complete' in v['phase'] or 'signal' in v['phase']},'ends':r['ends']},indent=2))

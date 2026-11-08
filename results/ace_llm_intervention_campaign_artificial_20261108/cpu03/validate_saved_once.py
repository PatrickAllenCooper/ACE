from pathlib import Path
import json,subprocess,datetime,shlex
r=Path('/Users/pat/code/ACE/results/ace_llm_intervention_campaign_artificial_20261108/cpu03');e=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-cpu-20261108-03')
p=json.loads((r/'inputs/packet.json').read_bytes());meta=json.loads((r/'packet-metadata.json').read_bytes());job=json.loads((r/'submission.result.json').read_bytes())['job_id'];assert job.isdigit()
# Use exactly the source-bound bootstrap already frozen in the service argv.
bootstrap=p['persistent_specs']['frozen_llm']['command'][4]
entry=meta['input_directory']+'/prepare_llm_intervention_campaign_cpu.py'
command=[p['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,entry,p['files']['prepare_llm_intervention_campaign_cpu.py'],'--packet',meta['input_directory']+'/packet.json','--packet-sha256',meta['cpu_packet_sha256'],'--validate-saved']
source='''from pathlib import Path
import json,subprocess,hashlib,datetime
job=JOB;packet=Path(PACKET);command=COMMAND
q=subprocess.run(['sacct','-X','-j',job,'-n','-P','--format=JobIDRaw,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180'],capture_output=True,text=True,timeout=15)
rows=[l.split('|') for l in q.stdout.splitlines() if l.startswith(job+'|')]
if q.returncode or len(rows)!=1 or rows[0][1:3]!=['COMPLETED','0:0']:raise ValueError('successful terminal allocation required')
if hashlib.sha256(packet.read_bytes()).hexdigest()!=PIN:raise ValueError('packet drift')
v=json.loads(packet.read_bytes());d=Path(v['cpu_output'])/'saved-validation'
if d.exists():raise ValueError('saved-validation already claimed, never blind retry')
p=subprocess.run(command,capture_output=True,text=True,timeout=50)
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'accounting':rows[0],'no_numerical_execution':True}
f=d/'validation.json'
if f.exists():result['validation_sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
print(json.dumps(result))
'''.replace('JOB',repr(job)).replace('PACKET',repr(meta['input_directory']+'/packet.json')).replace('COMMAND',repr(command)).replace('PIN',repr(meta['cpu_packet_sha256']))
for d in (r,e):
 with (d/'saved-validation.intent.json').open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':source},f,indent=2)
p=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','python3 -'],input=source,capture_output=True,text=True,timeout=65)
v={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for d in (r,e):
 with (d/'saved-validation.result.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v))

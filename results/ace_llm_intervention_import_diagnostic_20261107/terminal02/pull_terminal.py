from pathlib import Path
import subprocess,json,datetime,hashlib,shutil
repo=Path('/Users/pat/code/ACE/results/ace_llm_intervention_import_diagnostic_20261107/terminal02')
ext=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-import-diagnostic-terminal-20261107-02')
ext.mkdir(exist_ok=False);repo.mkdir(exist_ok=False)
remote='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_import_diagnostic_20261107_02'
src="""from pathlib import Path
import hashlib,json,subprocess,datetime
r=Path('"""+remote+"""')
files={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in r.rglob('*') if p.is_file()}
results={}
for k,c in {'accounting':['sacct','-j','35343088','-n','-P','--format=JobIDRaw,JobID,JobName%60,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%150,MaxRSS,TotalCPU'],'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],'job':['scontrol','show','job','-o','35343088']}.items():
 p=subprocess.run(c,capture_output=True,text=True,timeout=15);results[k]={'command':c,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps({'files':files,'results':results,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
"""
p=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','python3 -'],input=src,capture_output=True,text=True,timeout=50)
(ext/'inventory_transport.json').write_text(json.dumps({'command_python':src,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2)+'\n')
assert p.returncode==0,p.stderr
v=json.loads(p.stdout);assert '35343088|35343088|ace_llm_import02|COMPLETED|0:0|1|301|' in v['results']['accounting']['stdout']
p=subprocess.run(['scp','-r','-o','ControlPath=/tmp/defab-curc.sock','-o','BatchMode=yes','paco0228@login.rc.colorado.edu:'+remote,str(ext/'remote')],capture_output=True,text=True,timeout=60)
(ext/'transfer.json').write_text(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2)+'\n');assert p.returncode==0,p.stderr
files={str(p.relative_to(ext/'remote')):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (ext/'remote').rglob('*') if p.is_file()}
assert files==v['files']
assert files['probe.py']['sha256']=='8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951'
assert files['job.sbatch']['sha256']=='9d9f230db70214ef85283e634dbc5e99b4ac84a3a58bc866a984370c08b98ca3'
assert files['packet.json']['sha256']=='546ffe5d314e790339b38bc86521b92992d56e481057a70a00a07780210a9057'
rc={'status':'diagnostic_evidence_retained_independent_review_pending','job_id':'35343088','account':'ucb736_asc2','allocated_cpu_seconds':301,'gpu_slice_seconds':0,'known_closed_allocated_cpu_seconds':4763,'cycle_cpu_ceiling':57600,'cycle_gpu_ceiling':7200,'cumulative_prep_allowance':3600,'cpu_admission_after_release':8363,'gpu_admission_after_release':836,'process_cpu_seconds_sacct':14.374,'batch_max_rss_kib':1073460,'costs_overlap_not_additive':True,'scientific_responses':0,'model_weights_loaded':False,'scientific_runtime_qualified':False,'original_import_order':'timeout, retained failure and outer SIGKILL cleanup','scipy_first':'successful imports, no model loading','comparison_confound':'Sequential shared filesystem cache and changed import order. Does not identify a unique cause or prove an order-only fix.','raw_files':len(files),'raw_bytes':sum(x['bytes'] for x in files.values()),'inventory':files,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(ext/'disposition.json').write_text(json.dumps(rc,indent=2,sort_keys=True)+'\n')
shutil.copyfile(__file__,ext/'pull_terminal.py')
for p in ext.iterdir():
 if p.is_dir():shutil.copytree(p,repo/p.name)
 else:shutil.copyfile(p,repo/p.name)
print(json.dumps({k:v for k,v in rc.items() if k!='inventory'},indent=2))

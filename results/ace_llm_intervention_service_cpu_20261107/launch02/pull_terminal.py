"""Read-only terminal collection for the CPU service preflight."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time
def observed_allocation(row):
 errors=[]
 def integer(value,name):
  if type(value) is str and value.isdigit():return int(value)
  errors.append('unparseable '+name);return None
 cpus=integer(row[5],'AllocCPUS');elapsed=integer(row[6],'ElapsedRaw')
 tres={}
 for item in row[8].split(','):
  if not item:continue
  k,sep,value=item.partition('=')
  if not sep or k in tres:errors.append('invalid/duplicate AllocTRES field')
  else:tres[k]=value
 typed=[integer(v,k) for k,v in tres.items() if k.startswith('gres/gpu:')]
 generic=integer(tres['gres/gpu'],'gres/gpu') if 'gres/gpu' in tres else None
 if generic is not None:
  gpu=generic
  if typed and (None in typed or sum(typed)!=generic):errors.append('conflicting GPU counts');gpu=None
 elif typed:gpu=None if None in typed else sum(typed)
 elif 'gres/gpu' in tres or any(k.startswith('gres/') for k in tres):gpu=None;errors.append('unknown GPU allocation')
 elif tres or cpus==0 and elapsed==0:gpu=0
 else:gpu=None;errors.append('missing allocation inventory')
 if errors:gpu=None
 cpu_seconds=None if cpus is None or elapsed is None else cpus*elapsed
 gpu_seconds=None if gpu is None or elapsed is None else gpu*elapsed
 matched=(not errors and row[7]=='ucb736_asc2' and cpus==1 and tres.get('cpu')=='1'
          and tres.get('mem')=='3840M' and tres.get('node')=='1' and gpu==0
          and not any(k.startswith('gres/') for k in tres))
 return {'observed_account':row[7],'observed_cpus':cpus,'observed_elapsed_seconds':elapsed,
   'observed_alloc_tres':row[8],'observed_gpu_count':gpu,'resource_contract_matched':matched,
   'allocated_cpu_seconds':cpu_seconds,'gpu_slice_seconds':gpu_seconds,'accounting_errors':errors}
repo=Path('/Users/pat/code/ACE')
launch=Path(__file__).parent
job=json.loads((launch/'submission.result.json').read_bytes())['job_id']
assert job.isdigit() and len(job)<=12
root='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_service_cpu_20261107_02'
ext=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-service-cpu-terminal-20261107-02')
local=repo/'results/ace_llm_intervention_service_cpu_20261107/terminal02'
assert not ext.exists() and not local.exists()
source='''from pathlib import Path
import subprocess,hashlib,json,datetime,stat
root=Path(ROOT)
results={}
for k,c in {'accounting':['sacct','-j',JOB,'-n','-P','--format=JobIDRaw,JobID,JobName%60,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180,MaxRSS,TotalCPU'],'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],'job':['scontrol','show','job','-o',JOB]}.items():
 p=subprocess.run(c,capture_output=True,text=True,timeout=20);results[k]={'command':c,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
rows=[line.split('|') for line in results['accounting']['stdout'].splitlines() if line.startswith(JOB+'|')]
if len(rows)!=1 or rows[0][3] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'):
 raise RuntimeError('job not conclusively terminal, never restart on observation failure')
files={}
for p in sorted(root.rglob('*')):
 if p.is_symlink():raise ValueError('unexpected output symlink')
 if not p.is_file():continue
 name=str(p.relative_to(root));n=p.stat().st_size
 if n>64*1024*1024:raise ValueError('unexpected bulk artifact in CPU service preflight')
 files[name]={'bytes':n,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pin_origin':'fresh remote byte hash'}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'results':results,'parent_accounting':rows[0]}))
'''.replace('ROOT',repr(root)).replace('JOB',repr(job))
p=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','python3 -'],input=source,capture_output=True,text=True,timeout=60)
if p.returncode:raise RuntimeError(p.stderr)
v=json.loads(p.stdout)
needed=sum(x['bytes'] for x in v['files'].values())
fs=os.statvfs(ext.parent)
if fs.f_bavail*fs.f_frsize<needed+2*1024**3:raise ValueError('insufficient exclusive local custody space')
ext.mkdir();local.mkdir()
transport={'source':source,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(ext/'inventory_transport.json').write_text(json.dumps(transport,indent=2)+'\n')
t=time.monotonic();cpu=time.process_time()
p=subprocess.run(['scp','-r','-o','ControlPath=/tmp/defab-curc.sock','-o','BatchMode=yes','paco0228@login.rc.colorado.edu:'+root,str(ext/'remote')],capture_output=True,text=True,timeout=1800)
(ext/'transfer.json').write_text(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'elapsed_seconds':time.monotonic()-t},indent=2)+'\n')
if p.returncode:raise RuntimeError('transfer failed, retain partial custody and reconcile')
actual={}
for p in sorted((ext/'remote').rglob('*')):
 if not p.is_file():continue
 name=str(p.relative_to(ext/'remote'));h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 actual[name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
assert set(actual)==set(v['files'])
for n,a in actual.items():
 expect=v['files'][n]
 assert a['bytes']==expect['bytes'],n
 if expect['sha256'] is not None:assert a['sha256']==expect['sha256'],n
packet=json.loads((launch/'packet.json').read_bytes())
expected_inputs={**packet['new_input_files'],'packet.json':hashlib.sha256((launch/'packet.json').read_bytes()).hexdigest()}
input_mismatches=[n for n,h in expected_inputs.items() if actual.get(n,{}).get('sha256')!=h]
row=v['parent_accounting'];allocation=observed_allocation(row);seconds=allocation['allocated_cpu_seconds'];gpu_seconds=allocation['gpu_slice_seconds']
terminal=ext/'remote/qualification/terminal.json'
failure=ext/'remote/qualification/failure.json'
disposition={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'complete_custody_independent_review_pending','job_id':job,'scheduler_state':row[3],'scheduler_exit':row[4],**allocation,'resource_qualification':'pending_independent_review' if allocation['resource_contract_matched'] else 'failed_resource_contract','known_closed_cpu_seconds':None if seconds is None else 6384+seconds,'cpu_admission_after_release':None if seconds is None else 6384+seconds+3600,'cpu_ceiling_seconds':57600,'gpu_admission_seconds':None if gpu_seconds is None else 836+gpu_seconds,'gpu_ceiling_seconds':7200,'model_service_qualified':False,'scientific_result':False,'raw_files':len(actual),'raw_bytes':sum(x['bytes'] for x in actual.values()),'files':actual,'planned_runtime_archive_reuse':True,'observed_staging_scope':'This collector does not infer completed staging. Inspect the retained stage receipt and independent runtime review.','runtime_contract_adopted':False,'terminal':json.loads(terminal.read_bytes()) if terminal.exists() else None,'failure':json.loads(failure.read_bytes()) if failure.exists() else None,'local_collection_process_cpu_seconds':time.process_time()-cpu,'local_collection_elapsed_seconds':time.monotonic()-t,'cost_scope':'collector snapshots exclude earlier prep and remote work, not whole cycle'}
disposition.update(input_pin_mismatches=input_mismatches,input_qualification='failed_input_contract' if input_mismatches else 'pending_independent_review')
(ext/'disposition.json').write_text(json.dumps(disposition,sort_keys=True,indent=2)+'\n')
shutil.copyfile(__file__,ext/'pull_terminal.py')
for p in ext.rglob('*'):
 rel=p.relative_to(ext)
 if p.is_dir():(local/rel).mkdir(exist_ok=True)
 else:shutil.copyfile(p,local/rel)
print(json.dumps({k:x for k,x in disposition.items() if k not in ('files',)},indent=2))

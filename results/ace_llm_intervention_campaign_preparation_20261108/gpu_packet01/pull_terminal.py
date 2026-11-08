"""Complete read-only terminal custody. Does not confer GPU acceptance."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,time
repo=Path('/Users/pat/code/ACE');base=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
launch=repo/'results/ace_llm_intervention_campaign_gpu_20261108/launch01';meta=json.loads((launch/'packet-metadata.json').read_bytes())
job=json.loads((launch/'submission.result.json').read_bytes())['job_id'];assert job.isdigit()
roots={'launch':meta['remote_root'],'common':'/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03'}
ext=base/'ACE-LLM-intervention-campaign-gpu-terminal-20261108-01';local=repo/'results/ace_llm_intervention_campaign_gpu_20261108/terminal01'
assert not ext.exists() and not local.exists()
source='''from pathlib import Path
import subprocess,hashlib,json,datetime
roots=ROOTS;job=JOB
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':{},'files':{}}
for k,c in {'accounting':['sacct','-j',job,'-n','-P','--format=JobIDRaw,JobID,JobName%60,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180,MaxRSS,TotalCPU'],'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],'job':['scontrol','show','job','-o',job]}.items():
 p=subprocess.run(c,capture_output=True,text=True,timeout=20);v['results'][k]={'command':c,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
rows=[l.split('|') for l in v['results']['accounting']['stdout'].splitlines() if l.startswith(job+'|')]
if len(rows)!=1 or rows[0][3] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'):raise ValueError('job not conclusively terminal')
v['parent_accounting']=rows[0]
for label,path in roots.items():
 root=Path(path)
 for p in sorted(root.rglob('*')):
  if p.is_symlink():raise ValueError('unexpected raw symlink')
  if not p.is_file():continue
  n=p.stat().st_size
  if n>128*1024**2:raise ValueError('unexpected bulk output')
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b)
  v['files'][label+'/'+p.relative_to(root).as_posix()]={'bytes':n,'sha256':h.hexdigest()}
print(json.dumps(v))
'''.replace('ROOTS',repr(roots)).replace('JOB',repr(job))
p=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','python3 -'],input=source,capture_output=True,text=True,timeout=90)
if p.returncode:raise RuntimeError(p.stderr)
v=json.loads(p.stdout);size=sum(x['bytes'] for x in v['files'].values());fs=os.statvfs(base)
if fs.f_bavail*fs.f_frsize<size*2+2*1024**3:raise ValueError('insufficient two-copy custody space')
ext.mkdir();local.mkdir();(ext/'remote').mkdir()
(ext/'inventory_transport.json').write_text(json.dumps({'source':source,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2)+'\n')
started=time.monotonic();cpu=time.process_time();transfers={}
for label,root in roots.items():
 p=subprocess.run(['scp','-r','-o','ControlPath=/tmp/defab-curc.sock','-o','BatchMode=yes','paco0228@login.rc.colorado.edu:'+root,str(ext/'remote'/label)],capture_output=True,text=True,timeout=1800)
 transfers[label]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 with (ext/('transfer-'+label+'.json')).open('x') as f:json.dump(transfers[label],f,indent=2)
 if p.returncode:raise RuntimeError('partial custody retained; reconcile before any new action')
actual={}
for p in sorted((ext/'remote').rglob('*')):
 if p.is_symlink():raise ValueError('local symlink')
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b)
  actual[p.relative_to(ext/'remote').as_posix()]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
if actual!=v['files']:raise ValueError('complete custody membership/bytes mismatch; retain')
row=v['parent_accounting'];cpus=int(row[5]);elapsed=int(row[6]);tres=dict(x.split('=',1) for x in row[8].split(',') if x)
generic=tres.get('gres/gpu');typed=[int(v) for k,v in tres.items() if k.startswith('gres/gpu:')]
gpu=int(generic) if generic is not None else sum(typed)
accounting_errors=[]
if not tres and not (cpus==0 and elapsed==0):accounting_errors.append('missing allocation inventory')
if any(k.startswith('gres/') and not (k=='gres/gpu' or k.startswith('gres/gpu:')) for k in tres):accounting_errors.append('unknown generic resource')
if accounting_errors:gpu=None
if not accounting_errors and generic is not None and typed and gpu!=sum(typed):raise ValueError('GPU count conflict; preserve raw accounting')
expected={'launch/inputs/'+n:h for n,h in meta['input_files'].items()}
expected.update({'launch/'+n:h for n,h in meta['root_files'].items()})
mismatches=[n for n,h in expected.items() if actual.get(n,{}).get('sha256')!=h]
matched=(row[7]=='ucb736_asc2' and 4<=cpus<=8 and tres.get('cpu')==str(cpus) and tres.get('mem') in ('15360M','15G','15.00G') and tres.get('node')=='1' and gpu==1 and tres.get('gres/gpu:a100_3g.20gb')=='1')
values={}
for name in ['terminal.json','failure.json','controller/phase/terminal.json']:
 p=ext/'remote/common/gpu03'/name
 if p.exists():values[name]=json.loads(p.read_bytes())
disposition={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'complete_custody_independent_review_pending','job_id':job,'scheduler_state':row[3],'scheduler_exit':row[4],'account':row[7],'allocated_cpus':cpus,'elapsed_seconds':elapsed,'gpu_count':gpu,'alloc_tres':row[8],'resource_contract_matched':matched,'allocated_cpu_seconds':cpus*elapsed,'gpu_slice_seconds':None if gpu is None else gpu*elapsed,'accounting_errors':accounting_errors,'closed_cpu_seconds':9702+cpus*elapsed,'closed_gpu_slice_seconds':None if gpu is None else 1347+gpu*elapsed,'cumulative_preparation_cpu_allowance':3600,'cpu_admission_after_release':9702+cpus*elapsed+3600,'cpu_ceiling_seconds':57600,'gpu_ceiling_seconds':7200,'raw_files':len(actual),'raw_bytes':size,'files':actual,'input_pin_mismatches':mismatches,'terminal_records':values,'gpu_campaign_qualified':False,'scientific_qualified':False,'collector_elapsed_seconds':time.monotonic()-started,'collector_process_cpu_seconds':time.process_time()-cpu,'cost_scope':'local collector measurement excludes transfer subprocess CPU and earlier preparation. Not whole-cycle CPU.'}
(ext/'disposition.json').write_text(json.dumps(disposition,indent=2,sort_keys=True)+'\n');shutil.copyfile(__file__,ext/'pull_terminal.py')
for p in ext.rglob('*'):
 rel=p.relative_to(ext)
 if p.is_dir():(local/rel).mkdir(exist_ok=True)
 else:shutil.copyfile(p,local/rel)
print(json.dumps({k:x for k,x in disposition.items() if k not in ('files','terminal_records')},indent=2))

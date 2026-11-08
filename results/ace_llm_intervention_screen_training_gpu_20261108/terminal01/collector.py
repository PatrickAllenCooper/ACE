#!/usr/bin/env python3
"""Once-only read-only full custody; no submission, cancellation or qualification.

Supply independent --job-id, --packet-sha256, --submission-intent-sha256
(local intent), --submission-result-sha256 and --launch-commit (packet commit).
Both terminal01 destinations must be absent. Any failure retains claimed paths;
never rerun blindly. Only this job's root is transferred. Prior jobs/journals
are freshly read for exact 37->38 / 76->78 reconciliation, never modified.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, re, shlex, shutil, stat, subprocess, time

REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LAUNCH=REPO/'results/ace_llm_intervention_screen_training_gpu_20261108/launch01'
LAUNCH_EXT=BASE/'ACE-LLM-intervention-screen-training-gpu-20261108-01'
LOCAL=LAUNCH.parent/'terminal01'
EXT=BASE/'ACE-LLM-intervention-screen-training-gpu-terminal-20261108-01'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_training_gpu_20261108_01'
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
REVIEW='docs/development/guidance/reviews/ace_llm_intervention_screen_training_gpu_packet01_review_2026-11-08.md'
MAX_BYTES,MAX_FILES,MAX_FILE=512*1024**2,4096,64*1024**2
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
RESOURCES={'account':'ucb736_asc2','partition':'aa100','qos':'gpu-testing','requested_cpus':4,
 'max_allocated_cpus':8,'memory_mib':12288,'wall_seconds':600,'gpus':1,'gpu_type':'a100_3g.20gb'}

def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def encoded(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def text(v):return v.decode(errors='replace') if isinstance(v,bytes) else v
def read(p,cap=MAX_FILE):
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode) and a.st_size<=cap,'bounded regular file required: '+str(p))
  raw=f.read(cap+1);b=os.fstat(f.fileno())
 need(len(raw)<=cap and (a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns),'file changed: '+str(p))
 return raw
def inventory(root):
 need(root.is_dir() and not root.is_symlink(),'ordinary custody root required')
 result={};size=0
 for p in sorted(root.rglob('*')):
  need(not p.is_symlink(),'symlink in custody: '+str(p))
  if p.is_dir():continue
  raw=read(p);size+=len(raw)
  need(size<=MAX_BYTES and len(result)<MAX_FILES,'complete custody exceeds bound; retain partials')
  result[p.relative_to(root).as_posix()]={'bytes':len(raw),'sha256':sha(raw)}
 return result
def pairs(raw):
 entries=[s.split('=',1) for s in raw.split(',') if s]
 need(all(len(s)==2 and all(s) for s in entries) and len(dict(entries))==len(entries),'invalid/duplicate TRES')
 return dict(entries)
def gpu_count(t,cpus,elapsed):
 if not t:return 0 if cpus==elapsed==0 else None
 g={k:v for k,v in t.items() if k.startswith('gres/')}
 if any(not v.isdigit() or not (k=='gres/gpu' or k.startswith('gres/gpu:')) for k,v in g.items()):return None
 typed=[int(v) for k,v in g.items() if k.startswith('gres/gpu:')]
 aggregate=int(g['gres/gpu']) if 'gres/gpu' in g else None
 if aggregate is not None and typed and aggregate!=sum(typed):return None
 return aggregate if aggregate is not None else sum(typed)
def cost(rows):
 cpus=gpus=0
 for r in rows:
  need(len(r)==8 and r[4].isdigit() and r[5].isdigit(),'unparseable census accounting')
  c,e=int(r[4]),int(r[5]);n=gpu_count(pairs(r[7]),c,e)
  need(n is not None,'unknown GPU accounting; never infer zero')
  cpus+=c*e;gpus+=n*e
 return cpus,gpus

def requested(t):
 return (t.get('cpu')=='4' and t.get('node')=='1' and t.get('mem') in ('12G','12.00G','12288M')
  and gpu_count(t,4,1)==1
  and all(k=='gres/gpu' or k=='gres/gpu:a100_3g.20gb' for k in t if k.startswith('gres/')))

REMOTE=r'''
from pathlib import Path
import datetime,hashlib,json,os,stat,subprocess,time
root=Path(ROOT);deadline=time.monotonic()+150
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':{},'files':{},'submission_journals':{}}
def read(path,cap):
 if time.monotonic()>deadline:raise TimeoutError('inventory deadline')
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  a=os.fstat(f.fileno())
  if not stat.S_ISREG(a.st_mode) or a.st_size>cap:raise ValueError('bounded regular artifact required: '+str(path))
  raw=f.read(cap+1);b=os.fstat(f.fileno())
 if len(raw)>cap or (a.st_ino,a.st_size,a.st_mtime_ns)!=(b.st_ino,b.st_size,b.st_mtime_ns):raise ValueError('changing artifact')
 return raw
commands={
 'accounting':['sacct','-j',JOB,'-n','-P','--format=JobIDRaw,JobID,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180,MaxRSS,TotalCPU'],
 'details':['sacct','-X','-j',JOB,'-n','-P','--format=JobIDRaw,Partition,QOS,TimelimitRaw,ReqCPUS,ReqTRES%180,User'],
 'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],
 'census':['sacct','-X','-u','paco0228','-S','2026-11-02','-n','-P','--format=JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180'],
 'job':['scontrol','show','job','-o',JOB]}
try:
 for name,cmd in commands.items():
  try:
   p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
   v['results'][name]={'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
  except subprocess.TimeoutExpired as e:
   decode=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x
   v['results'][name]={'command':cmd,'returncode':None,'stdout':decode(e.stdout),'stderr':decode(e.stderr),'timed_out':True}
 if not root.is_dir() or root.is_symlink():raise ValueError('ordinary job root required')
 size=0
 for p in sorted(root.rglob('*')):
  if p.is_symlink():raise ValueError('output symlink: '+str(p))
  if p.is_dir():continue
  raw=read(p,64*1024**2);size+=len(raw)
  if size>512*1024**2 or len(v['files'])>=4096:raise ValueError('complete raw custody exceeds bound')
  v['files'][p.relative_to(root).as_posix()]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 for folder in sorted(root.parent.glob('ace_llm_intervention*')):
  if folder.is_symlink():raise ValueError('journal root symlink')
  for p in sorted(folder.glob('submission.*.json')):
   if len(v['submission_journals'])>=128:raise ValueError('journal bound exceeded')
   v['submission_journals'][str(p)]=hashlib.sha256(read(p,4*1024**2)).hexdigest()
except BaseException as e:v['inventory_error']={'type':type(e).__name__,'message':str(e)}
print(json.dumps(v))
'''

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ('job-id','packet-sha256','submission-intent-sha256','submission-result-sha256','launch-commit'):ap.add_argument('--'+name,required=True)
 args=ap.parse_args();need(re.fullmatch('[1-9][0-9]{0,11}',args.job_id),'explicit job ID required')
 for pin in (args.packet_sha256,args.submission_intent_sha256,args.submission_result_sha256):need(re.fullmatch('[0-9a-f]{64}',pin),'independent pin required')
 need(re.fullmatch('[0-9a-f]{40}',args.launch_commit),'full launch commit required')
 claimed=[];started=time.monotonic();cpu_start=time.process_time();deadline=started+900
 def save(name,value):
  raw=value if isinstance(value,bytes) else encoded(value)
  for dest in claimed:
   with (dest/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 def run(name,cmd,**kwargs):
  try:
   p=subprocess.run(cmd,capture_output=True,text=True,timeout=min(kwargs.pop('timeout',180),max(.1,deadline-time.monotonic())),**kwargs)
   record={'utc':utc(),'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
  except subprocess.TimeoutExpired as e:
   save(name,{'utc':utc(),'command':cmd,'returncode':None,'stdout':text(e.stdout),'stderr':text(e.stderr),'timed_out':True});raise
  save(name,record);need(p.returncode==0,'transport failed; evidence retained, no blind retry: '+name)
  return p.stdout
 def mirror_partial():
  if len(claimed)!=2 or not (EXT/'remote').exists():return
  dst=LOCAL/'remote'
  if not dst.exists():shutil.copytree(EXT/'remote',dst,symlinks=True)
 try:
  # Claim both roots before observations, so first-observation failures survive.
  for dest in (EXT,LOCAL):need(not os.path.lexists(dest),'terminal01 already claimed; reconcile, never blind retry')
  for dest in (EXT,LOCAL):dest.mkdir(mode=0o700);claimed.append(dest)
  save('collector.py',read(Path(__file__)))
  save('collector-start.json',{'utc':utc(),'arguments':vars(args),'remote_root':ROOT,'retry_allowed':False,'job_actions':False})
  before_launch=inventory(LAUNCH);need(inventory(LAUNCH_EXT)==before_launch,'local/external launch custody differs')
  for dest in claimed:
   shutil.copytree(LAUNCH,dest/'launch')
   need(inventory(dest/'launch')==before_launch,'launch custody copy differs')
  def pinned(name,pin):
   raw=read(LAUNCH/name);need(sha(raw)==pin,'independent launch pin differs: '+name);return json.loads(raw)
  packet=pinned('packet.json',args.packet_sha256)
  intent=pinned('submission.intent.json',args.submission_intent_sha256)
  result=pinned('submission.result.json',args.submission_result_sha256)
  need(result.get('returncode')==0 and result.get('job_id')==args.job_id and not result.get('ambiguous'),'submission unresolved or wrong job')
  need(re.fullmatch('[0-9a-f]{64}',result.get('intent_sha256','')),'remote intent pin absent')
  need(packet['remote_root']==intent['remote_root']==ROOT and packet['schema']=='ace_screen_training_gpu_launch_v1','wrong launch root/schema')
  need(intent['source_commit']==args.launch_commit and intent['packet_sha256']==args.packet_sha256,'launch commit/packet binding differs')
  need(packet['resources']==intent['resources']==RESOURCES,'resource contract changed')
  admission=packet['admission']
  need(intent['admission']==admission and admission=={'closed_actual_cpu_seconds':10568,'closed_actual_gpu_seconds':1483,
   'cumulative_cpu_preparation_allowance':3600,'candidate_cpu_reservation':4800,'candidate_gpu_reservation':600,
   'total_cpu_seconds':18968,'total_gpu_seconds':2083,'ceiling_cpu_seconds':57600,'ceiling_gpu_seconds':7200},'admission differs')
  expected={**packet['new_input_files'],'packet.json':args.packet_sha256}
  need(intent['pins']==expected and all(Path(n).name==n for n in expected),'flat exact launch pin roster differs')
  def committed(rel):
   return subprocess.check_output(['git','--no-replace-objects','show',args.launch_commit+':'+rel],cwd=REPO,timeout=10)
  for name,pin in expected.items():
   raw=read(LAUNCH/name);need(sha(raw)==pin and committed(str((LAUNCH/name).relative_to(REPO)))==raw,'committed input differs: '+name)
  for name in ('launch_once.py','pull_terminal.py'):
   need(committed(str((LAUNCH/name).relative_to(REPO)))==read(LAUNCH/name),'committed controller differs: '+name)
  need(read(LAUNCH/'pull_terminal.py')==read(Path(__file__)),'running collector differs from packet commit')
  need(sha(read(LAUNCH/'launch_once.py'))==intent['launcher_sha256'] and sha(committed(REVIEW))==intent['review_sha256'],'launcher/review pin differs')
  save('committed-packet-review.md',committed(REVIEW))
  source='\n'.join(k+'='+repr(v) for k,v in {'ROOT':ROOT,'JOB':args.job_id}.items())+'\n'+REMOTE
  save('remote_inventory.py',source.encode())
  def observe(name):return json.loads(run(name,['bash',HELPER,'run',shlex.join(['python3','-I','-B','-'])],input=source,env={**os.environ,'CURC_SOCKET':'/tmp/defab-curc.sock'}))
  def parent(v):
   r=v['results']['accounting'];need(r['returncode']==0,'sacct parent unavailable')
   rows=[l.split('|') for l in r['stdout'].splitlines() if l.startswith(args.job_id+'|')]
   need(len(rows)==1 and len(rows[0])>=11 and rows[0][3].split()[0] in TERMINAL,'not conclusively terminal')
   q=v['results']['queue'];need(q['returncode']==0 and not any(l.split('|')[0]==args.job_id for l in q['stdout'].splitlines()),'target still queued or queue unavailable')
   return rows[0]
  before=observe('inventory-before-transport.json');row=parent(before)
  need('inventory_error' not in before,'remote inventory incomplete; inspect retained error')
  # Root/packet mismatch does not justify pulling a different root or retrying.
  need(before['files'].get('packet.json',{}).get('sha256')==args.packet_sha256,'remote root packet differs')
  size=sum(v['bytes'] for v in before['files'].values())
  for dest in claimed:
   fs=os.statvfs(dest);need(fs.f_bavail*fs.f_frsize>size*2+2*1024**3,'insufficient complete two-copy space')
  run('transfer.json',['scp','-r','-o','ControlPath=/tmp/defab-curc.sock','-o','ControlMaster=no','-o','BatchMode=yes',
   '-o','PasswordAuthentication=no','-o','KbdInteractiveAuthentication=no','-o','PubkeyAuthentication=no',
   'paco0228@login.rc.colorado.edu:'+ROOT,str(EXT/'remote')],timeout=300)
  actual=inventory(EXT/'remote');need(actual==before['files'],'external transferred roster/bytes differ')
  after=observe('inventory-after-transport.json');after_row=parent(after)
  need('inventory_error' not in after and after['files']==actual,'second remote file inventory differs')
  mirror_partial();need(inventory(LOCAL/'remote')==actual,'repository raw roster/bytes differ')
  errors=[]
  def check(ok,message):
   if not ok:errors.append(message)
  check(after_row[:9]==row[:9],'parent accounting changed between observations')
  check(after['submission_journals']==before['submission_journals'],'journal census changed between observations')
  check(inventory(LAUNCH)==before_launch and inventory(LAUNCH_EXT)==before_launch,'launch custody changed during collection')
  mismatches=[n for n,pin in expected.items() if actual.get(n,{}).get('sha256')!=pin]
  check(not mismatches,'independently pinned remote input bytes differ: '+repr(mismatches))
  raw_root=EXT/'remote'
  ri_raw=read(raw_root/'submission.intent.json');rr_raw=read(raw_root/'submission.result.json')
  ri,rr=json.loads(ri_raw),json.loads(rr_raw)
  check(sha(ri_raw)==result['intent_sha256'] and rr==result,'remote submission journal binding differs')
  check(all(ri.get(k)==v for k,v in intent.items()),'remote intent differs from independently pinned local intent')
  # No success is assumed from counts; compare every historical row and journal pin.
  def census(v):
   r=v['results']['census'];need(r['returncode']==0,'fresh all-chain sacct unavailable')
   rows=[l.split('|')[:8] for l in r['stdout'].splitlines() if '|ace_llm' in l]
   need(all(len(r)==8 and r[0].isdigit() for r in rows),'invalid census rows')
   need(len({r[0] for r in rows})==len(rows),'duplicate census job IDs')
   return sorted(rows,key=lambda r:int(r[0]))
  chain_ok=False;chain_cpu=chain_gpu=None
  try:
   prior=sorted([r[:8] for r in ri['prior_accounting']],key=lambda r:int(r[0]))
   old_journals=ri['prior_journals'];fresh=census(before)
   expected_rows=sorted(prior+[[row[0],*row[2:9]]],key=lambda r:int(r[0]))
   expected_journals={**old_journals,ROOT+'/submission.intent.json':sha(ri_raw),ROOT+'/submission.result.json':sha(rr_raw)}
   chain_checks=[len(prior)==37,len({r[0] for r in prior})==37,args.job_id not in {r[0] for r in prior},
    len(old_journals)==76,cost(prior)==(10568,1483),len(fresh)==38,fresh==expected_rows,
    fresh==census(after),all(r[2].split()[0] in TERMINAL for r in fresh),len(expected_journals)==78,
    before['submission_journals']==expected_journals,after['submission_journals']==expected_journals]
   chain_cpu,chain_gpu=cost(fresh);chain_ok=all(chain_checks)
   check(chain_ok,'fresh 38-terminal-job/78-journal exact extension conflict')
   save('chain-reconciliation.json',{'matched':chain_ok,'prior_accounting':prior,'fresh_accounting':fresh,'expected_accounting':expected_rows,
    'expected_journals':expected_journals,'observed_journals':before['submission_journals'],'observed_cpu_seconds':chain_cpu,'observed_gpu_seconds':chain_gpu,'checks':chain_checks})
  except (ValueError,KeyError,TypeError,IndexError) as e:
   errors.append('all-chain reconciliation unavailable/conflicted: '+str(e));save('chain-reconciliation.json',{'matched':False,'error':str(e)})
  cpus=elapsed=gpus=cpu_cost=gpu_cost=None;resource_ok=False
  try:
   need(row[5].isdigit() and row[6].isdigit(),'unparseable allocation accounting')
   cpus,elapsed=int(row[5]),int(row[6]);alloc=pairs(row[8]);gpus=gpu_count(alloc,cpus,elapsed)
   cpu_cost=cpus*elapsed;gpu_cost=None if gpus is None else gpus*elapsed
   details=before['results']['details'];dr=[l.split('|') for l in (details.get('stdout') or '').splitlines() if l.strip()]
   need(details['returncode']==0 and len(dr)==1 and len(dr[0])>=7,'scheduler request evidence unavailable')
   d=dr[0];req=pairs(d[5])
   allocation_path=raw_root/'qualification/allocation.json';fields={}
   if allocation_path.exists():fields=json.loads(read(allocation_path)).get('fields',{})
   if not fields and before['results']['job']['returncode']==0:
    fields=dict(re.findall(r'([^\s=]+)=(\S+)',before['results']['job']['stdout']))
   identity=(row[0]==row[1]==args.job_id and row[2]=='ace_llm_shape01' and row[7]=='ucb736_asc2'
    and d[:5]==[args.job_id,'aa100','gpu-testing','10','4'] and d[6]=='paco0228')
   no_requeue=(fields.get('JobId')==args.job_id and fields.get('Requeue')=='0' and fields.get('Restarts')=='0')
   resource_ok=(identity and no_requeue and requested(req) and 4<=cpus<=8 and 0<=elapsed<=600 and gpus==1
    and alloc.get('cpu')==str(cpus) and alloc.get('mem') in ('12G','12.00G','12288M') and alloc.get('node')=='1'
    and alloc.get('gres/gpu:a100_3g.20gb')=='1'
    and not any(k.startswith('gres/') and k not in ('gres/gpu','gres/gpu:a100_3g.20gb') for k in alloc))
   check(resource_ok,'Slurm identity/request/allocation/no-requeue/time-cap contract mismatch')
   check(after['results']['details'].get('returncode')==0 and after['results']['details'].get('stdout')==details['stdout'],'scheduler request changed between reads')
  except (ValueError,KeyError,TypeError,IndexError) as e:errors.append('resource/accounting uncertainty: '+str(e))
  expected_cpu=None if cpu_cost is None else 10568+cpu_cost
  expected_gpu=None if gpu_cost is None else 1483+gpu_cost
  check(chain_cpu==expected_cpu and chain_gpu==expected_gpu and None not in (expected_cpu,expected_gpu),'actual cumulative costs unresolved/conflicted')
  if expected_cpu is not None:check(expected_cpu+3600<=57600,'CPU ceiling exceeded')
  if expected_gpu is not None:check(expected_gpu<=7200,'GPU ceiling exceeded')
  ends={}
  for n in actual:
   if Path(n).name in ('terminal.json','failure.json','failure-detail.json','cleanup.json'):
    try:
     v=json.loads(read(raw_root/n));ends[n]={k:v[k] for k in ('status','exception_type','type','message','completed_steps','training_execution_qualified','scientific_qualified','group_still_exists','leader_returncode') if k in v and isinstance(v[k],(str,int,float,bool,type(None)))}
    except (ValueError,TypeError):ends[n]={'summary_parse_failed':True}
  status='complete_custody_with_conflicts' if errors else 'complete_custody_independent_review_pending'
  disposition={'utc':utc(),'status':status,'job_id':args.job_id,'packet_sha256':args.packet_sha256,
   'scheduler_state':row[3],'scheduler_exit':row[4],'observed_cpus':cpus,'observed_elapsed_seconds':elapsed,'observed_gpu_slices':gpus,
   'allocated_cpu_seconds':cpu_cost,'gpu_slice_seconds':gpu_cost,'resource_contract_matched':resource_ok,
   'fresh_38_jobs_78_journals_matched':chain_ok,'observed_chain_cpu_seconds':chain_cpu,'observed_chain_gpu_seconds':chain_gpu,
   'known_closed_cpu_seconds':expected_cpu if chain_ok and not errors else None,'known_closed_gpu_seconds':expected_gpu if chain_ok and not errors else None,
   'candidate_closed_cpu_seconds':expected_cpu,'candidate_closed_gpu_seconds':expected_gpu,
   'cpu_admission_after_release':expected_cpu+3600 if chain_ok and not errors else None,
   'candidate_reservation_released':chain_ok and not errors,'cpu_ceiling_seconds':57600,'gpu_ceiling_seconds':7200,
   'raw_files':len(actual),'raw_bytes':sum(v['bytes'] for v in actual.values()),'files':actual,'launch_files':before_launch,
   'remote_inventories_matched':True,'both_local_raw_copies_matched':True,'input_pin_mismatches':mismatches,'conflicts':errors,
   'terminal_and_failure_summaries':ends,'training_gpu_qualified':False,'training_execution_qualified':False,
   'runtime_contract_adopted':False,'scientific_qualified':False,'retry_allowed':False,
   'local_collection_process_cpu_seconds':time.process_time()-cpu_start,'local_collection_elapsed_seconds':time.monotonic()-started,
   'cost_scope':'Local process counters exclude transfer/helper subprocess CPU; Slurm allocation costs are separately observed, not utilization evidence.',
   'acceptance_scope':'Custody/accounting only; exit 0 and retained terminal claims never qualify runtime, gradients, reload, GPU work or science.'}
  save('disposition.json',disposition)
  need(inventory(EXT)==inventory(LOCAL),'final exact two-copy custody differs')
  print(json.dumps({k:v for k,v in disposition.items() if k not in ('files','launch_files')},indent=2))
 except BaseException as e:
  if claimed:
   record={'utc':utc(),'status':'collector_failed_partial_custody','type':type(e).__name__,'message':str(e),'retry_allowed':False,'job_actions':False}
   try:mirror_partial()
   except BaseException as mirror_error:record['partial_mirror_error']=str(mirror_error)
   save('collector-failure.json',record)
  raise

if __name__=='__main__':main()

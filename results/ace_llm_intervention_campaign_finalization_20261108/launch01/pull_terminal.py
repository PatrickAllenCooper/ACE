#!/usr/bin/env python3
"""Future exclusive CPU final01 terminal custody, adapted from GPU01 collector.

Requires an unambiguous submitted job, terminal parent accounting and absence
from the queue. Collects every file in new launch and old common roots, including
all evolving campaign/private curves and failed-attempt evidence. Rechecks the
remote inventory after transfer. No scoring, retry, job mutation or acceptance.
Partial copies are retained; an existing terminal01 destination blocks a rerun.
"""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, shutil, subprocess, time

REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LAUNCH=REPO/'results/ace_llm_intervention_campaign_finalization_20261108/launch01'
LOCAL=LAUNCH.parent/'terminal01'
EXT=BASE/'ACE-LLM-intervention-campaign-finalization-terminal-20261108-01'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_finalization_20261108_01'
COMMON='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03'
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'

def need(ok,msg):
 if not ok:raise ValueError(msg)
def raw(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def save(name,v):
 for root in (EXT,LOCAL):
  with (root/name).open('xb') as f:f.write(raw(v));f.flush();os.fsync(f.fileno())

def main():
 meta=json.loads((LAUNCH/'packet-metadata.json').read_bytes())
 need(meta['remote_root']==ROOT and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==meta['root_files']['pull_terminal.py'],'captured collector/root differs')
 result=json.loads((LAUNCH/'submission.result.json').read_bytes());job=result['job_id']
 need(result.get('returncode')==0 and isinstance(job,str) and job.isdigit(),'submission unresolved; reconcile first')
 roots={'launch':ROOT,'common':COMMON}
 source='\n'.join(k+'='+repr(v) for k,v in dict(ROOTS=roots,JOB=job).items())+r'''
from pathlib import Path
import subprocess,hashlib,json,datetime,stat
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':{},'files':{}}
for k,c in {'accounting':['sacct','-j',JOB,'-n','-P','--format=JobIDRaw,JobID,JobName%60,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180,MaxRSS,TotalCPU'],
 'details':['sacct','-X','-j',JOB,'-n','-P','--format=Partition,QOS,TimelimitRaw,ReqTRES%180,User'],
 'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],'job':['scontrol','show','job','-o',JOB]}.items():
 p=subprocess.run(c,capture_output=True,text=True,timeout=20);v['results'][k]={'command':c,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
if any(v['results'][k]['returncode'] for k in ('accounting','queue')):raise ValueError('terminal scheduler observation failed')
rows=[l.split('|') for l in v['results']['accounting']['stdout'].splitlines() if l.startswith(JOB+'|')]
if len(rows)!=1 or rows[0][3] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'):raise ValueError('job not conclusively terminal')
if any(l.split('|')[0]==JOB or '|ace_llm' in l for l in v['results']['queue']['stdout'].splitlines()):raise ValueError('related job active; custody not frozen')
v['parent_accounting']=rows[0]
for label,path in ROOTS.items():
 root=Path(path)
 if not root.is_dir() or root.is_symlink():raise ValueError('ordinary retained root required')
 for p in sorted(root.rglob('*')):
  if p.is_symlink():raise ValueError('raw symlink forbidden')
  if p.is_dir():continue
  before=p.stat()
  if not stat.S_ISREG(before.st_mode):raise ValueError('nonregular raw custody')
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b)
  after=p.stat()
  if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('raw file changed during capture')
  v['files'][label+'/'+p.relative_to(root).as_posix()]={'bytes':after.st_size,'sha256':h.hexdigest()}
print(json.dumps(v))
'''
 def observe():
  p=subprocess.run(['bash',HELPER,'run','python3 -'],input=source,capture_output=True,text=True,timeout=180)
  transport=dict(source=source,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
  return transport,json.loads(p.stdout) if p.returncode==0 else None
 locks=[];claimed=False;started=time.monotonic();cpu=time.process_time()
 try:
  for n in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
   h=open(BASE/n,'r+')
   try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BaseException:h.close();raise
   locks.append(h)
  for dest in (EXT,LOCAL):need(not dest.exists() and not dest.is_symlink(),'terminal custody already claimed; no blind retry')
  transport,v=observe();need(v is not None,'terminal observation failed; no custody claim: '+transport['stderr'])
  size=sum(x['bytes'] for x in v['files'].values())
  for parent in (BASE,LOCAL.parent):
   fs=os.statvfs(parent);need(fs.f_bavail*fs.f_frsize>size*2+2*1024**3,'insufficient full two-copy custody space')
  for dest in (EXT,LOCAL):dest.mkdir();(dest/'remote').mkdir()
  claimed=True;save('inventory_transport.json',transport)
  for label,root in roots.items():
   p=subprocess.run(['scp','-r','-o','ControlPath=/tmp/defab-curc.sock','-o','BatchMode=yes',
    'paco0228@login.rc.colorado.edu:'+root,str(EXT/'remote'/label)],capture_output=True,text=True,timeout=1800)
   save('transfer-'+label+'.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
   need(p.returncode==0,'partial custody retained; reconcile, never rerun blindly')
  actual={}
  for p in sorted((EXT/'remote').rglob('*')):
   need(not p.is_symlink(),'local symlink')
   if p.is_dir():continue
   need(p.is_file(),'nonregular local custody');h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1024**2),b''):h.update(b)
   actual[p.relative_to(EXT/'remote').as_posix()]=dict(bytes=p.stat().st_size,sha256=h.hexdigest())
  need(actual==v['files'],'complete custody membership/bytes differs')
  after_transport,after=observe();save('inventory_after_transport.json',after_transport)
  need(after is not None and after['files']==actual and after['parent_accounting'][:9]==v['parent_accounting'][:9],
       'upstream changed during custody; retain and reconcile')
  for label in roots:shutil.copytree(EXT/'remote'/label,LOCAL/'remote'/label)
  for n,entry in actual.items():
   p=LOCAL/'remote'/n;h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1024**2),b''):h.update(b)
   need(p.stat().st_size==entry['bytes'] and h.hexdigest()==entry['sha256'],'second custody copy differs')
  r=v['parent_accounting'];cpus=int(r[5]);elapsed=int(r[6]);parts=[x.split('=',1) for x in r[8].split(',') if x]
  need(all(len(x)==2 for x in parts) and len(dict(parts))==len(parts),'malformed terminal TRES')
  tres=dict(parts);errors=[]
  details=v['results']['details'];dr=[l.split('|')[:5] for l in details['stdout'].splitlines() if l.strip()]
  requested_ok=False
  if details['returncode']==0 and len(dr)==1 and dr[0][:3]==['acpu','cpu-normal','5'] and dr[0][4]=='paco0228':
   rp=[x.split('=',1) for x in dr[0][3].split(',') if x]
   if all(len(x)==2 for x in rp) and len(dict(rp))==len(rp):
    req=dict(rp);requested_ok=(req.get('cpu')=='2' and req.get('mem') in ('7680M','7.5G','7.50G')
     and req.get('node')=='1' and not any(k.startswith('gres/') for k in req))
  if not requested_ok:errors.append('requested account/partition/QOS/resource evidence missing or mismatched')
  if not tres and not cpus==elapsed==0:errors.append('missing allocation inventory')
  gres={k:x for k,x in tres.items() if k.startswith('gres/')}
  if gres:errors.append('unexpected GPU/generic-resource allocation on CPU finalizer')
  generic=tres.get('gres/gpu');typed=[int(x) for k,x in gres.items() if k.startswith('gres/gpu:') and x.isdigit()]
  gpu=int(generic) if generic is not None and generic.isdigit() else sum(typed)
  if gres and (any(not x.isdigit() or not (k=='gres/gpu' or k.startswith('gres/gpu:')) for k,x in gres.items())
       or generic is not None and typed and gpu!=sum(typed)):gpu=None;errors.append('uncertain GPU accounting')
  if not tres and (cpus or elapsed):gpu=None
  expected={'launch/inputs/'+n:h for n,h in meta['input_files'].items()}
  expected.update({'launch/'+n:h for n,h in meta['root_files'].items()})
  expected['launch/packet-metadata.json']=hashlib.sha256((LAUNCH/'packet-metadata.json').read_bytes()).hexdigest()
  mismatches=[n for n,h in expected.items() if actual.get(n,{}).get('sha256')!=h]
  remote_result=json.loads((EXT/'remote/launch/submission.result.json').read_bytes())
  remote_intent=(EXT/'remote/launch/submission.intent.json').read_bytes()
  need(remote_result==result and remote_result['intent_sha256']==hashlib.sha256(remote_intent).hexdigest(),'submission journal lineage differs')
  values={}
  for n in ('launch/finalization/attempt.json','launch/finalization/failure.json','launch/finalization/worker/terminal.json',
            'launch/finalization/worker.cleanup.json','common/cpu03/worker/artifacts/campaign/finalization.json'):
   p=EXT/'remote'/n
   if p.is_file():values[n]=json.loads(p.read_bytes())
  a=meta['admission'];cost=cpus*elapsed;gcost=None if gpu is None else gpu*elapsed
  disposition=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='complete_custody_independent_review_pending',
   job_id=job,scheduler_state=r[3],scheduler_exit=r[4],account=r[7],allocated_cpus=cpus,elapsed_seconds=elapsed,gpu_count=gpu,alloc_tres=r[8],
   resource_contract_matched=(requested_ok and r[2]=='ace_llm_campaign_final01' and r[7]=='ucb736_asc2' and cpus==2 and tres.get('cpu')=='2'
    and tres.get('mem') in ('7680M','7.5G','7.50G') and tres.get('node')=='1' and not gres and 0<=elapsed<=300),
   allocated_cpu_seconds=cost,gpu_slice_seconds=gcost,accounting_errors=errors,closed_cpu_seconds=a['closed_cpu_seconds']+cost,
   closed_gpu_slice_seconds=None if gcost is None else a['closed_gpu_slice_seconds']+gcost,
   cumulative_preparation_cpu_allowance=3600,cpu_admission_after_release=a['closed_cpu_seconds']+cost+3600,
   cpu_ceiling_seconds=57600,gpu_ceiling_seconds=7200,raw_files=len(actual),raw_bytes=size,files=actual,input_pin_mismatches=mismatches,
   terminal_records=values,finalization_qualified=False,scientific_qualified=False,retry_allowed=False,
   publication_authority='Original campaign/finalization.json, even if outer checks fail; reconcile before any further action.',
   collector_elapsed_seconds=time.monotonic()-started,collector_process_cpu_seconds=time.process_time()-cpu,
   cost_scope='Local collector process only, excluding transfer and prior preparation; not whole-cycle CPU.')
  save('disposition.json',disposition)
  for dest in (EXT,LOCAL):
   with (dest/'pull_terminal.py').open('xb') as f:f.write(Path(__file__).read_bytes())
  print(json.dumps({k:x for k,x in disposition.items() if k not in ('files','terminal_records')},indent=2))
 except BaseException as exc:
  if claimed:save('collector_failure.json',dict(type=type(exc).__name__,message=str(exc),retry_allowed=False))
  raise
 finally:
  for h in reversed(locks):fcntl.flock(h,fcntl.LOCK_UN);h.close()

if __name__=='__main__':main()

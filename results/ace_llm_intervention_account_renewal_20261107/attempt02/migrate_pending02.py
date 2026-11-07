"""Once-only pending ACE job account migration after user-provided renewal."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess
REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_account_renewal_20261107/attempt02'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-account-renewal-20261107-02')
OLD=REPO/'results/ace_llm_intervention_import_diagnostic_20261107'
def save(name,value):
 raw=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
 for root in [EXT,LOCAL]:
  with (root/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
EXT.mkdir(exist_ok=False)
(EXT/'migrate_pending02.py').write_bytes(Path(__file__).read_bytes())
intent={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'35341862','old_account':'ucb736_asc1','new_account':'ucb736_asc2','authority':'Patrick supplied CURC renewal INC1508105 in this chat, 2026-11-07. Routine approved project continuation.','renewal_su':450000,'renewal_expiry':'2027-11-06','research_ceiling_unchanged':{'cpu_seconds':57600,'gpu_slice_seconds':7200},'operation':'scontrol update JobId=35341862 Account=ucb736_asc2','new_submission':False,'resource_change':False,'original_pins':{n:hashlib.sha256((OLD/n).read_bytes()).hexdigest() for n in ['job.sbatch','packet.json','probe.py','submission.intent.json','submission.result.json']},'controller_pid_local':os.getpid(),'scheduler_reference':'https://slurm.schedmd.com/scontrol.html'}
intent['original_pins']['submission.intent.json']=json.loads((OLD/'submission.result.json').read_bytes())['intent_sha256']
intent['original_local_intent']=json.loads((OLD/'submission.intent.json').read_bytes())
intent['preflight_correction']='Original local intent is the pre-submission form. The original independently committed result pins enriched remote intent. Preflight01 rejected before mutation, remote reconciliation found no account-transition records.'
save('intent.json',intent)
locks=[]
try:
 for n in ['ace-delivery-controller.lock','ace-foundation-controller.lock']:
  f=open(EXT.parent/n,'r+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 save('locks.json',{'held':[str(f.name) for f in locks],'pid':os.getpid(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 remote='''import pathlib,json,hashlib,subprocess,datetime,os
intent=INTENT
root=pathlib.Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_import_diagnostic_20261107_01')
def run(c):
 p=subprocess.run(c,capture_output=True,text=True,timeout=20)
 r={'command':c,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 if p.returncode:raise RuntimeError(json.dumps(r))
 return r
def write(n,v):
 with (root/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\\n');f.flush();os.fsync(f.fileno())
if any(root.glob('account-transition01.*.json')):raise RuntimeError('prior account migration evidence exists; reconcile, never retry')
pre=run(['scontrol','show','job','-o','35341862'])
fields=dict(t.split('=',1) for t in pre['stdout'].split() if '=' in t)
for k,v in {'JobId':'35341862','UserId':'paco0228(348288)','JobState':'PENDING','Account':'ucb736_asc1','JobName':'ace_llm_import01','Partition':'acpu','QOS':'cpu-normal','Requeue':'0','Restarts':'0','RunTime':'00:00:00','TimeLimit':'00:08:00','NumCPUs':'1','NumTasks':'1','CPUs/Task':'1','MinMemoryNode':'3G','ReqTRES':'cpu=1,mem=3G,node=1','Command':str(root/'job.sbatch')}.items():
 if fields.get(k)!=v:raise RuntimeError('precondition differs '+k+': '+str(fields.get(k)))
queue=run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'])
related=[x for x in queue['stdout'].splitlines() if '|ace_llm' in x]
if len(related)!=1 or not related[0].startswith('35341862|'):raise RuntimeError('related queue changed')
assoc=run(['sacctmgr','-n','-P','show','assoc','where','account=ucb736_asc2','user=paco0228','format=Cluster,Account,User,Partition,MaxJobs,GrpJobs,MaxSubmitJobs,GrpSubmitJobs,QOS'])
rows=[x.split('|') for x in assoc['stdout'].splitlines() if x.strip()]
if len(rows)!=1 or rows[0][:3]!=['alpine','ucb736_asc2','paco0228'] or rows[0][4]=='0' or 'cpu-normal' not in rows[0][8].split(','):raise RuntimeError('renewed association not qualified')
for n,h in intent['original_pins'].items():
 if hashlib.sha256((root/n).read_bytes()).hexdigest()!=h:raise RuntimeError('original pin drift '+n)
original_intent=json.loads((root/'submission.intent.json').read_bytes())
if any(original_intent.get(k)!=v for k,v in intent['original_local_intent'].items()):raise RuntimeError('original remote declaration differs from local intent')
journals={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for d in root.parent.glob('ace_llm_intervention*') for p in d.glob('submission.*.json')}
if len(journals)!=30:raise RuntimeError('submission journal inventory changed')
accounting=run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'])
closed=[l.split('|') for l in accounting['stdout'].splitlines() if '|ace_llm' in l and not l.startswith('35341862|')]
if len(closed)!=13 or any(v[2].split()[0] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for v in closed):raise RuntimeError('prior job dispositions changed')
if sum(int(v[4])*int(v[5]) for v in closed)!=4450 or sum(int(v[5]) for v in closed if 'gres/gpu:' in v[7])!=836:raise RuntimeError('prior accounting changed')
probe=root/'account-transition01.write-probe'
with probe.open('xb') as f:f.write(b'ACE renewed account readiness\\n');f.flush();os.fsync(f.fileno())
if probe.read_bytes()!=b'ACE renewed account readiness\\n':raise RuntimeError('scratch readback differs')
probe.unlink()
fs=os.statvfs(root)
intent.update(remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pre_job=pre,queue=queue,association=assoc,original_journals=journals,prior_closed=closed,scratch={'stat':{'uid':root.stat().st_uid,'mode':oct(root.stat().st_mode)},'free_bytes':fs.f_bavail*fs.f_frsize,'df':run(['df','-h',str(root)]),'write_read_cleanup':True})
write('account-transition01.intent.json',intent)
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'intent_sha256':hashlib.sha256((root/'account-transition01.intent.json').read_bytes()).hexdigest(),'job_id':'35341862','new_submission':False}
try:
 c=['scontrol','update','JobId=35341862','Account=ucb736_asc2']
 p=subprocess.run(c,capture_output=True,text=True,timeout=20)
 result.update(update={'command':c,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 result['after']=run(['scontrol','show','job','-o','35341862'])
 f=dict(t.split('=',1) for t in result['after']['stdout'].split() if '=' in t)
 result['verified']=p.returncode==0 and f.get('Account')=='ucb736_asc2' and all(f.get(k)==fields[k] for k in ['JobId','Command','TimeLimit','NumTasks','CPUs/Task','MinMemoryNode','Partition','QOS','Requeue','Restarts','ReqTRES'])
except BaseException as exc:result.update(ambiguous=True,exception_type=type(exc).__name__,message=str(exc),verified=False)
write('account-transition01.result.json',result)
print(json.dumps({'intent':intent,'result':result}))
'''.replace('INTENT',repr(intent),1)
 save('remote_program.json',{'python_source':remote})
 p=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','python3 -'],input=remote,capture_output=True,text=True,timeout=120)
 transport={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 save('transport.json',transport)
 if p.returncode!=0:raise RuntimeError('migration transport nonzero; inspect saved evidence')
 response=json.loads(p.stdout);save('remote_receipts.json',response)
 print(json.dumps(response['result'],indent=2))
except BaseException as exc:
 save('exception.json',{'type':type(exc).__name__,'message':str(exc),'blind_retry_forbidden':True})
 raise
finally:
 for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
 save('locks_released.json',{'pid':os.getpid(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'released':True})

#!/usr/bin/env python3
"""Future once-only GPU01 coordinator, following the CPU03 launcher pattern.

Do not run during source preparation. Invocation requires --review (a concrete
packet review committed in this repo) and its independently supplied SHA256.
--commit defaults to HEAD and must contain the exact packet, builder, launcher,
existing component review snapshots and new concrete review. No review schema.
No GPU qualification or successor follows submission, even if the job exits 0.
The parent retains responsibility for live follow-up, complete terminal custody,
requested/observed resource reconciliation and independent GPU acceptance.
"""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, shlex, subprocess

REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_campaign_gpu_20261108/launch01'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-gpu-20261108-01')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_gpu_20261108_01'
CPUROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03'
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
ENTRY='qualify_llm_intervention_campaign_gpu.py'

def sha(b):return hashlib.sha256(b).hexdigest()
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,value):
 for root in (EXT,LOCAL):
  with (root/name).open('xb') as f:f.write(raw(value));f.flush();os.fsync(f.fileno())

# Only byte admission and scheduler observations run on the login node. The
# reviewed outer's authenticate/admit_cpu helpers load no model or tensor and
# perform no stage, numerical work, campaign execution or private finalization.
REMOTE_COMMON=r'''
from pathlib import Path
import datetime,hashlib,json,os,re,subprocess,time,types
root=Path(ROOT);cpu_root=Path(CPUROOT)
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(command):
 p=subprocess.run(command,capture_output=True,text=True,timeout=20)
 need(p.returncode==0,json.dumps({'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
 return p.stdout
def write(name,value):
 with (root/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())
outer=types.ModuleType('gpu01_byte_admission');outer.__file__=str(root/'inputs'/ENTRY)
need(sha(OUTER_SOURCE.encode())==META['input_files'][ENTRY],'outer source pin differs')
exec(compile(OUTER_SOURCE,outer.__file__,'exec'),outer.__dict__)
def tres(text):
 result={}
 for part in text.split(',') if text else []:
  key,sep,value=part.partition('=');need(sep and key not in result and value,'malformed TRES');result[key]=value
 return result
def gpu_count(values,cpus,elapsed):
 g={k:v for k,v in values.items() if k.startswith('gres/')}
 need(all(k=='gres/gpu' or k.startswith('gres/gpu:') for k in g) and all(v.isdigit() for v in g.values()),'unknown GPU allocation')
 typed=sum(int(v) for k,v in g.items() if k.startswith('gres/gpu:'))
 generic=int(g['gres/gpu']) if 'gres/gpu' in g else None
 need(generic is None or not any(k.startswith('gres/gpu:') for k in g) or generic==typed,'conflicting GPU counts')
 need(values or cpus==elapsed==0,'missing allocation inventory is uncertain')
 return generic if generic is not None else typed
def queue():
 value=run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'])
 need(not any('|ace_llm' in line for line in value.splitlines()),'related active job; do not submit')
 return value

def cpu_guard():
 need(not Path(PACKET['paths']['gpu_output']).exists() and not Path(PACKET['paths']['gpu_output']).is_symlink(),'fixed GPU output already claimed')
 need(outer.inventory(cpu_root/'inputs')==dict(OLD['files'],**{'packet.json':META['cpu_packet_sha256']}),'original CPU input roster/bytes changed')
 need(outer.read(cpu_root/'inputs/packet.json',META['cpu_packet_sha256'])==CPU_PACKET_RAW.encode(),'original CPU packet differs')
 # This is the already-reviewed strict CPU acceptance gate, not a new schema.
 outer.admit_cpu(ACCEPTANCE,PACKET,OLD)

def reconcile():
 started=time.monotonic();q=queue()
 accounting=run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format',
  'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'])
 rows=[line.split('|')[:8] for line in accounting.splitlines() if '|ace_llm' in line]
 need(len(rows)==34 and all(len(r)==8 and r[0].isdigit() and r[4].isdigit() and r[5].isdigit()
  and r[2].split()[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for r in rows),'prior job census/state changed')
 rows=sorted(rows,key=lambda r:int(r[0]))
 need(len({r[0] for r in rows})==34 and rows==META['prior_accounting'],'exact relevant prior accounting changed')
 cpu_seconds=sum(int(r[4])*int(r[5]) for r in rows)
 gpu_seconds=sum(gpu_count(tres(r[7]),int(r[4]),int(r[5]))*int(r[5]) for r in rows)
 need(cpu_seconds==9702 and gpu_seconds==1347,'current closed accounting changed')
 need(cpu_seconds+3600+4800==META['admission']['cpu_admission']<=57600
  and gpu_seconds+600==META['admission']['gpu_slice_admission']<=7200,'current admission differs')
 journals={str(p):sha(outer.read(p)) for d in root.parent.glob('ace_llm_intervention*') for p in d.glob('submission.*.json')}
 need(len(journals)==70 and journals==META['prior_journals'],'exact relevant prior journal pins changed')
 details=run(['sacct','-X','-j','35387068','-n','-P','--format=Partition,QOS,TimelimitRaw,ReqTRES%180,User'])
 dr=[l.split('|')[:5] for l in details.splitlines() if l.strip()]
 need(len(dr)==1 and dr[0][:3]==['acpu','cpu-normal','10'] and dr[0][4]=='paco0228','CPU03 scheduler identity differs')
 requested=tres(dr[0][3]);need(requested.get('cpu')=='2' and requested.get('mem') in ('7680M','7.5G','7.50G')
  and not any(k.startswith('gres/') for k in requested),'CPU03 requested resources differ')
 partition=run(['scontrol','show','partition','aa100','-o'])
 limits=dict(part.split('=',1) for part in partition.split() if '=' in part)
 cap=limits.get('MaxMemPerCPU','')
 need(limits.get('PartitionName')=='aa100' and limits.get('State')=='UP' and cap.isdigit() and int(cap)>=3840,'GPU partition/memory ratio not admissible')
 for key,wanted in [('AllowAccounts','ucb736_asc2'),('AllowQos','gpu-testing')]:
  permitted=limits.get(key,'ALL');need(permitted=='ALL' or wanted in permitted.split(','),'partition admission excludes '+wanted)
 need('ucb736_asc2' not in limits.get('DenyAccounts','').split(',') and 'gpu-testing' not in limits.get('DenyQos','').split(','),'partition denies request')
 assoc=run(['sacctmgr','-n','-P','show','assoc','where','account=ucb736_asc2','user=paco0228','format=Cluster,Account,User,Partition,MaxJobs,QOS'])
 ar=[l.split('|') for l in assoc.splitlines() if l.strip()]
 need(any(len(r)>=6 and r[:3]==['alpine','ucb736_asc2','paco0228'] and r[3] in ('','aa100')
  and r[4]!='0' and 'gpu-testing' in r[5].split(',') for r in ar),'renewed account/QOS unavailable')
 gres=run(['sinfo','-h','-p','aa100','-o','%G'])
 need(re.search(r'gpu:a100_3g\.20gb:[1-9][0-9]*',gres) is not None,'requested GPU slice absent from partition')
 cpu_guard()
 manifest=Path(OLD['manifest']['path']);archive=Path(OLD['archive']['path'])
 outer.read(manifest,OLD['manifest']['sha256'])
 need(archive.is_file() and not archive.is_symlink() and archive.stat().st_size==8438650880,'accepted runtime archive metadata changed')
 # Full bulk archive authentication remains the existing bounded allocated stage.
 final_queue=queue()
 need(time.monotonic()-started<120,'readiness observations became stale')
 return {'utc':now(),'queue':q,'queue_before_mutation':final_queue,'prior_accounting':rows,'prior_journals':journals,
  'closed_cpu_seconds':cpu_seconds,'closed_gpu_slice_seconds':gpu_seconds,'cpu03_details':details,
  'partition':partition,'max_memory_per_cpu_mib':int(cap),'association':assoc,'configured_gres':gres,
  'capacity_note':'configured slice verified; no inference of immediate free capacity',
  'cpu_acceptance_sha256':META['cpu_acceptance_sha256'],'predecessor_manifest_sha256':OLD['manifest']['sha256'],
  'predecessor_archive_size':archive.stat().st_size,'bulk_archive_byte_authentication':'allocated stage before copied runtime executes'}

def verify_packet():
 for name,pin in PINS.items():outer.read(root/name,pin)
 need(outer.inventory(root/'inputs')==META['input_files'],'new input roster/readback differs')
 # Existing outer validates exact CPUfiles+5 extras+two receipts, original
 # CPUinputs, frozen commands, training-role bytes and untouched CPU campaign.
 outer.authenticate(str(root/'inputs/packet.json'),META['gpu_packet_sha256'])
'''

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--review',required=True)
 parser.add_argument('--review-sha256',required=True)
 parser.add_argument('--commit',default='HEAD')
 args=parser.parse_args()
 for root in (LOCAL,EXT):
  need(root.is_dir() and not root.is_symlink(),'built packet directory missing')
  need(not any((root/n).exists() for n in ('controller.json','submission.intent.json','submission.result.json','launcher_failure.json','readiness_transport.json')),
       'prior launch attempt exists; retain/reconcile, never blind retry')
 revision=subprocess.check_output(['git','rev-parse',args.commit+'^{commit}'],cwd=REPO,text=True).strip()
 meta=json.loads((LOCAL/'packet-metadata.json').read_bytes())
 need(meta['remote_root']==ROOT and meta['gpu_output']==CPUROOT+'/gpu03' and meta['cpu_input_directory']==CPUROOT+'/inputs','launch/output identity differs')
 files=['inputs/'+n for n in meta['input_files']]+list(meta['root_files'])+['packet-metadata.json']
 need(len(files)==len(set(files)) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in files),'unsafe packet filenames')
 pins={}
 for name in files:
  b=(LOCAL/name).read_bytes()
  need(not (LOCAL/name).is_symlink() and not (EXT/name).is_symlink() and b==(EXT/name).read_bytes(),'local/external packet differs')
  committed=subprocess.check_output(['git','show',revision+':'+str((LOCAL/name).relative_to(REPO))],cwd=REPO)
  need(b==committed,'commit does not bind '+name);pins[name]=sha(b)
 for n,h in meta['input_files'].items():need(pins['inputs/'+n]==h,'input metadata pin differs')
 for n,h in meta['root_files'].items():need(pins[n]==h,'root metadata pin differs')
 need(Path(__file__).read_bytes()==(LOCAL/'launch_once.py').read_bytes(),'running launcher is not committed launcher')
 review_path=Path(args.review).resolve(strict=True);review_bytes=review_path.read_bytes()
 need(sha(review_bytes)==args.review_sha256,'independent concrete review pin differs')
 need(subprocess.check_output(['git','show',revision+':'+str(review_path.relative_to(REPO))],cwd=REPO)==review_bytes,'concrete packet review not committed')
 packet=json.loads((LOCAL/'inputs/packet.json').read_bytes())
 old_raw=(LOCAL/'inputs/cpu-packet.json').read_bytes();old=json.loads(old_raw)
 acceptance_raw=(LOCAL/'inputs/cpu-acceptance.json').read_bytes();acceptance=json.loads(acceptance_raw)
 need(sha(acceptance_raw)==meta['cpu_acceptance_sha256'] and sha(old_raw)==meta['cpu_packet_sha256']
      and sha((LOCAL/'inputs/packet.json').read_bytes())==meta['gpu_packet_sha256'],'packet/acceptance independent pins differ')
 need(meta['admission']['closed_cpu_seconds']==9702 and meta['admission']['closed_gpu_slice_seconds']==1347
      and meta['admission']['new_cpu_reservation']==4800 and meta['admission']['new_gpu_slice_reservation']==600,'resource admission differs')
 intent={'utc':utc(),'packet_commit':revision,'source_commit':meta['source_commit'],'pins':pins,
  'gpu_packet_sha256':meta['gpu_packet_sha256'],'cpu_acceptance_sha256':meta['cpu_acceptance_sha256'],
  'cpu_disposition_sha256':meta['cpu_disposition_sha256'],'cpu_review_receipt_sha256':meta['cpu_review_receipt_sha256'],
  'review_path':str(review_path.relative_to(REPO)),'review_sha256':sha(review_bytes),'review_text':review_bytes.decode(),
  'launcher_sha256':sha(Path(__file__).read_bytes()),'remote_root':ROOT,'gpu_output':meta['gpu_output'],
  'controller_pid_local':os.getpid(),'resources':meta['resources'],'admission':meta['admission'],'blind_retry_forbidden':True}
 common='\n'.join(name+'='+repr(value) for name,value in {'ROOT':ROOT,'CPUROOT':CPUROOT,'ENTRY':ENTRY,'META':meta,'PINS':pins,
  'PACKET':packet,'OLD':old,'ACCEPTANCE':acceptance,'CPU_PACKET_RAW':old_raw.decode(),'OUTER_SOURCE':(LOCAL/'inputs'/ENTRY).read_text()}.items())+'\n'+REMOTE_COMMON
 remote_command=shlex.join([old['runtime_source']+'/bin/python','-I','-B','-'])
 def remote(name,source):
  r=subprocess.run(['bash',HELPER,'run',remote_command],input=source,capture_output=True,text=True,timeout=180)
  save(name,{'python_source':source,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'utc':utc()})
  need(r.returncode==0,name+' failed; retain and reconcile before any new action')
  return json.loads(r.stdout)
 locks=[]
 try:
  for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
   f=open(EXT.parent/name,'r+')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BaseException:f.close();raise
   locks.append(f)
  ready=remote('readiness_transport.json',common+'''
need(not root.exists() and not root.is_symlink(),'GPU launch root already exists; no retry')
r=reconcile()
need(root.parent.is_dir() and not root.parent.is_symlink() and root.parent.stat().st_uid==os.getuid(),'scratch parent ownership differs')
fs=os.statvfs(root.parent);need(fs.f_bavail*fs.f_frsize>2*1024**3,'insufficient output scratch space')
root.mkdir(mode=0o700);(root/'logs').mkdir();(root/'inputs').mkdir()
p=root/'write-read-probe'
with p.open('xb') as f:f.write(b'ACE GPU01 scratch readiness\\n');f.flush();os.fsync(f.fileno())
need(p.read_bytes()==b'ACE GPU01 scratch readiness\\n','scratch readback differs');p.unlink()
r['scratch']={'uid':root.stat().st_uid,'mode':oct(root.stat().st_mode),'free_bytes':fs.f_bavail*fs.f_frsize,'df':run(['df','-h',str(root)]),'write_read_cleanup':True}
write('readiness.json',r);print(json.dumps(r))
''')
  save('controller.json',{'held':[f.name for f in locks],'pid':os.getpid(),'utc':utc()})
  save('readiness.json',ready)
  for group in ('inputs','root'):
   names=[n for n in files if n.startswith('inputs/')==(group=='inputs')]
   target=ROOT+('/inputs/' if group=='inputs' else '/')
   r=subprocess.run(['scp','-o','ControlPath=/tmp/defab-curc.sock','-o','BatchMode=yes',*[str(LOCAL/n) for n in names],
        'paco0228@login.rc.colorado.edu:'+target],capture_output=True,text=True,timeout=180)
   save('transfer-'+group+'.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'utc':utc()})
   need(r.returncode==0,'transfer failed; retain partial packet, no retry')
  verified=remote('verification_transport.json',common+'''\nverify_packet()
print(json.dumps({'utc':now(),'verified':PINS,'cpu_originals_and_acceptance_rechecked':True}))
''')
  save('remote_verified.json',verified)
  save('submission.intent.json',intent)
  result=remote('submission_transport.json',common+'\nintent='+repr(intent)+'''
need(not (root/'submission.intent.json').exists() and not (root/'submission.result.json').exists(),'existing GPU submission journal')
verify_packet()
intent.update(reconcile())
# Re-read every independent new-packet pin immediately before claiming intent.
for name,pin in PINS.items():outer.read(root/name,pin)
queue()
intent['remote_utc']=now();write('submission.intent.json',intent)
r={'utc':now(),'intent_sha256':sha((root/'submission.intent.json').read_bytes()),'job_id':None,'gpu_qualified':False}
try:
 p=subprocess.run(['sbatch','--parsable',str(root/'job.sbatch')],capture_output=True,text=True,timeout=25)
 job=p.stdout.strip().split(';')[0]
 r.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,job_id=job if p.returncode==0 and job.isdigit() else None)
except BaseException as exc:r.update(ambiguous=True,exception_type=type(exc).__name__,message=str(exc))
write('submission.result.json',r);print(json.dumps(r))
''')
  save('submission.result.json',result)
  need(result['job_id'] is not None,'failed/ambiguous submission; never blind retry')
  print(json.dumps(result,indent=2))
 except BaseException as exc:
  save('launcher_failure.json',{'utc':utc(),'type':type(exc).__name__,'message':str(exc),'blind_retry_forbidden':True})
  raise
 finally:
  for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
  if locks:save('controller_released.json',{'pid':os.getpid(),'utc':utc(),'released':True})

if __name__=='__main__':main()

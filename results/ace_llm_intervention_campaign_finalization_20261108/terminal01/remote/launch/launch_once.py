#!/usr/bin/env python3
"""Future once-only CPU final01 coordinator. Do not run during preparation.

Requires completed independent GPU acceptance, a built packet, and --review plus
--review-sha256 for a NEW concrete packet review committed with that packet;
the review text must name the full finalizer packet SHA256.
Independent reviewers supply pins; no new review schema or automatic acceptance.
A failed/ambiguous attempt is retained and never retried. Parent owns monitoring,
terminal custody and independent final review; submission does not qualify output.
"""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, shlex, subprocess

REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_campaign_finalization_20261108/launch01'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-finalization-20261108-01')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_finalization_20261108_01'
GPUROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_gpu_20261108_01'
CPUROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03'
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
ENTRY='finalize_llm_intervention_campaign_cpu.py'
SOURCEPIN='88102baf56395e7ef06fc7e96aae3ed2d7b8eef4ecb8b71f77acfc7e4b8e813f'
RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,memory_mib=7680,wall_seconds=300,gpus=0,requeue=False)
MARKERS=('controller.json','submission.intent.json','submission.result.json','launcher_failure.json','readiness_transport.json')

def sha(b):return hashlib.sha256(b).hexdigest()
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,value):
 for root in (EXT,LOCAL):
  with (root/name).open('xb') as f:f.write(raw(value));f.flush();os.fsync(f.fileno())
def unclaimed():
 for root in (LOCAL,EXT):
  need(root.is_dir() and not root.is_symlink(),'built packet missing')
  need(not any((root/n).exists() or (root/n).is_symlink() for n in MARKERS),'prior launch attempt; reconcile, never retry')

# Only pinned stdlib byte checks and scheduler observations on the login node.
# The reviewed finalizer's authenticate/admit_evidence do not stage or finalize.
REMOTE_COMMON=r'''
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time,types
root=Path(ROOT)
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(c):
 p=subprocess.run(c,capture_output=True,text=True,timeout=20)
 need(p.returncode==0,json.dumps({'command':c,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
 return p.stdout
def write(n,v):
 with (root/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())
need(sha(OUTER_SOURCE.encode())==SOURCEPIN==META['input_files'][ENTRY],'finalizer source differs')
outer=types.ModuleType('final01_byte_checks');outer.__file__=str(root/'inputs'/ENTRY)
exec(compile(OUTER_SOURCE,outer.__file__,'exec'),outer.__dict__)
need(sha(GPU_SOURCE.encode())==outer.GPU_SHA,'GPU helper differs')
api=outer.module('final01_gpu_byte_checks',root/'inputs'/outer.GPU_ENTRY,GPU_SOURCE.encode())
def tres(text):
 d={}
 for s in text.split(',') if text else []:
  k,sep,v=s.partition('=');need(sep and k not in d and v,'malformed TRES');d[k]=v
 return d
def gpu_count(d,cpus,elapsed):
 g={k:v for k,v in d.items() if k.startswith('gres/')}
 need(all(k=='gres/gpu' or k.startswith('gres/gpu:') for k in g) and all(v.isdigit() for v in g.values()),'unknown GPU allocation')
 typed=sum(int(v) for k,v in g.items() if k.startswith('gres/gpu:'))
 generic=int(g['gres/gpu']) if 'gres/gpu' in g else None
 need(generic is None or not any(k.startswith('gres/gpu:') for k in g) or generic==typed,'GPU count conflict')
 need(d or cpus==elapsed==0,'missing allocation inventory')
 return generic if generic is not None else typed
def queue():
 q=run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'])
 need(not any('|ace_llm' in l for l in q.splitlines()),'related active job; do not submit')
 return q
def upstream():
 need(not Path(PACKET['paths']['output']).exists() and not Path(PACKET['paths']['output']).is_symlink(),'finalizer output already claimed')
 # Freeze both complete original roots, including their journals and campaign.
 for label,path in [('launch',GPUROOT),('common',CPUROOT)]:
  expected={n[len(label)+1:]:v['sha256'] for n,v in DISPOSITION['files'].items() if n.startswith(label+'/')}
  outer.checked_inventory(api,path,expected)
 for key,previous,pin in [('cpu_inputs',OLD,META['cpu_packet_sha256']),('gpu_inputs',GP,META['gpu_packet_sha256'])]:
  outer.checked_inventory(api,PACKET['paths'][key],dict(previous['files'],**{'packet.json':pin}))
 # Strict existing 16/17-field acceptance, full outputs, untouched campaign and
 # eight attempts before any sbatch. This is not the private finalizer call.
 outer.admit_evidence(PACKET,OLD,GP,EVIDENCE,api)
def reconcile():
 started=time.monotonic();q=queue()
 accounting=run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format',
  'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'])
 rows=[l.split('|')[:8] for l in accounting.splitlines() if '|ace_llm' in l]
 need(len(rows)==35 and all(len(r)==8 and r[0].isdigit() and r[4].isdigit() and r[5].isdigit()
  and r[2].split()[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for r in rows),'terminal job census changed')
 rows=sorted(rows,key=lambda r:int(r[0]))
 need(len({r[0] for r in rows})==35 and rows==META['prior_accounting'],'exact prior accounting changed')
 cpu=sum(int(r[4])*int(r[5]) for r in rows)
 gpu=sum(gpu_count(tres(r[7]),int(r[4]),int(r[5]))*int(r[5]) for r in rows)
 a=META['admission']
 need(cpu==a['closed_cpu_seconds']==DISPOSITION['closed_cpu_seconds'] and gpu==a['closed_gpu_slice_seconds']==DISPOSITION['closed_gpu_slice_seconds'],'closed accounting differs')
 need(a['new_cpu_reservation']==600 and a['new_gpu_slice_reservation']==0 and a['cumulative_preparation_cpu_allowance']==3600
  and cpu+3600+600==a['cpu_admission']<=57600 and gpu==a['gpu_slice_admission']<=7200,'admission differs')
 journals={str(p):sha(outer.read(p)) for d in root.parent.glob('ace_llm_intervention*') for p in d.glob('submission.*.json')}
 need(len(journals)==72 and journals==META['prior_journals'],'exact prior journals changed')
 details=run(['sacct','-X','-j','35388249','-n','-P','--format=Partition,QOS,TimelimitRaw,ReqTRES%180,User'])
 dr=[l.split('|')[:5] for l in details.splitlines() if l.strip()]
 need(len(dr)==1 and dr[0][:3]==['aa100','gpu-testing','10'] and dr[0][4]=='paco0228','GPU01 scheduler identity differs')
 req=tres(dr[0][3]);need(req.get('cpu')=='4' and req.get('mem') in ('15360M','15G','15.00G')
  and req.get('gres/gpu:a100_3g.20gb')=='1' and gpu_count(req,4,600)==1,'GPU01 request differs')
 partition=run(['scontrol','show','partition','acpu','-o'])
 limits=dict(p.split('=',1) for p in partition.split() if '=' in p);cap=limits.get('MaxMemPerCPU','')
 need(limits.get('PartitionName')=='acpu' and limits.get('State')=='UP' and cap.isdigit() and int(cap)>=3840,'CPU partition/memory ratio changed')
 for key,want in [('AllowAccounts','ucb736_asc2'),('AllowQos','cpu-normal')]:
  allow=limits.get(key,'ALL');need(allow=='ALL' or want in allow.split(','),'partition excludes '+want)
 need('ucb736_asc2' not in limits.get('DenyAccounts','').split(',') and 'cpu-normal' not in limits.get('DenyQos','').split(','),'partition denies request')
 assoc=run(['sacctmgr','-n','-P','show','assoc','where','account=ucb736_asc2','user=paco0228','format=Cluster,Account,User,Partition,MaxJobs,QOS'])
 ar=[l.split('|') for l in assoc.splitlines() if l.strip()]
 need(any(len(r)>=6 and r[:3]==['alpine','ucb736_asc2','paco0228'] and r[3] in ('','acpu') and r[4]!='0'
  and 'cpu-normal' in r[5].split(',') for r in ar),'account/QOS unavailable')
 upstream()
 outer.read(OLD['manifest']['path'],OLD['manifest']['sha256'])
 archive=Path(OLD['archive']['path'])
 need(archive.is_file() and not archive.is_symlink() and archive.stat().st_size==8438650880,'runtime archive metadata differs')
 need(root.parent.is_dir() and not root.parent.is_symlink() and root.parent.stat().st_uid==os.getuid(),'scratch owner differs')
 fs=os.statvfs(root.parent);need(fs.f_bavail*fs.f_frsize>2*1024**3,'insufficient scratch space')
 final_queue=queue();need(time.monotonic()-started<120,'readiness stale')
 return dict(utc=now(),queue=q,queue_before_mutation=final_queue,prior_accounting=rows,prior_journals=journals,
  closed_cpu_seconds=cpu,closed_gpu_slice_seconds=gpu,gpu01_details=details,partition=partition,association=assoc,
  scratch_free_bytes=fs.f_bavail*fs.f_frsize,bulk_archive_byte_authentication='allocated stage before runtime executes')
def verify_packet():
 for n,h in PINS.items():outer.read(root/n,h)
 outer.checked_inventory(api,root/'inputs',META['input_files'])
 outer.authenticate(str(root/'inputs/packet.json'),META['finalizer_packet_sha256'],PACKET['paths']['output'])
'''

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--review',required=True);p.add_argument('--review-sha256',required=True);p.add_argument('--commit',default='HEAD')
 a=p.parse_args();unclaimed()
 revision=subprocess.check_output(['git','rev-parse',a.commit+'^{commit}'],cwd=REPO,text=True).strip()
 meta=json.loads((LOCAL/'packet-metadata.json').read_bytes())
 need(meta['remote_root']==ROOT and meta['input_directory']==ROOT+'/inputs' and meta['output']==ROOT+'/finalization'
      and meta['resources']==RESOURCES,'fixed finalizer identity differs')
 files=['inputs/'+n for n in meta['input_files']]+list(meta['root_files'])+['packet-metadata.json']
 need(len(files)==len(set(files)) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in files),'unsafe packet names')
 pins={}
 for n in files:
  b=(LOCAL/n).read_bytes()
  need(not (LOCAL/n).is_symlink() and not (EXT/n).is_symlink() and b==(EXT/n).read_bytes(),'local/external bytes differ')
  need(subprocess.check_output(['git','show',revision+':'+str((LOCAL/n).relative_to(REPO))],cwd=REPO)==b,'commit does not bind '+n)
  pins[n]=sha(b)
 for n,h in meta['input_files'].items():need(pins['inputs/'+n]==h,'input pin differs')
 for n,h in meta['root_files'].items():need(pins[n]==h,'root pin differs')
 need(Path(__file__).read_bytes()==(LOCAL/'launch_once.py').read_bytes(),'running launcher not committed')
 review=Path(a.review).resolve(strict=True);review_raw=review.read_bytes()
 need(review!=LOCAL/'finalizer-review.md' and bool(review_raw.strip()) and sha(review_raw)==a.review_sha256,'new independent concrete packet review required')
 need(meta['finalizer_packet_sha256'] in review_raw.decode(),'concrete review does not name this finalizer packet')
 need(subprocess.check_output(['git','show',revision+':'+str(review.relative_to(REPO))],cwd=REPO)==review_raw,'concrete review not committed')
 packet=json.loads((LOCAL/'inputs/packet.json').read_bytes())
 need(pins['inputs/packet.json']==meta['finalizer_packet_sha256'] and packet['resources']==RESOURCES,'finalizer packet differs')
 for name in ('cpu-packet','gpu-packet','cpu-acceptance','gpu-acceptance'):
  key=name.replace('-','_')+'_sha256';need(pins['inputs/'+name+'.json']==meta[key]==packet[key],'upstream pin differs')
 for name in ('gpu-disposition','gpu-review-receipt'):
  need(pins[name+'.json']==meta[name.replace('-','_')+'_sha256'],'independent receipt pin differs')
 old=json.loads((LOCAL/'inputs/cpu-packet.json').read_bytes());gp=json.loads((LOCAL/'inputs/gpu-packet.json').read_bytes())
 disposition=json.loads((LOCAL/'gpu-disposition.json').read_bytes())
 evidence={n:(LOCAL/'inputs'/n).read_bytes() for n in ('cpu-acceptance.json','gpu-acceptance.json')}
 intent=dict(utc=utc(),packet_commit=revision,source_commit=meta['source_commit'],pins=pins,
  finalizer_packet_sha256=meta['finalizer_packet_sha256'],gpu_acceptance_sha256=meta['gpu_acceptance_sha256'],
  gpu_disposition_sha256=meta['gpu_disposition_sha256'],gpu_review_receipt_sha256=meta['gpu_review_receipt_sha256'],
  review_path=str(review.relative_to(REPO)),review_sha256=sha(review_raw),review_text=review_raw.decode(),
  launcher_sha256=sha(Path(__file__).read_bytes()),remote_root=ROOT,output=meta['output'],controller_pid_local=os.getpid(),
  resources=RESOURCES,admission=meta['admission'],blind_retry_forbidden=True)
 values=dict(ROOT=ROOT,GPUROOT=GPUROOT,CPUROOT=CPUROOT,ENTRY=ENTRY,SOURCEPIN=SOURCEPIN,META=meta,PINS=pins,
  PACKET=packet,OLD=old,GP=gp,DISPOSITION=disposition,EVIDENCE=evidence,
  OUTER_SOURCE=(LOCAL/'inputs'/ENTRY).read_text(),GPU_SOURCE=(LOCAL/'inputs/qualify_llm_intervention_campaign_gpu.py').read_text())
 common='\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+REMOTE_COMMON
 command=shlex.join([old['runtime_source']+'/bin/python','-I','-B','-'])
 def remote(name,source):
  r=subprocess.run(['bash',HELPER,'run',command],input=source,capture_output=True,text=True,timeout=180)
  save(name,dict(python_source=source,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,utc=utc()))
  need(r.returncode==0,name+' failed; retain/reconcile, never retry');return json.loads(r.stdout)
 locks=[];claimed=False
 try:
  for n in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
   h=open(EXT.parent/n,'r+')
   try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BaseException:h.close();raise
   locks.append(h)
  unclaimed();save('controller.json',dict(held=[h.name for h in locks],pid=os.getpid(),utc=utc()));claimed=True
  ready=remote('readiness_transport.json',common+'''
need(not root.exists() and not root.is_symlink(),'launch root already exists; no retry')
r=reconcile()
root.mkdir(mode=0o700);(root/'logs').mkdir();(root/'inputs').mkdir()
p=root/'write-read-probe'
with p.open('xb') as h:h.write(b'ACE final01 readiness');h.flush();os.fsync(h.fileno())
need(p.read_bytes()==b'ACE final01 readiness','scratch readback failed');p.unlink()
r['scratch_write_read_cleanup']=True;write('readiness.json',r);print(json.dumps(r))
''')
  save('readiness.json',ready)
  for group in ('inputs','root'):
   names=[n for n in files if n.startswith('inputs/')==(group=='inputs')]
   target=ROOT+('/inputs/' if group=='inputs' else '/')
   r=subprocess.run(['scp','-o','ControlPath=/tmp/defab-curc.sock','-o','BatchMode=yes',*[str(LOCAL/n) for n in names],
    'paco0228@login.rc.colorado.edu:'+target],capture_output=True,text=True,timeout=180)
   save('transfer-'+group+'.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,utc=utc()))
   need(r.returncode==0,'partial transfer retained; no retry')
  verified=remote('verification_transport.json',common+'\nverify_packet()\nprint(json.dumps(dict(utc=now(),verified=PINS)))\n')
  save('remote_verified.json',verified);save('submission.intent.json',intent)
  result=remote('submission_transport.json',common+'\nintent='+repr(intent)+'''
need(not (root/'submission.intent.json').exists() and not (root/'submission.result.json').exists(),'existing submission journal')
verify_packet();intent.update(reconcile())
for n,h in PINS.items():outer.read(root/n,h)
queue();intent['remote_utc']=now();write('submission.intent.json',intent)
r=dict(utc=now(),intent_sha256=sha((root/'submission.intent.json').read_bytes()),job_id=None,finalization_qualified=False)
try:
 env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_')}
 p=subprocess.run(['sbatch','--parsable',str(root/'job.sbatch')],env=env,capture_output=True,text=True,timeout=25)
 job=p.stdout.strip().split(';')[0]
 r.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,job_id=job if p.returncode==0 and job.isdigit() else None)
except BaseException as exc:r.update(ambiguous=True,exception_type=type(exc).__name__,message=str(exc))
write('submission.result.json',r);print(json.dumps(r))
''')
  save('submission.result.json',result);need(result['job_id'] is not None,'failed/ambiguous submission; never retry')
  print(json.dumps(result,indent=2))
 except BaseException as exc:
  if claimed:save('launcher_failure.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),blind_retry_forbidden=True))
  raise
 finally:
  for h in reversed(locks):fcntl.flock(h,fcntl.LOCK_UN);h.close()
  if claimed:save('controller_released.json',dict(pid=os.getpid(),utc=utc(),released=True))

if __name__=='__main__':main()

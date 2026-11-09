"""Once-only scientific GPU campaign launcher; accepted CPU campaign, no registry mutation.
Requires accepted actual CPU preparation, full actual baseline and reviewed packet.
"""
from pathlib import Path
import argparse, fcntl, os, subprocess, sys
sys.dont_write_bytecode=True
import build_packet as b
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('packet-commit','packet-sha256','packet-review','packet-review-sha256'):
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--packet-directory',required=True)
    a=ap.parse_args();locks=[];started=False
    def save(name,value):
        for dest in (b.EXT,b.LOCAL):b.write(dest/name,b.raw(value))
    def transport(name,command,*,source=None,timeout):
        try:
            p=subprocess.run(command,input=source,capture_output=True,text=True,timeout=timeout)
            result=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,utc=b.utc())
        except subprocess.TimeoutExpired as exc:
            txt=lambda value:value.decode(errors='replace') if isinstance(value,bytes) else value
            result=dict(command=command,returncode=None,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr),
                        utc=b.utc(),ambiguous=True,retry_allowed=False)
        if source is not None:result['python_source']=source
        save(name,result);b.need(result['returncode']==0,'retained failed/ambiguous transport; no retry: '+name)
        return result
    def remote(name,source):return b.decode(transport(name,['bash',HELPER,'run','python3 -'],source=source,timeout=1230)['stdout'])
    try:
        for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
            path=b.BASE/name;b.need(path==path.resolve(),'coordinator lock alias')
            f=os.fdopen(os.open(path,os.O_RDWR|os.O_NOFOLLOW),'r+')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:f.close();raise
            locks.append(f)
        local=Path(a.packet_directory).resolve(strict=True)
        b.need(Path(__file__).resolve()==local/'launch_once.py','use committed concrete launcher only')
        packet_raw=b.read(local/'packet.json',a.packet_sha256);packet=b.decode(packet_raw)
        b.need(packet['schema']==b.OUTER_VERSION
            and packet['resources']==b.RESOURCES and packet['limits']==b.LIMITS,
            'exact frozen scientific campaign packet required')
        pins={**packet['new_input_files'],'packet.json':a.packet_sha256}
        b.need(all(type(n) is str and not Path(n).is_absolute() and '..' not in Path(n).parts and b.pin(h)
            for n,h in pins.items()),'safe pinned packet membership required')
        supplied={name:b.read(local/(b.metadata_name(name)),
            pins[b.metadata_name(name)],b.metadata_maximum(name)) for name in b.METADATA}
        ctx=b.authenticate(supplied);m=ctx['metadata']
        b.need(local==b.LOCAL and packet['source_commit']==b.SOURCE_COMMIT and packet['remote_root']==b.ROOT,
            'amended frozen packet roots/source differ')
        b.need(not any(os.path.lexists(dest/name) for dest in (b.LOCAL,b.EXT) for name in
            ('controller.json','launcher_failure.json','build-failure.json','submission.intent.json','submission.result.json','storage-use.intent.json')),
            'existing build/controller/submission attempt; no retry')
        for name,h in pins.items():
            value=b.read(b.LOCAL/name,h,256*1024**2)
            b.need(value==b.read(b.EXT/name,h,256*1024**2)==b.committed(a.packet_commit,str((b.LOCAL/name).relative_to(b.REPO))),
                   'committed two-copy packet differs: '+name)
        review=Path(a.packet_review).absolute();review_raw=b.read(review,a.packet_review_sha256)
        b.need(review_raw.strip() and review_raw==b.committed(a.packet_commit,str(review.relative_to(b.REPO))),
               'committed independently pinned packet review required')
        started=True
        b.start_storage_operations(ctx,b.EXT.with_name(b.EXT.name+'-storage-launch-operations'),(b.EXT,b.LOCAL),
            dict(phase='launch',packet_sha256=a.packet_sha256,packet_commit=a.packet_commit))
        b.configure_packet_storage(ctx,local,pins,packet)
        cpu_data=b.admit_cpu_bytes(ctx,Path(packet['cpu_remote_custody']))
        b.need(packet['baseline_sha256']==ctx['pins']['gpu-failure-baseline'] and packet['cpu_job_id']==m['cpu-acceptance']['job_id']
               and packet['operational_amendment_sha256']==ctx['pins']['operational-amendment']
               and packet['namespace_claim_sha256']==ctx['pins']['namespace-claim'],'accepted baseline/namespace differs')
        b.need(packet['token_recovery']==ctx['token_recovery'] and packet['failed_token_root']==b.FAILED_TOKEN_ROOT
               and packet['failed_token_job_id']==b.FAILED_TOKEN_JOB and packet['history_directories']==ctx['history_directories'],
               'complete original TOKEN02 recovery history differs')
        b.need(packet['gpu_recovery']==ctx['gpu_recovery'] and packet['prior_jobs']==47 and packet['prior_journals']==96
               and packet['operational_source_commit']==m['operational-source-manifest']['source_commit'], 'exact independent GPU02 authority/history differs')
        claim_pin=ctx['pins']['namespace-claim']
        b.need(b.read(b.LOCAL_REGISTRY,claim_pin)==supplied['namespace-claim'],'existing local claim changed; never recreate')
        b.read(b.ORIGINAL_LOCAL_REGISTRY,b.ORIGINAL_CLAIM_SHA)
        stage=b.decode(b.read(b.LOCAL/'inputs/packet.json',pins['inputs/packet.json']))
        expected,data,_=b.make_packet(ctx,cpu_data,supplied)
        for name,value in data.items():
            b.need(b.read(local/'inputs'/name,stage['files'][name],256*1024**2)==value,'reconstructed exact source/input bytes differ')
        b.need(packet['gpu_packet_sha256']==b.sha(b.raw(expected)), 'outer/inner GPU packet join differs')
        b.need(b.raw(stage)==b.raw(expected),'captured campaign input/resource contract differs')
        b.need(b.read(b.LOCAL/'inputs/admission.json',pins['inputs/admission.json'])==b.raw(b.admission(ctx,stage)), 'stage admission differs')
        b.need(packet['original_cpu_inputs']==ctx['upstream_pins'],'complete original CPU input pins differ')
        context=dict(root=b.ROOT,registry=b.REMOTE_REGISTRY,pins=pins,baseline=m['gpu-failure-baseline'],
            failed_gpu_root=b.OLDROOT,gpu_output=b.OUTPUT,failed_gpu_inventory=ctx['failed_gpu_inventory'],gpu_recovery=ctx['gpu_recovery'],
            cpu_directories=sorted(n[7:] for n in ctx['observations']['cpu']['directories'] if n.startswith('launch/')),
            archive=m['cpu-packet']['archive'],manifest=m['cpu-packet']['manifest'],namespace_claim_sha256=claim_pin,
            cpu_inputs=ctx['upstream_pins'],history_directories=ctx['history_directories'],model_directory=m['cpu-packet']['model_directory'],model_files=m['cpu-packet']['model_files'],
            original_registry=b.ORIGINAL_REMOTE_REGISTRY,original_namespace_claim_sha256=b.ORIGINAL_CLAIM_SHA,
            cpu_root=b.CPU_ROOT,cpu_roster={n.removeprefix('launch/'):v['sha256'] for n,v in ctx['observations']['cpu']['files'].items() if n.startswith('launch/')})
        common='C='+repr(context)+'\n'+REMOTE_COMMON
        save('storage-launch-use.json',dict(binding=b.storage_packet_binding(ctx),operation_reference=ctx['storage_operation_evidence'].last_reference,acceptance=False))
        save('controller.json',dict(pid=os.getpid(),utc=b.utc(),held=[str(b.BASE/n) for n in
            ('ace-delivery-controller.lock','ace-foundation-controller.lock')]));started=True
        ready=remote('readiness_transport.json',common+r'''
need(root.parent==root.parent.resolve() and root.parent.stat().st_uid==os.getuid(),'scratch parent differs')
need(not os.path.lexists(root),'new control root already exists; no adoption/retry')
need(set(root.parent.glob('ace_llm_intervention_scientific_campaign_gpu*'))=={pathlib.Path(C['failed_gpu_root'])},
     'only exact independently accepted failed GPU01 sibling may exist')
r=reconcile()
root.mkdir(mode=0o700);(root/'logs').mkdir(mode=0o700);(root/'inputs').mkdir(mode=0o700)
probe=root/'write-read-probe'
with probe.open('xb') as f:f.write(b'ACE scientific GPU campaign readiness\n');f.flush();os.fsync(f.fileno())
need(read(probe)==b'ACE scientific GPU campaign readiness\n','scratch readback differs');probe.unlink()
fs=os.statvfs(root);need(fs.f_bavail*fs.f_frsize>=4*1024**3,'scratch output headroom insufficient')
r['scratch']=dict(uid=root.stat().st_uid,free_bytes=fs.f_bavail*fs.f_frsize,write_read_cleanup=True,df=run(['df','-h',str(root)]))
write(root/'readiness.json',r);print(json.dumps(r))
''')
        save('readiness.json',ready)
        for group,subdir in (('root',''),('inputs','inputs/')):
            names=[n for n in pins if n.startswith('inputs/')==(group=='inputs')]
            transport('transfer-'+group+'.json',['scp','-o','ControlPath=/tmp/defab-curc.sock','-o','BatchMode=yes',
                *[str(b.LOCAL/n) for n in names],'paco0228@login.rc.colorado.edu:'+b.ROOT+'/'+subdir],timeout=45)
        verified=remote('verification_transport.json',common+'\nprint(json.dumps(verify()))\n');save('remote_verified.json',verified)
        for name,h in pins.items():b.need(b.read(b.LOCAL/name,h,256*1024**2)==b.read(b.EXT/name,h,256*1024**2),'local pinned input changed')
        b.need(b.read(b.LOCAL_REGISTRY,claim_pin)==supplied['namespace-claim'],'existing local namespace changed')
        b.read(b.ORIGINAL_LOCAL_REGISTRY,b.ORIGINAL_CLAIM_SHA)
        intent=dict(utc=b.utc(),packet_commit=a.packet_commit,source_commit=b.SOURCE_COMMIT,storage_binding=b.storage_packet_binding(ctx),
            packet_sha256=a.packet_sha256,review_sha256=a.packet_review_sha256,pins=pins,remote_root=b.ROOT,
            token_recovery=ctx['token_recovery'],failed_token_root=b.FAILED_TOKEN_ROOT,failed_token_job_id=b.FAILED_TOKEN_JOB,
            baseline_sha256=ctx['pins']['gpu-failure-baseline'],gpu_recovery=ctx['gpu_recovery'],
            operational_source_commit=m['operational-source-manifest']['source_commit'],cpu_acceptance_sha256=ctx['pins']['cpu-acceptance'],
            cpu_job_id=m['cpu-acceptance']['job_id'],existing_namespace_claim_sha256=claim_pin,
            namespace_admission_sha256=ctx['pins']['namespace-admission'],
            recovery_authority_sha256=ctx['pins']['recovery-authority'],original_namespace_claim_sha256=b.ORIGINAL_CLAIM_SHA,
            resources=b.RESOURCES,limits=b.LIMITS,admission=packet['admission'],
            blind_retry_forbidden=True,new_namespace_claimed=False,replacement_worlds=False)
        save('submission.intent.json',intent)
        result=remote('submission_transport.json',common+'\nintent='+repr(intent)+
            '\nREADINESS_SHA='+repr(b.sha(b.raw(ready)))+r'''
need(not os.path.lexists(root/'submission.intent.json') and not os.path.lexists(root/'submission.result.json')
     and not os.path.lexists(root/'gpu'),'existing GPU campaign attempt')
need(sha(read(root/'readiness.json'))==READINESS_SHA,'original readiness differs')
fresh=reconcile();verify();failed_gpu_unchanged()
intent.update(remote_utc=utc(),fresh_readiness=fresh)
write(root/'submission.intent.json',intent)
r=dict(utc=utc(),job_id=None,intent_sha256=sha(read(root/'submission.intent.json')))
try:
 p=subprocess.run(['sbatch','--parsable',str(root/'job.sbatch')],capture_output=True,text=True,timeout=25)
 value=p.stdout.strip();valid=re.fullmatch(r'[1-9][0-9]*(;alpine)?',value) is not None
 r.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
     job_id=value.split(';')[0] if p.returncode==0 and valid else None,ambiguous=not(p.returncode==0 and valid))
except BaseException as exc:
 txt=lambda value:value.decode(errors='replace') if isinstance(value,bytes) else value
 r.update(ambiguous=True,type=type(exc).__name__,message=str(exc),
     stdout=txt(getattr(exc,'stdout',None)),stderr=txt(getattr(exc,'stderr',None)))
write(root/'submission.result.json',r);print(json.dumps(r))
''')
        save('submission.result.json',result)
        b.need(b.read(b.LOCAL_REGISTRY,claim_pin)==supplied['namespace-claim'],'existing recovery namespace changed; no retry')
        b.read(b.ORIGINAL_LOCAL_REGISTRY,b.ORIGINAL_CLAIM_SHA)
        b.need(result['job_id'] is not None,'ambiguous/failed submit retained; never retry')
        print(b.raw(result).decode())
    except BaseException as exc:
        if started:
            try:save('launcher_failure.json',dict(utc=b.utc(),type=type(exc).__name__,message=str(exc),
                     retry_allowed=False,preserve_namespace_and_all_intents=True))
            except BaseException:pass
        raise
    finally:
        for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
        if started:
            try:save('controller_released.json',dict(utc=b.utc(),pid=os.getpid(),released=True))
            except BaseException:pass

REMOTE_COMMON=r'''
import pathlib,json,hashlib,subprocess,datetime,os,stat,re,signal,sys,time
os.umask(0o077)
def remote_deadline(signum,frame):raise TimeoutError('absolute remote coordinator deadline')
signal.signal(signal.SIGALRM,remote_deadline);signal.alarm(1200)
_READ_COUNT=0;_READ_STARTED=time.monotonic()
root=pathlib.Path(C['root']); registry=pathlib.Path(C['registry'])
def need(ok,message):
 if not ok:raise RuntimeError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def raw(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def read(path,allow_hardlink=False):
 global _READ_COUNT
 path=pathlib.Path(path)
 need(path.absolute()==path.resolve(),'unresolved/symlink path '+str(path))
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  s=os.fstat(f.fileno());need(stat.S_ISREG(s.st_mode) and (allow_hardlink or s.st_nlink==1),'regular sole-linked input required')
  value=f.read(256*1024**2+1);after=os.fstat(f.fileno())
 identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_nlink)
 need(identity(s)==identity(after)==identity(path.lstat()),'input changed during capture')
 need(len(value)<=256*1024**2,'oversize metadata')
 _READ_COUNT+=1
 if _READ_COUNT%1000==0:print(json.dumps(dict(phase='upstream_input_verification',files_read=_READ_COUNT,elapsed_seconds=time.monotonic()-_READ_STARTED)),file=sys.stderr,flush=True)
 return value
def write(path,value):
 with pathlib.Path(path).open('xb') as f:f.write(raw(value));f.flush();os.fsync(f.fileno())
def run(command):
 try:p=subprocess.run(command,capture_output=True,text=True,timeout=20)
 except subprocess.TimeoutExpired as exc:
  txt=lambda value:value.decode(errors='replace') if isinstance(value,bytes) else value
  raise RuntimeError(json.dumps(dict(command=command,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr)))) from exc
 need(p.returncode==0,json.dumps(dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
 return p.stdout
def normalized(rows):
 result=[]
 for row in rows:
  need(len(row)>=8 and not any(row[8:]) and row[0].isdigit() and row[4].isdigit() and row[5].isdigit(),
       'malformed accounting row')
  pairs=[v.split('=',1) for v in row[7].split(',') if v]
  need(all(len(v)==2 for v in pairs) and len({v[0] for v in pairs})==len(pairs),'malformed allocation')
  result.append([row[0],row[1].strip(),row[2].strip(),row[3].strip(),int(row[4]),int(row[5]),row[6].strip(),sorted(pairs)])
 need(len({r[0] for r in result})==len(result),'duplicate job')
 return sorted(result,key=lambda r:int(r[0]))
def cost(rows):
    cpu=gpu=0
    for r in normalized(rows):
        t=dict(r[7]);need(t or r[4]==r[5]==0,'missing actual TRES; unknown cost')
        g={k:v for k,v in t.items() if k.startswith('gres/')}
        need(all(v.isdigit() and (k=='gres/gpu' or k.startswith('gres/gpu:')) for k,v in g.items()),'unknown accelerator charge')
        typed=sum(int(v) for k,v in g.items() if k.startswith('gres/gpu:'))
        n=int(g['gres/gpu']) if 'gres/gpu' in g else typed
        need(not ('gres/gpu' in g and any(k.startswith('gres/gpu:') for k in g)) or n==typed,'inconsistent GPU charge')
        cpu+=r[4]*r[5];gpu+=n*r[5]
    return cpu,gpu

def duration(s):
 if s in ('UNLIMITED','INFINITE'):return float('inf')
 days=0
 if '-' in s:
  d,s=s.split('-',1);days=int(d)
 fields=s.split(':');need(len(fields)==3 and all(v.isdigit() for v in fields),'unreadable live wall limit')
 h,m,sec=map(int,fields);return days*86400+h*3600+m*60+sec
def allows(value,name):return value=='ALL' or name in value.split(',')
def failed_gpu_unchanged():
 old=pathlib.Path(C['failed_gpu_root']);output=pathlib.Path(C['gpu_output'])
 need(old!=root and output==old/'gpu' and old.is_dir() and old==old.resolve(),'exact original failed GPU root required')
 need(not os.path.lexists(output),'frozen output exists, including alias/partial; never retry')
 need(set(root.parent.glob('ace_llm_intervention_scientific_campaign_gpu*'))<={old,root},'unexpected GPU sibling')
 entries=list(old.rglob('*'));need(len(entries)<=262144 and not any(p.is_symlink() for p in entries),'ordinary bounded failed root required')
 want=C['failed_gpu_inventory']
 need(sorted(p.relative_to(old).as_posix() for p in entries if p.is_dir())==want['directories'],'old failed directories changed')
 actual={}
 for p in entries:
  if p.is_dir():continue
  value=read(p,allow_hardlink=True);actual[p.relative_to(old).as_posix()]=dict(bytes=len(value),sha256=sha(value))
 need(actual==want['files'],'old failed bytes/membership changed')
 return True

def reconcile():
 failed_gpu_unchanged()
 queue=run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'])
 for line in queue.splitlines():
  fields=line.split('|');need(len(fields)==5,'malformed queue row')
  need(not fields[1].lower().startswith('ace'),'active ACE job; leave it untouched')
 accounting=run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format',
  'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'])
 rows=[line.split('|') for line in accounting.splitlines() if '|ace_llm' in line]
 need(len(rows)==C['baseline']['terminal_jobs'] and normalized(rows)==normalized(C['baseline']['related_accounting']),
      'exact historical accounting changed')
 need(all(r[2].split()[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for r in rows),
      'nonterminal historical job')
 need(cost(rows)==(C['baseline']['closed_cpu_seconds'],C['baseline']['closed_gpu_slice_seconds']),'closed costs changed')
 need(C['baseline']['closed_cpu_seconds']+17520<=57600 and C['baseline']['closed_gpu_slice_seconds']+2880<=7200,'full remaining GPU/finalizer caps exceed budget')
 journals={str(p):sha(read(p)) for d in root.parent.glob('ace_llm_intervention*')
           for p in d.glob('submission.*.json')}
 need(len(journals)==C['baseline']['submission_journals'] and journals==C['baseline']['journal_pins'],
      'exact full prior journal inventory changed')
 partition=run(['scontrol','show','partition','aa100','-o'])
 fields=dict(x.split('=',1) for x in partition.split() if '=' in x)
 need(fields.get('PartitionName')=='aa100' and fields.get('State')=='UP' and
      fields.get('MaxMemPerCPU')=='3840' and duration(fields.get('MaxTime',''))>=2880,
      '4CPU/15360M/2880s partition feasibility changed')
 need(allows(fields.get('AllowAccounts',''),'ucb736_asc2') and allows(fields.get('AllowQos',''),'gpu-testing')
      and 'ucb736_asc2' not in fields.get('DenyAccounts','').split(',')
      and 'gpu-testing' not in fields.get('DenyQos','').split(','),'partition admission changed')
 assoc=run(['sacctmgr','-n','-P','show','assoc','where','account=ucb736_asc2','user=paco0228',
            'format=Cluster,Account,User,Partition,MaxJobs,QOS'])
 ar=[l.split('|') for l in assoc.splitlines() if l.strip()]
 need(len(ar)==1 and ar[0][:3]==['alpine','ucb736_asc2','paco0228'] and ar[0][3] in ('','aa100')
      and ar[0][4]!='0' and 'gpu-testing' in ar[0][5].split(','),'account GPU QoS differs')
 qos=run(['sacctmgr','-n','-P','show','qos','where','name=gpu-testing','format=Name,MaxWall,MaxTRESPU'])
 qr=[l.split('|') for l in qos.splitlines() if l.strip()]
 need(len(qr)==1 and qr[0][0]=='gpu-testing' and (qr[0][1]=='' or duration(qr[0][1])>=2880),
      'live QoS wall insufficient')
 limits=dict(part.split('=',1) for part in qr[0][2].split(',') if '=' in part)
 cap=limits.get('gres/gpu:a100_3g.20gb')
 need(cap is not None and cap.isdigit() and int(cap)>=1,'live QoS MIG cap unavailable/inadequate')
 gpu_types=run(['sinfo','-p','aa100','-h','-o','%G'])
 need('gpu:a100_3g.20gb:' in gpu_types,'required20GB MIG type unavailable')
 manifest=pathlib.Path(C['manifest']['path']);archive=pathlib.Path(C['archive']['path'])
 need(sha(read(manifest))==C['manifest']['sha256'],'accepted runtime manifest changed')
 st=archive.lstat()
 need(archive==archive.resolve() and stat.S_ISREG(st.st_mode) and st.st_size==8438650880,
      'accepted archive metadata changed')
 need(sha(read(registry))==C['namespace_claim_sha256'],'existing accepted namespace claim changed')
 need(sha(read(C['original_registry']))==C['original_namespace_claim_sha256'],'preserved original namespace changed')
 cpu_root=pathlib.Path(C['cpu_root'])
 need(cpu_root.is_dir() and cpu_root==cpu_root.resolve(),'original CPU root alias/missing')
 members=list(cpu_root.rglob('*'))
 need(len(members)<=262144 and not any(p.is_symlink() for p in members),'bounded ordinary CPU custody required')
 need({str(p.relative_to(cpu_root)) for p in members if p.is_file()}==set(C['cpu_roster']),'complete original CPU custody roster changed')
 need(sorted(str(p.relative_to(cpu_root)) for p in members if p.is_dir())==C['cpu_directories'],'complete original CPU directories changed')
 for base,expected_dirs in C['history_directories'].items():
  tree=pathlib.Path(base);need(tree.is_dir() and not tree.is_symlink(),'original upstream root missing')
  entries=list(tree.rglob('*'));need(not any(p.is_symlink() for p in entries),'upstream symlink')
  need(sorted(p.relative_to(tree).as_posix() for p in entries if p.is_dir())==expected_dirs,'whole original upstream directories changed')
  expected={n[len(base)+1:]:h for n,h in C['cpu_inputs'].items() if n.startswith(base+'/')}
  need({p.relative_to(tree).as_posix():sha(read(p,allow_hardlink=True)) for p in entries if p.is_file()}==expected,'whole original upstream files changed')
 for path,h in C['cpu_inputs'].items():
  need(sha(read(path,allow_hardlink=True))==h,'accepted CPU input changed: '+path)
 model=pathlib.Path(C['model_directory'])
 need(model==model.resolve() and model.is_dir(),'frozen model directory alias')
 model_verified={}
 for name,expected in C['model_files'].items():
  path=model/name
  need(path.name==name and not path.is_symlink(),'ordinary exact model member required')
  h=hashlib.sha256()
  with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
   before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode),'regular model bytes required')
   for block in iter(lambda:f.read(1024**2),b''):h.update(block)
   after=os.fstat(f.fileno())
  need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
       and h.hexdigest()==expected,'frozen model bytes differ')
  model_verified[name]=expected
 return dict(utc=utc(),gpu_recovery=C['gpu_recovery'],failed_gpu_original_bytes_verified=True,frozen_output_absent=True,queue=queue,related_accounting=rows,journal_pins=journals,
             partition=partition,association=assoc,qos=qos,gpu_types=gpu_types,model_files=model_verified,manifest_sha256=C['manifest']['sha256'],
             archive_size=st.st_size,archive_byte_check='allocated stage authenticates full bytes before execution',
             existing_namespace_claim_sha256=C['namespace_claim_sha256'],
             original_namespace_claim_sha256=C['original_namespace_claim_sha256'],
             closed_cpu_seconds=C['baseline']['closed_cpu_seconds'],closed_gpu_slice_seconds=C['baseline']['closed_gpu_slice_seconds'],
             cpu_reserved_total=C['baseline']['closed_cpu_seconds']+15120,gpu_reserved_total=C['baseline']['closed_gpu_slice_seconds']+2880)
def verify():
 for name,h in C['pins'].items():need(sha(read(root/name))==h,'remote pin differs: '+name)
 return dict(utc=utc(),verified=C['pins'])
'''


if __name__=='__main__':main()

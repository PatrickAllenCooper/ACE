"""Once-only scientific token preparation launcher; existing namespace is read-only.
Requires accepted actual realization and collection, dynamic independent baseline and reviewed packet.
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
    def remote(name,source):return b.decode(transport(name,['bash',HELPER,'run','python3 -'],source=source,timeout=255)['stdout'])
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
            and packet['resources']==b.RESOURCES and packet['deadlines']==b.DEADLINES,
            'exact frozen token packet required')
        pins={**packet['new_input_files'],'packet.json':a.packet_sha256}
        b.need(all(type(n) is str and not Path(n).is_absolute() and '..' not in Path(n).parts and b.pin(h)
            for n,h in pins.items()),'safe pinned packet membership required')
        supplied={name:b.read(local/(b.metadata_name(name)),
            pins[b.metadata_name(name)]) for name in b.METADATA}
        ctx=b.admit(supplied);m=ctx['metadata']
        b.need(local==b.LOCAL and packet['local_root']==str(b.LOCAL) and packet['external_root']==str(b.EXT)
            and packet['source_commit']==b.SOURCE_COMMIT and packet['remote_root']==b.ROOT,
            'amended frozen packet roots/source differ')
        b.need(not any(os.path.lexists(dest/name) for dest in (b.LOCAL,b.EXT) for name in
            ('controller.json','launcher_failure.json','build-failure.json','submission.intent.json','submission.result.json')),
            'existing build/controller/submission attempt; no retry')
        for name,h in pins.items():
            value=b.read(b.LOCAL/name,h)
            b.need(value==b.read(b.EXT/name,h)==b.committed(a.packet_commit,str((b.LOCAL/name).relative_to(b.REPO))),
                   'committed two-copy packet differs: '+name)
        review=Path(a.packet_review).absolute();review_raw=b.read(review,a.packet_review_sha256)
        b.need(review_raw.strip() and review_raw==b.committed(a.packet_commit,str(review.relative_to(b.REPO))),
               'committed independently pinned packet review required')
        b.need(packet['baseline_sha256']==ctx['pins']['token-failure-baseline']
               and packet['token_recovery']==ctx['token_recovery'] and packet['realization_job_id']==ctx['job']
               and packet['collection_job_id']==ctx['collection_job']
               and packet['collection_acceptance_sha256']==ctx['pins']['collection-acceptance']
               and packet['namespace_claim_sha256']==ctx['pins']['namespace-claim']
               and packet['operational_amendment_sha256']==ctx['pins']['operational-amendment']
               and packet['operational_adoption_review_sha256']==ctx['pins']['operational-adoption-review']
               and packet['collection_failure_acceptance_sha256']==ctx['pins']['collection-failure-acceptance']
               and packet['no_response_proof_sha256']==ctx['pins']['no-response-proof']
               and packet['science_equality']==ctx['science_equality'],'accepted baseline/namespace/amendment differs')
        claim_pin=ctx['pins']['namespace-claim']
        b.need(b.read(b.LOCAL_REGISTRY,claim_pin)==supplied['namespace-claim'],'existing local claim changed; never recreate')
        b.read(b.ORIGINAL_LOCAL_REGISTRY,b.ORIGINAL_CLAIM_SHA)
        stage=b.decode(b.read(b.LOCAL/'inputs/packet.json',pins['inputs/packet.json']))
        b.need(packet['token_science_equality']==b.token_science_equality(m['failed-token-stage'],stage),'exact original child equality receipt differs')
        original={}
        for kind,root in (('realization',b.REALIZATION_ROOT),('collection',b.COLLECTION_ROOT)):
            original.update({root+'/packet.json':ctx['pins'][kind+'-packet'],
                root+'/submission.result.json':ctx['pins'][kind+'-result'],
                root+'/submission.intent.json':m[kind+'-result']['intent_sha256']})
        original[b.REALIZATION_ROOT+'/realization/session/worker/task/terminal.json']=m['realization-acceptance']['realization_terminal_sha256']
        collection=m['collection-acceptance']['collection']
        b.need(stage['collection']==collection and stage['model_directory']==b.ROOT+'/tokenizer'
            and stage['resources']==b.RESOURCES and stage['deadlines']==b.DEADLINES
            and stage['files']['collection-acceptance.json']==ctx['pins']['collection-acceptance'],
            'original collection/tokenizer stage binding differs')
        failed_files={n.removeprefix('launch/'):v['sha256'] for n,v in ctx['failed_files'].items() if n.startswith('launch/')}
        context=dict(root=b.ROOT,failed_root=b.FAILED_COLLECTION_ROOT,failed_files=failed_files,registry=b.REMOTE_REGISTRY,pins=pins,baseline=m['token-failure-baseline'],failed_token_root=b.FAILED_TOKEN_ROOT,
            prior_roots=ctx['failed_token']['preserved_roots'],prior_directories=ctx['failed_token']['preserved_directories'],
            archive=stage['archive'],manifest=stage['manifest'],namespace_claim_sha256=claim_pin,upstream_inputs=original,
            collection=collection,model_source=m['freeze']['model_directory'],tokenizer_files=stage['tokenizer_files'],
            original_registry=b.ORIGINAL_REMOTE_REGISTRY,original_namespace_claim_sha256=b.ORIGINAL_CLAIM_SHA)
        common='C='+repr(context)+'\n'+REMOTE_COMMON
        save('controller.json',dict(pid=os.getpid(),utc=b.utc(),held=[str(b.BASE/n) for n in
            ('ace-delivery-controller.lock','ace-foundation-controller.lock')]));started=True
        ready=remote('readiness_transport.json',common+r'''
need(root.parent==root.parent.resolve() and root.parent.stat().st_uid==os.getuid(),'scratch parent differs')
need(not os.path.lexists(root) and {str(p) for p in root.parent.glob('ace_llm_intervention_scientific_tokenization*')}=={C['failed_token_root']},
     'only exact preserved TOKEN01 may preexist; no adoption/retry')
r=reconcile()
root.mkdir(mode=0o700);(root/'logs').mkdir(mode=0o700);(root/'inputs').mkdir(mode=0o700);(root/'tokenizer').mkdir(mode=0o700)
probe=root/'write-read-probe'
with probe.open('xb') as f:f.write(b'ACE token preparation readiness\n');f.flush();os.fsync(f.fileno())
need(read(probe)==b'ACE token preparation readiness\n','scratch readback differs');probe.unlink()
fs=os.statvfs(root);need(fs.f_bavail*fs.f_frsize>=2*1024**3,'scratch output headroom insufficient')
r['scratch']=dict(uid=root.stat().st_uid,free_bytes=fs.f_bavail*fs.f_frsize,write_read_cleanup=True,df=run(['df','-h',str(root)]))
write(root/'readiness.json',r);print(json.dumps(r))
''')
        save('readiness.json',ready)
        for group,subdir in (('root',''),('inputs','inputs/'),('tokenizer','tokenizer/')):
            names=[n for n in pins if (n.split('/',1)[0] if '/' in n else 'root')==group]
            transport('transfer-'+group+'.json',['scp','-o','ControlPath=/tmp/defab-curc.sock','-o','BatchMode=yes',
                *[str(b.LOCAL/n) for n in names],'paco0228@login.rc.colorado.edu:'+b.ROOT+'/'+subdir],timeout=90)
        verified=remote('verification_transport.json',common+'\nprint(json.dumps(verify()))\n');save('remote_verified.json',verified)
        for name,h in pins.items():b.need(b.read(b.LOCAL/name,h)==b.read(b.EXT/name,h),'local pinned input changed')
        b.need(b.read(b.LOCAL_REGISTRY,claim_pin)==supplied['namespace-claim'],'existing local namespace changed')
        b.read(b.ORIGINAL_LOCAL_REGISTRY,b.ORIGINAL_CLAIM_SHA)
        intent=dict(utc=b.utc(),packet_commit=a.packet_commit,source_commit=b.SOURCE_COMMIT,
            packet_sha256=a.packet_sha256,review_sha256=a.packet_review_sha256,pins=pins,remote_root=b.ROOT,
            baseline_sha256=ctx['pins']['token-failure-baseline'],token_recovery=ctx['token_recovery'],
            token_science_equality=packet['token_science_equality'],realization_acceptance_sha256=ctx['pins']['realization-acceptance'],
            realization_job_id=ctx['job'],collection_job_id=ctx['collection_job'],
            collection_acceptance_sha256=ctx['pins']['collection-acceptance'],existing_namespace_claim_sha256=claim_pin,
            operational_amendment_sha256=ctx['pins']['operational-amendment'],
            operational_adoption_review_sha256=ctx['pins']['operational-adoption-review'],
            collection_failure_acceptance_sha256=ctx['pins']['collection-failure-acceptance'],
            no_response_proof_sha256=ctx['pins']['no-response-proof'],
            namespace_admission_sha256=ctx['pins']['namespace-admission'],
            recovery_authority_sha256=ctx['pins']['recovery-authority'],original_namespace_claim_sha256=b.ORIGINAL_CLAIM_SHA,
            resources=b.RESOURCES,deadlines=b.DEADLINES,admission=packet['admission'],
            blind_retry_forbidden=True,new_namespace_claimed=False,replacement_worlds=False)
        save('submission.intent.json',intent)
        result=remote('submission_transport.json',common+'\nintent='+repr(intent)+
            '\nREADINESS_SHA='+repr(b.sha(b.raw(ready)))+r'''
need(not os.path.lexists(root/'submission.intent.json') and not os.path.lexists(root/'submission.result.json')
     and not os.path.lexists(root/'preparation'),'existing token attempt')
need(sha(read(root/'readiness.json'))==READINESS_SHA,'original readiness differs')
fresh=reconcile();verify()
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
import pathlib,json,hashlib,subprocess,datetime,os,stat,re,signal
os.umask(0o077)
def remote_deadline(signum,frame):raise TimeoutError('absolute remote coordinator deadline')
signal.signal(signal.SIGALRM,remote_deadline);signal.alarm(240)
root=pathlib.Path(C['root']); registry=pathlib.Path(C['registry'])
def need(ok,message):
 if not ok:raise RuntimeError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def raw(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def read(path,allow_hardlink=False):
 path=pathlib.Path(path)
 need(path.absolute()==path.resolve(),'unresolved/symlink path '+str(path))
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  s=os.fstat(f.fileno());need(stat.S_ISREG(s.st_mode) and (allow_hardlink or s.st_nlink==1),'regular sole-linked input required')
  value=f.read(32*1024**2+1)
 need(len(value)<=32*1024**2,'oversize metadata')
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
def duration(s):
 if s in ('UNLIMITED','INFINITE'):return float('inf')
 days=0
 if '-' in s:
  d,s=s.split('-',1);days=int(d)
 fields=s.split(':');need(len(fields)==3 and all(v.isdigit() for v in fields),'unreadable live wall limit')
 h,m,sec=map(int,fields);return days*86400+h*3600+m*60+sec
def allows(value,name):return value=='ALL' or name in value.split(',')
def verify_tree(c):
 tree=pathlib.Path(c['root'])
 need(tree.is_dir() and tree==tree.resolve(),'ordinary accepted collection root required')
 files={};directories=[];total=0
 for path in sorted(tree.rglob('*')):
  need(not path.is_symlink(),'collection member symlink')
  if path.is_dir():directories.append(path.relative_to(tree).as_posix());continue
  with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
   before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode) and before.st_size<=64*1024**2,'bounded regular collection member')
   digest=hashlib.sha256()
   for chunk in iter(lambda:f.read(1024**2),b''):digest.update(chunk)
   after=os.fstat(f.fileno())
  identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
  need(identity(before)==identity(after)==identity(path.lstat()),'collection member changed during read')
  files[path.relative_to(tree).as_posix()]=digest.hexdigest();total+=after.st_size
  need(len(files)<=65536 and total<=2*1024**3,'complete collection bound exceeded')
 need(files==c['files'],'complete accepted collection changed; no subset or filtering')
 if 'directories' in c:need(sorted(directories)==sorted(c['directories']),'complete preserved directory roster changed')
 return dict(root=str(tree),files=len(files),bytes=total,verified=True,private_json_decoded=False)
def reconcile():
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
 need(sum(int(r[4])*int(r[5]) for r in rows)==C['baseline']['closed_cpu_seconds'] and
      sum(int(r[5]) for r in rows if 'gres/gpu:' in r[7])==C['baseline']['closed_gpu_slice_seconds'],'closed costs changed')
 journals={str(p):sha(read(p)) for d in root.parent.glob('ace_llm_intervention*')
           for p in d.glob('submission.*.json')}
 need(len(journals)==C['baseline']['submission_journals'] and journals==C['baseline']['journal_pins'],
      'exact full prior journal inventory changed')
 partition=run(['scontrol','show','partition','acpu','-o'])
 fields=dict(x.split('=',1) for x in partition.split() if '=' in x)
 need(fields.get('PartitionName')=='acpu' and fields.get('State')=='UP' and
      fields.get('MaxMemPerCPU')=='3840' and duration(fields.get('MaxTime',''))>=600,
      '3CPU/11520M/600s partition feasibility changed')
 need(allows(fields.get('AllowAccounts',''),'ucb736_asc2') and allows(fields.get('AllowQos',''),'cpu-normal')
      and 'ucb736_asc2' not in fields.get('DenyAccounts','').split(',')
      and 'cpu-normal' not in fields.get('DenyQos','').split(','),'partition admission changed')
 assoc=run(['sacctmgr','-n','-P','show','assoc','where','account=ucb736_asc2','user=paco0228',
            'format=Cluster,Account,User,Partition,MaxJobs,QOS'])
 ar=[l.split('|') for l in assoc.splitlines() if l.strip()]
 need(len(ar)==1 and ar[0][:3]==['alpine','ucb736_asc2','paco0228'] and ar[0][3] in ('','acpu')
      and ar[0][4]!='0' and 'cpu-normal' in ar[0][5].split(','),'account CPU QoS differs')
 qos=run(['sacctmgr','-n','-P','show','qos','where','name=cpu-normal','format=Name,MaxWall'])
 qr=[l.split('|') for l in qos.splitlines() if l.strip()]
 need(len(qr)==1 and qr[0][0]=='cpu-normal' and (qr[0][1]=='' or duration(qr[0][1])>=600),
      'live QoS wall insufficient')
 manifest=pathlib.Path(C['manifest']['path']);archive=pathlib.Path(C['archive']['path'])
 need(sha(read(manifest))==C['manifest']['sha256'],'accepted runtime manifest changed')
 st=archive.lstat()
 need(archive==archive.resolve() and stat.S_ISREG(st.st_mode) and st.st_size==8438650880,
      'accepted archive metadata changed')
 need(sha(read(registry))==C['namespace_claim_sha256'],'existing accepted namespace claim changed')
 need(sha(read(C['original_registry']))==C['original_namespace_claim_sha256'],'preserved original namespace changed')
 for path,h in C['upstream_inputs'].items():
  need(sha(read(path,allow_hardlink=True))==h,'accepted upstream journal/terminal changed: '+path)
 collection_receipt=verify_tree(C['collection'])
 failed_receipt=verify_tree(dict(root=C['failed_root'],files=C['failed_files']))
 prior_receipts={path:verify_tree(dict(root=path,files=files,directories=C['prior_directories'][path])) for path,files in C['prior_roots'].items()}
 for name,h in C['tokenizer_files'].items():
  need(sha(read(pathlib.Path(C['model_source'])/name))==h,'original public tokenizer changed')
 return dict(utc=utc(),queue=queue,related_accounting=rows,journal_pins=journals,
             partition=partition,association=assoc,qos=qos,collection=collection_receipt,failed_collection=failed_receipt,preserved_failed_token_and_prior_roots=prior_receipts,manifest_sha256=C['manifest']['sha256'],
             archive_size=st.st_size,archive_byte_check='allocated stage authenticates full bytes before execution',
             existing_namespace_claim_sha256=C['namespace_claim_sha256'],
             original_namespace_claim_sha256=C['original_namespace_claim_sha256'],
             closed_cpu_seconds=C['baseline']['closed_cpu_seconds'],closed_gpu_slice_seconds=C['baseline']['closed_gpu_slice_seconds'],
             cpu_reserved_total=C['baseline']['closed_cpu_seconds']+5400,gpu_reserved_total=C['baseline']['closed_gpu_slice_seconds'])
def verify():
 need({p.name for p in (root/'tokenizer').iterdir()}==set(C['tokenizer_files']),'exact six-file tokenizer staging required')
 for name,h in C['pins'].items():need(sha(read(root/name))==h,'remote pin differs: '+name)
 return dict(utc=utc(),verified=C['pins'])
'''


if __name__=='__main__':main()

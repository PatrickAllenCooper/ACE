"""Once-only FIRST realization launcher. No registry adoption, restart or retry.

Requires a separately reviewed, committed packet and both coordinator locks.
Never invoke while merely authoring this preparation. No import-time actions.
"""
from pathlib import Path
import argparse
import fcntl
import os
import subprocess
import sys

sys.dont_write_bytecode=True
import build_packet as b

HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'

# Captured remote stdlib controller. Inputs are embedded with repr, never shell interpolation.
REMOTE_COMMON=r'''
import pathlib,json,hashlib,subprocess,datetime,os,stat,re,signal
os.umask(0o077)
def remote_deadline(signum,frame):raise TimeoutError('absolute remote coordinator deadline')
signal.signal(signal.SIGALRM,remote_deadline);signal.alarm(48)
root=pathlib.Path(C['root']); registry=pathlib.Path(C['registry'])
def need(ok,message):
 if not ok:raise RuntimeError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def raw(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def read(path):
 path=pathlib.Path(path)
 need(path.absolute()==path.resolve(),'unresolved/symlink path '+str(path))
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  s=os.fstat(f.fileno());need(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'regular sole-linked input required')
  value=f.read(32*1024**2+1)
 need(len(value)<=32*1024**2,'oversize metadata')
 return value
def write(path,value):
 with pathlib.Path(path).open('xb') as f:f.write(raw(value));f.flush();os.fsync(f.fileno())
def run(command):
 p=subprocess.run(command,capture_output=True,text=True,timeout=20)
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
def reconcile():
 queue=run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'])
 for line in queue.splitlines():
  fields=line.split('|');need(len(fields)==5,'malformed queue row')
  need(not fields[1].lower().startswith('ace'),'active ACE job; leave it untouched')
 accounting=run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format',
  'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'])
 rows=[line.split('|') for line in accounting.splitlines() if '|ace_llm' in line]
 need(len(rows)==38 and normalized(rows)==normalized(C['baseline']['related_accounting']),
      'exact historical accounting changed')
 need(all(r[2].split()[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for r in rows),
      'nonterminal historical job')
 need(sum(int(r[4])*int(r[5]) for r in rows)==11040 and
      sum(int(r[5]) for r in rows if 'gres/gpu:' in r[7])==1601,'closed costs changed')
 journals={str(p):sha(read(p)) for d in root.parent.glob('ace_llm_intervention*')
           for p in d.glob('submission.*.json')}
 need(len(journals)==78 and journals==C['baseline']['journal_pins'],'exact full prior journal inventory changed')
 partition=run(['scontrol','show','partition','acpu','-o'])
 fields=dict(x.split('=',1) for x in partition.split() if '=' in x)
 need(fields.get('PartitionName')=='acpu' and fields.get('State')=='UP' and
      fields.get('MaxMemPerCPU')=='3840' and duration(fields.get('MaxTime',''))>=300,
      '2CPU/7680M/300s partition feasibility changed')
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
 need(len(qr)==1 and qr[0][0]=='cpu-normal' and (qr[0][1]=='' or duration(qr[0][1])>=300),
      'live QoS wall insufficient')
 manifest=pathlib.Path(C['manifest']['path']);archive=pathlib.Path(C['archive']['path'])
 need(sha(read(manifest))==C['manifest']['sha256'],'accepted runtime manifest changed')
 st=archive.lstat()
 need(archive==archive.resolve() and stat.S_ISREG(st.st_mode) and st.st_size==8438650880,
      'accepted archive metadata changed')
 need(not os.path.lexists(registry),'existing scientific namespace registry; never adopt')
 return dict(utc=utc(),queue=queue,related_accounting=rows,journal_pins=journals,
             partition=partition,association=assoc,qos=qos,manifest_sha256=C['manifest']['sha256'],
             archive_size=st.st_size,archive_byte_check='allocated stage authenticates full bytes before execution',
             closed_cpu_seconds=11040,closed_gpu_slice_seconds=1601,cpu_reserved_total=15240,gpu_reserved_total=1601)
def verify():
 for name,h in C['pins'].items():need(sha(read(root/name))==h,'remote pin differs: '+name)
 return dict(utc=utc(),verified=C['pins'])
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('packet-commit','packet-sha256','packet-review','packet-review-sha256'):
        ap.add_argument('--'+name,required=True)
    args=ap.parse_args();locks=[];started=False
    def save(name,value):
        data=b.raw(value)
        for root in (b.EXT,b.LOCAL):b.write(root/name,data)
    def remote(name,source):
        try:
            p=subprocess.run(['bash',HELPER,'run','python3 -'],input=source,capture_output=True,text=True,timeout=55)
        except subprocess.TimeoutExpired as exc:
            def txt(v):return v.decode(errors='replace') if isinstance(v,bytes) else v
            save(name,dict(python_source=source,utc=b.utc(),transport_timeout=True,
                 stdout=txt(exc.stdout),stderr=txt(exc.stderr),ambiguous=True,retry_allowed=False))
            raise
        save(name,dict(python_source=source,utc=b.utc(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        b.need(p.returncode==0,name+' failed; retain all state and never retry')
        return b.decode(p.stdout)
    try:
        for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
            path=b.BASE/name
            b.need(path==path.resolve(),'coordinator lock symlink')
            f=os.fdopen(os.open(path,os.O_RDWR|os.O_NOFOLLOW),'r+')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:f.close();raise
            locks.append(f)
        b.need(Path(__file__).resolve()==b.LOCAL/'launch_once.py','run only the committed concrete launcher')
        b.need(not any(os.path.lexists(b.LOCAL/n) or os.path.lexists(b.EXT/n) for n in
               ('controller.json','launcher_failure.json','submission.intent.json','submission.result.json','build-failure.json')),
               'previous controller/build/submission attempt; no retry')
        b.need(not os.path.lexists(b.LOCAL_REGISTRY),'existing local namespace; no adoption')
        packet_raw=b.read(b.LOCAL/'packet.json',args.packet_sha256);packet=b.decode(packet_raw)
        b.need(packet['schema']=='ace-scientific-realization-launch-v1' and packet['remote_root']==b.ROOT
               and packet['resources']==b.RESOURCES and packet['deadlines']==b.DEADLINES
               and packet['baseline_sha256']==b.BASELINE_SHA,'fixed launch contract differs')
        pins={**packet['new_input_files'],'packet.json':args.packet_sha256}
        b.need(all(type(n) is str and '..' not in Path(n).parts and not Path(n).is_absolute()
                   and b.pin(h) for n,h in pins.items()),'bounded relative packet membership')
        for name,h in pins.items():
            data=b.read(b.LOCAL/name,h)
            b.need(b.read(b.EXT/name,h)==data,'external packet mirror differs')
            b.need(b.committed(args.packet_commit,str((b.LOCAL/name).relative_to(b.REPO)))==data,
                   'packet commit differs: '+name)
        review=Path(args.packet_review).absolute();review_raw=b.read(review,args.packet_review_sha256)
        b.need(b.committed(args.packet_commit,str(review.relative_to(b.REPO)))==review_raw and review_raw.strip(),
               'independently pinned committed packet review required')
        baseline_raw=b.read(b.LOCAL/'accepted-accounting.json',b.BASELINE_SHA);baseline=b.decode(baseline_raw)
        b.need(b.read(b.BASELINE,b.BASELINE_SHA)==baseline_raw,'existing baseline changed; never create another')
        stage=b.decode(b.read(b.LOCAL/'inputs/packet.json',pins['inputs/packet.json']))
        b.validate_authority(b.decode(b.read(b.LOCAL/'freeze.json')),b.read(b.LOCAL/'freeze-review.md'),
            b.decode(b.read(b.LOCAL/'adoption.json')),b.decode(b.read(b.LOCAL/'namespace-admission.json')),
            pins['freeze.json'],pins['freeze-review.md'],pins['namespace-admission.json'],packet['source_commit'],packet['source_pins'])
        intent=dict(utc=b.utc(),packet_commit=args.packet_commit,source_commit=packet['source_commit'],pins=pins,
            packet_sha256=args.packet_sha256,review_sha256=args.packet_review_sha256,remote_root=b.ROOT,
            launcher_sha256=pins['launch_once.py'],baseline_sha256=b.BASELINE_SHA,resources=b.RESOURCES,
            deadlines=b.DEADLINES,admission=packet['admission'],freeze_sha256=pins['freeze.json'],
            namespace_admission_sha256=pins['namespace-admission.json'],controller_pid_local=os.getpid(),
            blind_retry_forbidden=True,replacement_worlds=False)
        save('controller.json',dict(pid=os.getpid(),held=[str(b.BASE/n) for n in
            ('ace-delivery-controller.lock','ace-foundation-controller.lock')],utc=b.utc()));started=True
        context=dict(root=b.ROOT,registry=b.REMOTE_REGISTRY,pins=pins,baseline=baseline,
                     archive=stage['archive'],manifest=stage['manifest'])
        common='C='+repr(context)+'\n'+REMOTE_COMMON
        ready=remote('readiness_transport.json',common+r'''
need(root.parent==root.parent.resolve() and root.parent.stat().st_uid==os.getuid(),'scratch parent ownership differs')
need(not os.path.lexists(root),'existing realization root; no retry/adoption')
need(not list(root.parent.glob('ace_llm_intervention_scientific_realization*')),'prior realization root exists')
r=reconcile()
root.mkdir(mode=0o700);(root/'logs').mkdir(mode=0o700);(root/'inputs').mkdir(mode=0o700)
probe=root/'write-read-probe'
with probe.open('xb') as f:f.write(b'ACE first scientific realization readiness\n');f.flush();os.fsync(f.fileno())
need(read(probe)==b'ACE first scientific realization readiness\n','scratch readback differs');probe.unlink()
fs=os.statvfs(root);need(fs.f_bavail*fs.f_frsize>=2*1024**3,'scratch output headroom insufficient')
r['scratch']=dict(uid=root.stat().st_uid,free_bytes=fs.f_bavail*fs.f_frsize,write_read_cleanup=True,df=run(['df','-h',str(root)]))
write(root/'readiness.json',r);print(json.dumps(r))
''')
        save('readiness.json',ready)
        # Two fixed destinations, never recursive transfer or merging a local directory.
        for group,subdir in (('root',''),('inputs','inputs/')):
            names=[n for n in pins if (n.startswith('inputs/'))==(group=='inputs')]
            command=['scp','-o','ControlPath=/tmp/defab-curc.sock','-o','BatchMode=yes',
                     *[str(b.LOCAL/n) for n in names],'paco0228@login.rc.colorado.edu:'+b.ROOT+'/'+subdir]
            try:p=subprocess.run(command,capture_output=True,text=True,timeout=45)
            except subprocess.TimeoutExpired as exc:
                save('transfer-'+group+'.json',dict(timeout=True,message=str(exc),retry_allowed=False));raise
            save('transfer-'+group+'.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
            b.need(p.returncode==0,'transfer failed; no retry')
        verified=remote('verification_transport.json',common+'\nprint(json.dumps(verify()))\n')
        save('remote_verified.json',verified)
        # Bind complete local claims before the final remote readiness/claim/submission transaction.
        for name,h in pins.items():
            b.need(b.read(b.LOCAL/name,h)==b.read(b.EXT/name,h),'local inputs changed under coordinator')
        claim=dict(schema='ace-scientific-screen-namespace-claim-v1',freeze_sha256=pins['freeze.json'],
            namespace_admission_sha256=pins['namespace-admission.json'],packet_sha256=args.packet_sha256,
            packet_commit=args.packet_commit,realization_root=b.ROOT,world_ids=b.WORLD_IDS,
            local_registry=str(b.LOCAL_REGISTRY),remote_registry=b.REMOTE_REGISTRY,
            retry_allowed=False,replacement_worlds=False,utc=b.utc())
        b.write(b.LOCAL_REGISTRY,b.raw(claim))
        b.need(b.read(b.LOCAL_REGISTRY)==b.raw(claim),'local exclusive namespace readback differs')
        save('namespace-claim.json',claim)
        intent['local_namespace_claim_sha256']=b.sha(b.raw(claim))
        save('submission.intent.json',intent)
        result=remote('submission_transport.json',common+'\nintent='+repr(intent)+'\nclaim='+repr(claim)+r'''
need(not os.path.lexists(root/'submission.intent.json') and not os.path.lexists(root/'submission.result.json')
     and not os.path.lexists(root/'realization'),'existing attempt or realization output')
need(sha(read(root/'readiness.json'))==C['readiness_sha256'],'original readiness differs')
fresh=reconcile();verify()
need(registry.parent==root.parent and registry.parent.stat().st_uid==os.getuid(),'registry parent differs')
write(registry,claim)
need(sha(read(registry))==intent['local_namespace_claim_sha256'],'remote namespace claim readback differs')
intent.update(remote_utc=utc(),fresh_readiness=fresh,remote_namespace_claim_sha256=sha(read(registry)))
write(root/'submission.intent.json',intent)
r=dict(utc=utc(),intent_sha256=sha(read(root/'submission.intent.json')),job_id=None)
try:
 p=subprocess.run(['sbatch','--parsable',str(root/'job.sbatch')],capture_output=True,text=True,timeout=25)
 token=p.stdout.strip();valid=re.fullmatch(r'[1-9][0-9]*(;alpine)?',token) is not None
 r.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
          job_id=token.split(';')[0] if p.returncode==0 and valid else None,
          ambiguous=not(p.returncode==0 and valid))
except BaseException as exc:
 def txt(v):return v.decode(errors='replace') if isinstance(v,bytes) else v
 r.update(ambiguous=True,type=type(exc).__name__,message=str(exc),
          stdout=txt(getattr(exc,'stdout',None)),stderr=txt(getattr(exc,'stderr',None)))
write(root/'submission.result.json',r);print(json.dumps(r))
'''.replace("C['readiness_sha256']",repr(b.sha(b.raw(ready)))))
        save('submission.result.json',result)
        b.need(result['job_id'] is not None,'failed/ambiguous submission; retain registries and reconcile, never retry')
        print(b.raw(result).decode())
    except BaseException as exc:
        if started:
            try:save('launcher_failure.json',dict(utc=b.utc(),type=type(exc).__name__,message=str(exc),
                     blind_retry_forbidden=True,preserve_all_namespace_claims_and_intents=True))
            except BaseException:pass
        raise
    finally:
        for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
        if started:
            try:save('controller_released.json',dict(utc=b.utc(),pid=os.getpid(),released=True))
            except BaseException:pass


if __name__=='__main__':main()

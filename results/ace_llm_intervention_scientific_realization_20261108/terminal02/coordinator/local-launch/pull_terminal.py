"""Complete explicit pre-draw recovery02 realization terminal custody, never acceptance.

Standalone sidecar. Requires independent current job/outer packet/result pins.
Copies the whole root and both separate namespace claims, retains all terminal
failures/partial outputs, and performs a second remote inventory/readback.
No scientific output JSON is decoded; no job mutation, retry, model or scoring.
"""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import time

REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LAUNCH=REPO/'results/ace_llm_intervention_scientific_realization_20261108/launch02'
LAUNCH_EXT=BASE/'ACE-LLM-intervention-scientific-realization-20261108-02'
LOCAL=LAUNCH.parent/'terminal02'
EXT=BASE/'ACE-LLM-intervention-scientific-realization-terminal-20261108-02'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02'
LOCAL_NAMESPACE=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json'
REMOTE_NAMESPACE=str(Path(ROOT).parent/'.ace-scientific-screen-20261107.namespace-recovery02.json')
OLD_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01'
OLD_LOCAL_NAMESPACE=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
OLD_REMOTE_NAMESPACE=str(Path(ROOT).parent/'.ace-scientific-screen-20261107.namespace.json')
OLD_CLAIM_SHA='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
FAILURE_SHA='5da56ce061706f29a23c9a37a898bcbac128d16421f6cfd30731655208021a8d'
NO_DRAW_SHA='5fb94f9d07da3acd25c979250471037e8981066e0b201716f8c32d0b681fb9ef'
FAILED_JOB='35402725'
BASELINE_SHA='63cd36dfd9c0dcda4ef8ceec7b99d91ecfa696676cd1d99149256c6a0fdb3727'
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL',
          'PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()
def raw(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def pin(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def decode(data):
    def pairs(items):
        result={}
        for k,v in items:
            need(k not in result,'duplicate metadata key');result[k]=v
        return result
    def bad(value):raise ValueError('nonfinite metadata: '+value)
    return json.loads(data,object_pairs_hook=pairs,parse_constant=bad)


def fingerprint(path,*,sole_link=False):
    path=Path(path)
    need(not path.is_symlink(),'symlink custody forbidden: '+str(path))
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        before=os.fstat(f.fileno())
        need(stat.S_ISREG(before.st_mode) and (not sole_link or before.st_nlink==1),'regular independent custody file required')
        digest=hashlib.sha256()
        for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
        after=os.fstat(f.fileno())
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
    need(identity(before)==identity(after)==identity(path.lstat()),'file changed during inventory')
    return dict(bytes=after.st_size,sha256=digest.hexdigest())


def read_metadata(path,expected=None):
    meta=fingerprint(path,sole_link=True);need(meta['bytes']<=32*1024**2,'bounded coordinator metadata required')
    data=Path(path).read_bytes()
    need(sha(data)==meta['sha256'],'metadata changed during read')
    if expected is not None:need(pin(expected) and sha(data)==expected,'independent pin differs: '+str(path))
    return data


def inventory(root):
    root=Path(root);need(root.is_dir() and not root.is_symlink(),'ordinary custody directory required')
    files={};directories=[]
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink(),'custody symlink forbidden')
        name=path.relative_to(root).as_posix()
        if path.is_dir():directories.append(name)
        else:files[name]=fingerprint(path,sole_link=True)
    return dict(files=files,directories=directories)


def write(path,data):
    with Path(path).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def normalize(rows):
    result=[]
    for row in rows:
        need(len(row)>=8 and not any(row[8:]) and row[0].isdigit() and row[4].isdigit() and row[5].isdigit(),
             'malformed allocation accounting row')
        pairs=[part.split('=',1) for part in row[7].split(',') if part]
        need(all(len(p)==2 for p in pairs) and len({p[0] for p in pairs})==len(pairs),'malformed allocation TRES')
        result.append([row[0],row[1].strip(),row[2].strip(),row[3].strip(),int(row[4]),int(row[5]),row[6].strip(),sorted(pairs)])
    need(len({r[0] for r in result})==len(result),'duplicate historical job')
    return sorted(result,key=lambda r:int(r[0]))


REMOTE_OBSERVATION=r'''
from pathlib import Path
import subprocess,hashlib,json,datetime,stat,os
root=Path(ROOT);namespace=Path(NAMESPACE);original=Path(OLD_ROOT);original_namespace=Path(OLD_NAMESPACE)
def need(ok,message):
 if not ok:raise ValueError(message)
def fingerprint(path):
 need(not path.is_symlink(),'remote symlink forbidden')
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode),'nonregular remote file')
  h=hashlib.sha256()
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
  after=os.fstat(f.fileno())
 identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
 need(identity(before)==identity(after)==identity(path.lstat()),'remote file changed during capture')
 return dict(bytes=after.st_size,sha256=h.hexdigest()),after
v=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results={},files={},directories=[],
       root_present=root.exists(),namespace_present=os.path.lexists(namespace),original_hardlink_groups=[],
       original_root_present=original.exists(),original_namespace_present=os.path.lexists(original_namespace))
commands={
 'related_accounting':['sacct','-X','-u','paco0228','-S','2026-11-02','--format',
   'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'],
 'steps':['sacct','-j',JOB,'--format',
   'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180,MaxRSS,TotalCPU','-n','-P'],
 'details':['sacct','-X','-j',JOB,'--format','JobIDRaw,Partition,QOS,TimelimitRaw,ReqTRES%180,User','-n','-P'],
 'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],
 'job':['scontrol','show','job',JOB,'-o']}
for name,command in commands.items():
 try:
  p=subprocess.run(command,capture_output=True,text=True,timeout=20)
  v['results'][name]=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 except subprocess.TimeoutExpired as exc:
  txt=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x
  v['results'][name]=dict(command=command,returncode=None,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr))
try:
 need(v['results']['related_accounting']['returncode']==0 and v['results']['queue']['returncode']==0,
      'scheduler observation unavailable')
 rows=[l.split('|') for l in v['results']['related_accounting']['stdout'].splitlines() if '|ace_llm' in l]
 selected=[r for r in rows if r[0]==JOB]
 need(len(selected)==1 and selected[0][2].split()[0] in TERMINAL,'requested job not conclusively terminal')
 queue=[l.split('|') for l in v['results']['queue']['stdout'].splitlines() if l.strip()]
 need(all(len(r)==5 and r[0]!=JOB and not r[1].lower().startswith('ace') for r in queue),'current/related ACE job active')
 v['parent_accounting']=selected[0];v['related_accounting']=rows;v['terminal_gate']=True
 links={}
 for label,tree,present in (('launch',root,v['root_present']),('failed01',original,v['original_root_present'])):
  if not present:continue
  need(tree.is_dir() and tree==tree.resolve(),'ordinary remote root required')
  v['directories'].append(label)
  for path in sorted(tree.rglob('*')):
   need(not path.is_symlink(),'remote tree symlink forbidden')
   name=label+'/'+path.relative_to(tree).as_posix()
   if path.is_dir():v['directories'].append(name);continue
   info,s=fingerprint(path);v['files'][name]=info
   if s.st_nlink>1:links.setdefault((s.st_dev,s.st_ino,s.st_nlink),[]).append(name)
 for name,path,present in (('namespace.json',namespace,v['namespace_present']),
                           ('original-namespace.json',original_namespace,v['original_namespace_present'])):
  if not present:continue
  need(path.parent==path.parent.resolve(),'registry parent alias')
  info,s=fingerprint(path);v['files'][name]=info
  if s.st_nlink>1:links.setdefault((s.st_dev,s.st_ino,s.st_nlink),[]).append(name)
 v['original_hardlink_groups']=sorted([dict(paths=sorted(paths),link_count=k[2]) for k,paths in links.items()],
                                     key=lambda group:group['paths'])
 v['journal_pins']={str(p):fingerprint(p)[0]['sha256'] for d in root.parent.glob('ace_llm_intervention*')
                    for p in d.glob('submission.*.json')}
except BaseException as exc:
 v['observation_error']=dict(type=type(exc).__name__,message=str(exc))
print(json.dumps(v,sort_keys=True))
'''


def resources(observation,job):
    row=observation['parent_accounting'];errors=[]
    cpus=int(row[4]);elapsed=int(row[5]);pairs=[v.split('=',1) for v in row[7].split(',') if v]
    need(all(len(v)==2 for v in pairs) and len({v[0] for v in pairs})==len(pairs),'malformed actual TRES')
    tres=dict(pairs);gres={k:v for k,v in tres.items() if k.startswith('gres/')}
    gpu=0
    if gres:
        try:
            typed=sum(int(v) for k,v in gres.items() if k.startswith('gres/gpu:'))
            gpu=int(gres['gres/gpu']) if 'gres/gpu' in gres else typed
            if any(not (k=='gres/gpu' or k.startswith('gres/gpu:')) for k in gres):gpu=None
            if 'gres/gpu' in gres and any(k.startswith('gres/gpu:') for k in gres) and gpu!=typed:gpu=None
        except ValueError:gpu=None
        errors.append('unexpected allocated generic/GPU resource on CPU realization')
    if not tres and (cpus or elapsed):gpu=None;errors.append('missing actual allocation TRES')
    details=observation['results']['details'];requested=False
    if details['returncode']==0:
        rows=[r.split('|') for r in details['stdout'].splitlines() if r.strip()]
        if len(rows)==1 and len(rows[0])>=6:
            r=rows[0];rp=[x.split('=',1) for x in r[4].split(',') if x];req=dict(rp) if all(len(x)==2 for x in rp) else {}
            requested=(r[:4]==[job,'acpu','cpu-normal','5'] and r[5]=='paco0228' and len(req)==len(rp)
                and req.get('cpu')=='2' and req.get('mem') in ('7680M','7.5G','7.50G')
                and req.get('node')=='1' and not any(k.startswith('gres/') for k in req))
    if not requested:errors.append('requested CPU profile missing/different')
    matched=(requested and row[1]=='ace_llm_scientific_realize02' and row[6]=='ucb736_asc2'
        and cpus==2 and tres.get('cpu')=='2' and tres.get('mem') in ('7680M','7.5G','7.50G')
        and tres.get('node')=='1' and not gres and 0<=elapsed<=300)
    if not matched:errors.append('actual allocation identity/profile differs')
    return dict(resource_contract_matched=matched,errors=errors,allocated_cpus=cpus,elapsed_seconds=elapsed,
        allocated_cpu_seconds=cpus*elapsed,gpu_count=gpu,gpu_slice_seconds=None if gpu is None else gpu*elapsed,
        scheduler_state=row[2],scheduler_exit=row[3],account=row[6],alloc_tres=row[7])



def validate_remote_intent(data, expected_pin, local_intent, claim_pin):
    need(sha(data)==expected_pin, 'authenticated remote intent bytes differ')
    remote=decode(data)
    additions={'remote_utc','fresh_readiness','remote_namespace_claim_sha256'}
    need(type(local_intent) is dict and not (set(local_intent)&additions)
         and type(remote) is dict and set(remote)==set(local_intent)|additions,
         'remote intent fields differ from local intent plus explicit additions')
    need(raw({k:remote[k] for k in local_intent})==raw(local_intent),
         'authenticated remote intent differs from original local intent')
    need(pin(claim_pin) and remote['remote_namespace_claim_sha256']==claim_pin
         and remote['local_namespace_claim_sha256']==claim_pin,
         'authenticated remote intent namespace claim differs')
    need(type(remote['remote_utc']) is str and bool(remote['remote_utc'])
         and type(remote['fresh_readiness']) is dict, 'remote intent augmentation malformed')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('job-id','packet-sha256','submission-result-sha256'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();need(re.fullmatch('[1-9][0-9]*',a.job_id) is not None,'exact current job ID required')
    packet_raw=read_metadata(LAUNCH/'packet.json',a.packet_sha256);packet=decode(packet_raw)
    result_raw=read_metadata(LAUNCH/'submission.result.json',a.submission_result_sha256);result=decode(result_raw)
    need(packet['schema']=='ace-scientific-realization-launch-v1' and packet['remote_root']==ROOT
         and result.get('job_id')==a.job_id and result.get('returncode')==0 and result.get('ambiguous') is False
         and pin(result.get('intent_sha256')),'unambiguous original packet/job/result required')
    intent_raw=read_metadata(LAUNCH/'submission.intent.json');intent=decode(intent_raw)
    pins={**packet['new_input_files'],'packet.json':a.packet_sha256}
    need(intent['packet_sha256']==a.packet_sha256 and intent['remote_root']==ROOT and intent['pins']==pins,
         'current local submission lineage differs')
    need(all(type(n) is str and not Path(n).is_absolute() and '..' not in Path(n).parts and pin(h) for n,h in pins.items()),
         'safe complete input pin roster required')
    baseline_raw=read_metadata(LAUNCH/'accepted-accounting.json',BASELINE_SHA);baseline=decode(baseline_raw)
    need(baseline['terminal_jobs']==39 and baseline['submission_journals']==80
         and baseline['closed_cpu_seconds']==11048 and baseline['closed_gpu_slice_seconds']==1601,
         'exact existing 39-job/80-journal baseline required')
    need(a.job_id!=FAILED_JOB,'new recovery job must differ from accepted failed attempt')
    accepted=decode(read_metadata(LAUNCH/'failure-acceptance.json',FAILURE_SHA))
    read_metadata(LAUNCH/'no-draw-proof.json',NO_DRAW_SHA)
    authority=decode(read_metadata(LAUNCH/'recovery-authority.json',pins['recovery-authority.json']))
    expected_authority=dict(schema='ace-scientific-realization-recovery-authority-v1',status='authorized',
        recovery_id='realization02',failed_job_id=FAILED_JOB,failure_acceptance_sha256=FAILURE_SHA,
        no_draw_proof_sha256=NO_DRAW_SHA,predecessor_freeze_sha256=accepted['freeze_sha256'],
        successor_freeze_sha256=pins['freeze.json'],source_commit=packet['source_commit'],
        original_namespace_claim_sha256=OLD_CLAIM_SHA,accounting_sha256=BASELINE_SHA,
        original_root=OLD_ROOT,recovery_root=ROOT,local_recovery_registry=str(LOCAL_NAMESPACE),
        remote_recovery_registry=REMOTE_NAMESPACE,one_submission_only=True,automatic_retry_allowed=False,
        replacement_worlds_allowed=False)
    need(raw(authority)==raw(expected_authority),'exact independently supplied recovery authority differs')
    recovery_bindings=dict(recovery_authority_sha256=pins['recovery-authority.json'],
        original_namespace_claim_sha256=OLD_CLAIM_SHA,failed_job_id=FAILED_JOB,
        failure_acceptance_sha256=FAILURE_SHA,no_draw_proof_sha256=NO_DRAW_SHA)
    need(all(intent.get(k)==v for k,v in recovery_bindings.items()) and intent.get('explicit_recovery_only') is True,
         'local intent recovery lineage differs')
    expected_failed=[]
    for name,accepted_name in (('failure-inventory.json','inventory_transport.json'),
                               ('failure-readback.json','inventory_after_transport.json')):
        transport=decode(read_metadata(LAUNCH/name,accepted['evidence_pins'][accepted_name]))
        need(transport['returncode']==0,'accepted predecessor inventory transport failed')
        v=decode(transport['stdout']);expected_failed.append(dict(files=v['files'],directories=v['directories']))
    failed=expected_failed[0]
    need(failed==expected_failed[1] and len(failed['files'])==accepted['raw_files']==41
         and sum(v['bytes'] for v in failed['files'].values())==accepted['raw_bytes']
         and failed['files']['namespace.json']['sha256']==OLD_CLAIM_SHA,
         'accepted complete original41-file roster differs')
    projection=decode(read_metadata(LAUNCH/'failed-raw-inventory.json',pins['failed-raw-inventory.json']))
    need(projection==dict(root=OLD_ROOT,registry=OLD_REMOTE_NAMESPACE,**failed),'captured failed roster differs')
    locks=[];claimed=[];sealed=False;started=time.monotonic();cpu=time.process_time()
    def save(name,value):
        data=raw(value)
        for dest in claimed:write(dest/name,data)
    def run_saved(name,command,*,source=None,timeout):
        try:
            p=subprocess.run(command,input=source,capture_output=True,text=True,timeout=timeout)
            transport=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,utc=utc())
        except subprocess.TimeoutExpired as exc:
            txt=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x
            transport=dict(command=command,returncode=None,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr),utc=utc())
        if source is not None:transport['python_source']=source
        save(name,transport);need(transport['returncode']==0,'failed/partial custody retained: '+name)
        return transport
    source='\n'.join(k+'='+repr(v) for k,v in dict(ROOT=ROOT,NAMESPACE=REMOTE_NAMESPACE,JOB=a.job_id,
        TERMINAL=TERMINAL,OLD_ROOT=OLD_ROOT,OLD_NAMESPACE=OLD_REMOTE_NAMESPACE).items())+'\n'+REMOTE_OBSERVATION
    def observe(name):
        transport=run_saved(name,['bash',HELPER,'run','python3 -'],source=source,timeout=240)
        v=decode(transport['stdout'])
        need(v.get('terminal_gate') is True and not v.get('observation_error'),
             'terminal inventory unavailable: '+str(v.get('observation_error')))
        return v
    def copy_file(src,dst):
        with Path(src).open('rb') as source_file,Path(dst).open('xb') as target:
            shutil.copyfileobj(source_file,target,1024**2);target.flush();os.fsync(target.fileno())
        return str(dst)
    try:
        for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
            lock=BASE/name;need(lock==lock.resolve(),'coordinator lock alias')
            f=os.fdopen(os.open(lock,os.O_RDWR|os.O_NOFOLLOW),'r+')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:f.close();raise
            locks.append(f)
        need(all(not os.path.lexists(p) for p in (LOCAL,EXT)),'terminal custody already claimed; never retry')
        for dest in (EXT,LOCAL):
            dest.mkdir(mode=0o700);claimed.append(dest)
            for sub in ('remote','coordinator','namespaces'):(dest/sub).mkdir(mode=0o700)
            write(dest/'pull_terminal.py',Path(__file__).read_bytes())
        save('collector-intent.json',dict(utc=utc(),job_id=a.job_id,packet_sha256=a.packet_sha256,
            submission_result_sha256=a.submission_result_sha256,remote_root=ROOT,remote_namespace=REMOTE_NAMESPACE,
            local_namespace=str(LOCAL_NAMESPACE),collector_sha256=sha(Path(__file__).read_bytes()),
            original_root=OLD_ROOT,original_local_namespace=str(OLD_LOCAL_NAMESPACE),
            original_remote_namespace=OLD_REMOTE_NAMESPACE,recovery_bindings=recovery_bindings,
            retry_allowed=False,acceptance=False))
        # Keep complete local and external coordinator histories, not just their successful receipts.
        coordinator={}
        for label,original in (('local-launch',LAUNCH),('external-launch',LAUNCH_EXT)):
            before=inventory(original)
            shutil.copytree(original,EXT/'coordinator'/label,copy_function=copy_file)
            need(inventory(original)==inventory(EXT/'coordinator'/label)==before,'coordinator changed during copy')
            coordinator[label]=before
        local_claim_present=os.path.lexists(LOCAL_NAMESPACE)
        if local_claim_present:
            claim_raw=read_metadata(LOCAL_NAMESPACE)
            for dest in claimed:write(dest/'namespaces/local.namespace.json',claim_raw)
        else:claim_raw=None
        save('local-namespace-observation.json',dict(path=str(LOCAL_NAMESPACE),present=local_claim_present,
            sha256=None if claim_raw is None else sha(claim_raw)))
        original_claim_raw=read_metadata(OLD_LOCAL_NAMESPACE) if os.path.lexists(OLD_LOCAL_NAMESPACE) else None
        if original_claim_raw is not None:
            for dest in claimed:write(dest/'namespaces/original-local.namespace.json',original_claim_raw)
        save('original-local-namespace-observation.json',dict(path=str(OLD_LOCAL_NAMESPACE),
            present=original_claim_raw is not None,sha256=None if original_claim_raw is None else sha(original_claim_raw)))
        before=observe('inventory_transport.json')
        size=sum(v['bytes'] for v in before['files'].values())
        for parent in (BASE,LOCAL.parent):
            fs=os.statvfs(parent);need(fs.f_bavail*fs.f_frsize>size*2+2*1024**3,'insufficient complete-custody space')
        for label,remote_path,present in (('launch',ROOT,before['root_present']),
                    ('namespace.json',REMOTE_NAMESPACE,before['namespace_present']),
                    ('failed01',OLD_ROOT,before['original_root_present']),
                    ('original-namespace.json',OLD_REMOTE_NAMESPACE,before['original_namespace_present'])):
            if not present:continue
            command=['scp',*(['-r'] if label in ('launch','failed01') else []),'-o','ControlPath=/tmp/defab-curc.sock',
                     '-o','BatchMode=yes','paco0228@login.rc.colorado.edu:'+remote_path,str(EXT/'remote'/label)]
            run_saved('transfer-'+label+'.json',command,timeout=1800)
        actual=inventory(EXT/'remote')
        need(actual==dict(files=before['files'],directories=sorted(before['directories'])),
             'whole remote tree/namespace custody differs; keep all partial bytes')
        after=observe('inventory_after_transport.json')
        need(after['files']==before['files'] and sorted(after['directories'])==sorted(before['directories'])
             and after['root_present']==before['root_present'] and after['namespace_present']==before['namespace_present']
             and after['original_root_present']==before['original_root_present']
             and after['original_namespace_present']==before['original_namespace_present']
             and after['original_hardlink_groups']==before['original_hardlink_groups']
             and after['parent_accounting']==before['parent_accounting']
             and after['journal_pins']==before['journal_pins']
             and normalize(after['related_accounting'])==normalize(before['related_accounting']),
             'second remote bytes/journals/terminal accounting differ')
        for sub in ('remote','coordinator'):
            for path in (EXT/sub).iterdir():
                target=LOCAL/sub/path.name
                if path.is_dir():shutil.copytree(path,target,copy_function=copy_file)
                else:copy_file(path,target)
            need(inventory(EXT/sub)==inventory(LOCAL/sub),'second independent byte copy differs')
        errors=[]
        oldfiles={('namespace.json' if n=='original-namespace.json' else 'launch/'+n.removeprefix('failed01/')):v
                  for n,v in actual['files'].items() if n=='original-namespace.json' or n.startswith('failed01/')}
        olddirs=[('launch' if n=='failed01' else 'launch/'+n.removeprefix('failed01/'))
                 for n in actual['directories'] if n=='failed01' or n.startswith('failed01/')]
        if dict(files=oldfiles,directories=sorted(olddirs))!=dict(files=failed['files'],directories=sorted(failed['directories'])):
            errors.append('complete accepted original41-file failed custody changed/missing')
        if original_claim_raw is None or sha(original_claim_raw)!=OLD_CLAIM_SHA:
            errors.append('original local registry missing/changed')
        if original_claim_raw is not None and (not os.path.lexists(OLD_LOCAL_NAMESPACE)
                or fingerprint(OLD_LOCAL_NAMESPACE)['sha256']!=sha(original_claim_raw)):
            errors.append('original local registry changed during collection')
        if not before['root_present']:errors.append('original remote root missing')
        mismatches=[n for n,h in pins.items() if actual['files'].get('launch/'+n,{}).get('sha256')!=h]
        if mismatches:errors.append('remote original input/packet pin disagreement')
        if actual['files'].get('launch/submission.result.json',{}).get('sha256')!=a.submission_result_sha256:
            errors.append('remote current result pin differs')
        if actual['files'].get('launch/submission.intent.json',{}).get('sha256')!=result['intent_sha256']:
            errors.append('remote current intent pin differs')
        expected_claim=intent.get('local_namespace_claim_sha256')
        if claim_raw is None or not pin(expected_claim) or sha(claim_raw)!=expected_claim:
            errors.append('separate local namespace claim missing/different')
        if actual['files'].get('namespace.json',{}).get('sha256')!=expected_claim:
            errors.append('separate remote namespace claim missing/different')
        try:
            remote_intent_raw=read_metadata(EXT/'remote/launch/submission.intent.json',result['intent_sha256'])
            validate_remote_intent(remote_intent_raw,result['intent_sha256'],intent,expected_claim)
        except (ValueError,TypeError,KeyError,OSError) as exc:
            errors.append('authenticated remote/local intent lineage disagreement: '+str(exc))
        if claim_raw is not None:
            claim=decode(claim_raw)  # Coordinator metadata only; never task/descriptors.
            if not (claim.get('packet_sha256')==a.packet_sha256 and claim.get('realization_root')==ROOT
                    and claim.get('freeze_sha256')==pins.get('freeze.json')
                    and claim.get('namespace_admission_sha256')==pins.get('namespace-admission.json')):
                errors.append('namespace claim packet/freeze binding differs')
            if not (all(claim.get(k)==v for k,v in recovery_bindings.items())
                    and claim.get('claim_kind')=='explicit_pre_draw_recovery02'
                    and claim.get('local_registry')==str(LOCAL_NAMESPACE)
                    and claim.get('remote_registry')==REMOTE_NAMESPACE
                    and claim.get('retry_allowed') is False and claim.get('replacement_worlds') is False):
                errors.append('namespace recovery authority/failed-attempt binding differs')
        prior=[r for r in before['related_accounting'] if r[0]!=a.job_id]
        accounting_ok=len(prior)==39 and normalize(prior)==normalize(baseline['related_accounting'])
        if not accounting_ok:errors.append('exact 39 prior accounting rows differ')
        steps=before['results']['steps']
        step_parents=[r.split('|') for r in (steps.get('stdout') or '').splitlines() if r.startswith(a.job_id+'|')]
        if steps['returncode']!=0 or len(step_parents)!=1 or normalize([step_parents[0][:8]])!=normalize([before['parent_accounting']]):
            errors.append('full job/step accounting missing or allocation row disagrees')
        journals={**baseline['journal_pins'],ROOT+'/submission.intent.json':result['intent_sha256'],
                  ROOT+'/submission.result.json':a.submission_result_sha256}
        journal_ok=len(journals)==82 and before['journal_pins']==journals
        if not journal_ok:errors.append('exact 80 prior plus two current journals differ')
        for label,original in (('local-launch',LAUNCH),('external-launch',LAUNCH_EXT)):
            if inventory(original)!=coordinator[label]:errors.append(label+' changed while custody was collected')
            captured=coordinator[label]['files']
            if any(captured.get(n,{}).get('sha256')!=h for n,h in pins.items()):errors.append(label+' input pin disagreement')
            if captured.get('submission.result.json',{}).get('sha256')!=a.submission_result_sha256:
                errors.append(label+' current submission result differs')
            if captured.get('submission.intent.json',{}).get('sha256')!=sha(intent_raw):
                errors.append(label+' original local intent differs')
        if local_claim_present and fingerprint(LOCAL_NAMESPACE)['sha256']!=sha(claim_raw):errors.append('local namespace changed')
        allocation=resources(before,a.job_id)
        markers={n:actual['files'].get('launch/realization/'+n) for n in (
            'terminal.json','terminal-candidate.json','failure.json','publication-interruption.json',
            'session/worker/terminal.json','session/worker/failure.json',
            'session/worker/task/terminal.json','session/worker/task/failure.json')}
        complete_markers=all(markers[n] is not None for n in (
            'terminal.json','session/worker/terminal.json','session/worker/task/terminal.json'))
        outcome='terminal_custody_review_pending' if complete_markers else 'terminal_partial_outputs_preserved'
        if allocation['scheduler_state'].split()[0]!='COMPLETED' or allocation['scheduler_exit']!='0:0' or any(
                value is not None for name,value in markers.items() if 'failure.json' in name or 'interruption.json' in name):
            outcome='terminal_failed_outputs_preserved'
        if errors:outcome='terminal_custody_with_binding_disagreements'
        disposition=dict(schema='ace-scientific-realization-terminal-custody-v1',utc=utc(),status=outcome,
            job_id=a.job_id,packet_sha256=a.packet_sha256,submission_result_sha256=a.submission_result_sha256,
            custody_bytes_verified=True,current_lineage_verified=not errors,errors=errors,input_pin_mismatches=mismatches,
            remote_root=ROOT,output_root=ROOT+'/realization',remote_files=actual['files'],remote_directories=actual['directories'],
            raw_files=len(actual['files']),raw_bytes=size,original_hardlink_groups=before['original_hardlink_groups'],
            hardlink_custody='All original names copied to separate regular files; original alias metadata retained, no alias dereference/omission.',
            namespace_local_sha256=None if claim_raw is None else sha(claim_raw),
            namespace_remote_sha256=actual['files'].get('namespace.json',{}).get('sha256'),
            recovery_bindings=recovery_bindings,original_failed_raw_files=len(oldfiles),
            original_failed_custody_verified=(oldfiles==failed['files'] and sorted(olddirs)==sorted(failed['directories'])),
            original_namespace_local_sha256=None if original_claim_raw is None else sha(original_claim_raw),
            original_namespace_remote_sha256=actual['files'].get('original-namespace.json',{}).get('sha256'),
            exact_prior_accounting_matched=accounting_ok,exact_prior_and_current_journals_matched=journal_ok,
            observed_related_jobs=len(before['related_accounting']),observed_submission_journals=len(before['journal_pins']),
            allocation=allocation,closed_cpu_seconds=(11048+allocation['allocated_cpu_seconds']) if accounting_ok else None,
            closed_gpu_slice_seconds=(1601+allocation['gpu_slice_seconds']) if accounting_ok and allocation['gpu_slice_seconds'] is not None else None,
            preparation_allowance_cpu_seconds=3600,
            cpu_admission_after_release=(14648+allocation['allocated_cpu_seconds']) if accounting_ok else None,
            cpu_ceiling_seconds=57600,gpu_ceiling_seconds=7200,output_marker_metadata=markers,
            output_json_decoded=False,scientific_qualified=False,realization_accepted=False,acceptance=False,retry_allowed=False,
            collector_wall_seconds=time.monotonic()-started,collector_process_cpu_seconds=time.process_time()-cpu,
            cost_scope='Local collector process only; subprocess/remote costs not inferred. Actual Slurm allocation costs reported separately.')
        save('disposition.json',disposition)
        need(not errors,'full bytes retained but bindings disagree; independent reconciliation required')
        inventories={str(dest):inventory(dest) for dest in claimed}
        need(inventories[str(EXT)]==inventories[str(LOCAL)],'complete two-copy capsule differs')
        save('custody-inventory.json',dict(copies=inventories,identical=True,
             excludes='custody-inventory.json itself',acceptance=False))
        need(inventory(EXT)==inventory(LOCAL),'final capsule readback differs')
        sealed=True
        print(raw({k:v for k,v in disposition.items() if k not in ('remote_files','remote_directories')}).decode())
    except BaseException as exc:
        if claimed and not sealed:
            try:save('collector_failure.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),
                retry_allowed=False,acceptance=False,partial_custody_preserved=True))
            except BaseException:pass
        raise
    finally:
        if claimed and not sealed:
            try:
                inventories={str(dest):inventory(dest) for dest in claimed}
                same=len(claimed)==2 and inventories[str(EXT)]==inventories[str(LOCAL)]
                name='custody-failure-inventory.json'
                save(name,dict(copies=inventories,identical=same,excludes=name+' itself',acceptance=False))
            except BaseException:pass
        for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()


if __name__=='__main__':main()

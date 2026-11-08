"""Complete collection01 terminal custody, never independent collection acceptance.

Standalone sidecar. Requires independent current job/outer packet/result pins.
Copies the whole root and both separate namespace claims, retains all terminal
failures/partial outputs, and performs a second remote inventory/readback.
Only public task terminal counts are decoded after custody; no payload/scoring/model execution.
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
LAUNCH=REPO/'results/ace_llm_intervention_scientific_collection_20261108/launch01'
LAUNCH_EXT=BASE/'ACE-LLM-intervention-scientific-collection-20261108-01'
LOCAL=LAUNCH.parent/'terminal01'
EXT=BASE/'ACE-LLM-intervention-scientific-collection-terminal-20261108-01'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01'
REALIZATION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02'
LOCAL_NAMESPACE=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json'
REMOTE_NAMESPACE=str(Path(ROOT).parent/'.ace-scientific-screen-20261107.namespace-recovery02.json')
OLD_LOCAL_NAMESPACE=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
OLD_REMOTE_NAMESPACE=str(Path(ROOT).parent/'.ace-scientific-screen-20261107.namespace.json')
OLD_CLAIM_SHA='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
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
root=Path(ROOT);namespace=Path(NAMESPACE);original_namespace=Path(OLD_NAMESPACE)
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
       original_namespace_present=os.path.lexists(original_namespace),upstream_pins={},upstream_errors={})
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
 for label,tree,present in (('launch',root,v['root_present']),):
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
 # Explicit training-only paths; never enumerate the realization root or read world032..055.
 for path in UPSTREAM_PINS:
  try:v['upstream_pins'][path]=fingerprint(Path(path))[0]['sha256']
  except (OSError,ValueError) as exc:v['upstream_errors'][path]=dict(type=type(exc).__name__,message=str(exc))

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
        errors.append('unexpected allocated generic/GPU resource on CPU collection')
    if not tres and (cpus or elapsed):gpu=None;errors.append('missing actual allocation TRES')
    details=observation['results']['details'];requested=False
    if details['returncode']==0:
        rows=[r.split('|') for r in details['stdout'].splitlines() if r.strip()]
        if len(rows)==1 and len(rows[0])>=6:
            r=rows[0];rp=[x.split('=',1) for x in r[4].split(',') if x];req=dict(rp) if all(len(x)==2 for x in rp) else {}
            requested=(r[:4]==[job,'acpu','cpu-normal','15'] and r[5]=='paco0228' and len(req)==len(rp)
                and req.get('cpu')=='2' and req.get('mem') in ('7680M','7.5G','7.50G')
                and req.get('node')=='1' and not any(k.startswith('gres/') for k in req))
    if not requested:errors.append('requested CPU profile missing/different')
    matched=(requested and row[1]=='ace_llm_scientific_collect01' and row[6]=='ucb736_asc2'
        and cpus==2 and tres.get('cpu')=='2' and tres.get('mem') in ('7680M','7.5G','7.50G')
        and tres.get('node')=='1' and not gres and 0<=elapsed<=900)
    if not matched:errors.append('actual allocation identity/profile differs')
    return dict(resource_contract_matched=matched,errors=errors,allocated_cpus=cpus,elapsed_seconds=elapsed,
        allocated_cpu_seconds=cpus*elapsed,gpu_count=gpu,gpu_slice_seconds=None if gpu is None else gpu*elapsed,
        scheduler_state=row[2],scheduler_exit=row[3],account=row[6],alloc_tres=row[7])



def validate_remote_intent(data, expected_pin, local_intent, claim_pin):
    need(sha(data)==expected_pin, 'authenticated remote intent bytes differ')
    remote=decode(data)
    additions={'remote_utc','fresh_readiness'}
    need(type(local_intent) is dict and not (set(local_intent)&additions)
         and type(remote) is dict and set(remote)==set(local_intent)|additions,
         'remote intent fields differ from local intent plus explicit additions')
    need(raw({k:remote[k] for k in local_intent})==raw(local_intent),
         'authenticated remote intent differs from original local intent')
    need(pin(claim_pin) and remote['existing_namespace_claim_sha256']==claim_pin,
         'authenticated remote intent namespace claim differs')
    need(type(remote['remote_utc']) is str and bool(remote['remote_utc'])
         and type(remote['fresh_readiness']) is dict, 'remote intent augmentation malformed')


def task_counts(task,files):
    prefix='launch/collection/session/worker/task/'
    evidence=files.get(prefix+'terminal.json')
    report=dict(present=evidence is not None,reported=None,errors=[],independent_acceptance=False,
        accounting_authority='Complete task/collection tree and original attempt journals; no missing charges inferred as zero.')
    if evidence is None:return report
    try:
        value=decode(read_metadata(task/'terminal.json',evidence['sha256']))
        need(value['schema']=='ace-scientific-training-collection-cpu-v1' and value['status']=='collection_complete',
             'saved public task terminal schema/status differs')
        need(value['packet_sha256']==files['launch/inputs/collection-packet.json']['sha256']
             and value['admission_sha256']==files['launch/inputs/collection-admission.json']['sha256'],
             'saved task terminal original collection packet/admission pins differ')
        need(type(value['record_count']) is int and value['record_count']==160
             and type(value['retained_pair_count']) is int and value['retained_pair_count']==3584,
             'reported complete collection record/pair counts differ')
        for key,name in (('collection_terminal_sha256','terminal.json'),('collection_inventory_sha256','inventory.json'),
                         ('collection_plan_sha256','plan.json')):
            need(value[key]==files[prefix+'collection/'+name]['sha256'],'saved collection terminal/inventory/plan pin differs')
        account=value['accounting']
        need(type(account) is dict and type(account['reserved_response_attempts']) is int
             and 0<=account['reserved_response_attempts']<=9216
             and account['reserved_response_attempts']==account['known_returns']==account['executor_accepted_responses']
             and account['unknown_return_attempts']==0 and account['total_reserved_cost_units']==10240,
             'reported complete collection accounting differs')
        need(value['worlds_generated']==value['training_updates']==value['tokenizer_calls']==value['model_loads']==0
             and value['development_evaluated'] is False and value['retry_allowed'] is False,
             'collection scope flags differ')
        report['reported']={k:value[k] for k in ('record_count','retained_pair_count','accounting',
            'worlds_generated','training_updates','tokenizer_calls','model_loads','development_evaluated')}
    except (ValueError,TypeError,KeyError,OSError) as exc:report['errors'].append('public task terminal metadata disagreement: '+str(exc))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('job-id','packet-sha256','submission-result-sha256'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();need(re.fullmatch('[1-9][0-9]*',a.job_id) is not None,'exact current job ID required')
    packet_raw=read_metadata(LAUNCH/'packet.json',a.packet_sha256);packet=decode(packet_raw)
    result_raw=read_metadata(LAUNCH/'submission.result.json',a.submission_result_sha256);result=decode(result_raw)
    need(packet['schema']=='ace-scientific-collection-launch-v1' and packet['remote_root']==ROOT
         and result.get('job_id')==a.job_id and result.get('returncode')==0 and result.get('ambiguous') is False
         and pin(result.get('intent_sha256')),'unambiguous original packet/job/result required')
    intent_raw=read_metadata(LAUNCH/'submission.intent.json');intent=decode(intent_raw)
    pins={**packet['new_input_files'],'packet.json':a.packet_sha256}
    need(intent['packet_sha256']==a.packet_sha256 and intent['remote_root']==ROOT and intent['pins']==pins,
         'current local submission lineage differs')
    need(all(type(n) is str and not Path(n).is_absolute() and '..' not in Path(n).parts and pin(h) for n,h in pins.items()),
         'safe complete input pin roster required')
    baseline_pin=pins['baseline.json']
    baseline=decode(read_metadata(LAUNCH/'baseline.json',baseline_pin))
    need(packet['baseline_sha256']==intent['baseline_sha256']==baseline_pin
         and baseline['terminal_jobs']==40 and baseline['submission_journals']==82
         and baseline['preparation_allowance_cpu_seconds']==3600 and baseline['outstanding_reservation']==0,
         'independently pinned actual40-job/82-journal baseline required')
    accepted=decode(read_metadata(LAUNCH/'realization-acceptance.json',pins['realization-acceptance.json']))
    realization_result=decode(read_metadata(LAUNCH/'realization-result.json',pins['realization-result.json']))
    realization_packet=decode(read_metadata(LAUNCH/'realization-packet.json',pins['realization-packet.json']))
    upstream_transport=decode(read_metadata(LAUNCH/'realization-inventory.json',pins['realization-inventory.json']))
    need(accepted['status']=='accepted' and accepted['required_findings']==0
         and accepted['complete_custody_verified'] is True
         and accepted['inventory_transport_sha256']==pins['realization-inventory.json']
         and accepted['namespace_claim_sha256']==pins['namespace-claim.json']
         and accepted['realization_packet_sha256']==pins['realization-packet.json']
         and accepted['freeze_sha256']==pins['freeze.json']
         and accepted['namespace_admission_sha256']==pins['namespace-admission.json']
         and accepted['job_id']==realization_result['job_id']==packet['realization_job_id']
         and a.job_id!=accepted['job_id'] and realization_packet['remote_root']==REALIZATION_ROOT
         and realization_result['returncode']==0 and realization_result['ambiguous'] is False,
         'actual accepted upstream realization lineage required')
    need(upstream_transport['returncode']==0,'accepted upstream inventory transport required')
    upstream=decode(upstream_transport['stdout']);need(upstream['terminal_gate'] is True,'accepted upstream terminal inventory required')
    upstream_pins={REALIZATION_ROOT+'/packet.json':pins['realization-packet.json'],
        REALIZATION_ROOT+'/submission.result.json':pins['realization-result.json'],
        REALIZATION_ROOT+'/submission.intent.json':realization_result['intent_sha256']}
    prefix='launch/realization/session/worker/task/'
    for name in ('terminal.json',*[f'world-{i:03d}.json' for i in range(32)]):
        upstream_pins[REALIZATION_ROOT+'/realization/session/worker/task/'+name]=upstream['files'][prefix+name]['sha256']
    need(upstream_pins[REALIZATION_ROOT+'/realization/session/worker/task/terminal.json']==accepted['realization_terminal_sha256'],
         'accepted upstream terminal inventory join differs')
    recovery_bindings=dict(recovery_authority_sha256=pins['recovery-authority.json'],
        original_namespace_claim_sha256=OLD_CLAIM_SHA)
    need(intent['realization_acceptance_sha256']==pins['realization-acceptance.json']
         and intent['realization_job_id']==accepted['job_id']
         and intent['existing_namespace_claim_sha256']==packet['namespace_claim_sha256']==pins['namespace-claim.json']
         and intent['namespace_admission_sha256']==pins['namespace-admission.json']
         and all(intent.get(k)==v for k,v in recovery_bindings.items()) and intent['new_namespace_claimed'] is False,
         'current collection intent existing-namespace/acceptance joins differ')
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
        TERMINAL=TERMINAL,OLD_NAMESPACE=OLD_REMOTE_NAMESPACE,UPSTREAM_PINS=upstream_pins).items())+'\n'+REMOTE_OBSERVATION
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
            accepted_realization_root=REALIZATION_ROOT,original_local_namespace=str(OLD_LOCAL_NAMESPACE),
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
                    ('original-namespace.json',OLD_REMOTE_NAMESPACE,before['original_namespace_present'])):
            if not present:continue
            command=['scp',*(['-r'] if label=='launch' else []),'-o','ControlPath=/tmp/defab-curc.sock',
                     '-o','BatchMode=yes','paco0228@login.rc.colorado.edu:'+remote_path,str(EXT/'remote'/label)]
            run_saved('transfer-'+label+'.json',command,timeout=1800)
        actual=inventory(EXT/'remote')
        need(actual==dict(files=before['files'],directories=sorted(before['directories'])),
             'whole remote tree/namespace custody differs; keep all partial bytes')
        after=observe('inventory_after_transport.json')
        need(after['files']==before['files'] and sorted(after['directories'])==sorted(before['directories'])
             and after['root_present']==before['root_present'] and after['namespace_present']==before['namespace_present']
             and after['original_namespace_present']==before['original_namespace_present']
             and after['original_hardlink_groups']==before['original_hardlink_groups']
             and after['parent_accounting']==before['parent_accounting']
             and after['journal_pins']==before['journal_pins']
             and after['upstream_pins']==before['upstream_pins'] and after['upstream_errors']==before['upstream_errors']
             and normalize(after['related_accounting'])==normalize(before['related_accounting']),
             'second remote bytes/journals/terminal accounting differ')
        for sub in ('remote','coordinator'):
            for path in (EXT/sub).iterdir():
                target=LOCAL/sub/path.name
                if path.is_dir():shutil.copytree(path,target,copy_function=copy_file)
                else:copy_file(path,target)
            need(inventory(EXT/sub)==inventory(LOCAL/sub),'second independent byte copy differs')
        errors=[]
        if before['upstream_pins']!=upstream_pins or before['upstream_errors']:
            errors.append('accepted training-only realization bytes changed/missing')
        if actual['files'].get('original-namespace.json',{}).get('sha256')!=OLD_CLAIM_SHA:
            errors.append('original remote namespace claim missing/changed')
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
        expected_claim=intent.get('existing_namespace_claim_sha256')
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
            if not (claim.get('packet_sha256')==pins['realization-packet.json'] and claim.get('realization_root')==REALIZATION_ROOT
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
        accounting_ok=len(prior)==40 and normalize(prior)==normalize(baseline['related_accounting'])
        if not accounting_ok:errors.append('exact 40 prior accounting rows differ')
        steps=before['results']['steps']
        step_parents=[r.split('|') for r in (steps.get('stdout') or '').splitlines() if r.startswith(a.job_id+'|')]
        if steps['returncode']!=0 or len(step_parents)!=1 or normalize([step_parents[0][:8]])!=normalize([before['parent_accounting']]):
            errors.append('full job/step accounting missing or allocation row disagrees')
        journals={**baseline['journal_pins'],ROOT+'/submission.intent.json':result['intent_sha256'],
                  ROOT+'/submission.result.json':a.submission_result_sha256}
        journal_ok=len(journals)==84 and before['journal_pins']==journals
        if not journal_ok:errors.append('exact 82 prior plus two current journals differ')
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
        errors.extend(allocation['errors'])
        markers={n:actual['files'].get('launch/collection/'+n) for n in (
            'terminal.json','terminal-candidate.json','failure.json','publication-interruption.json',
            'session/worker/terminal.json','session/worker/failure.json',
            'session/worker/task/terminal.json','session/worker/task/failure.json',
            'session/worker/task/publication-interruption.json','session/worker/task/observation-failure.json',
            'session/worker/task/collection/terminal.json','session/worker/task/collection/failure.json')}
        complete_markers=all(markers[n] is not None for n in (
            'terminal.json','session/worker/terminal.json','session/worker/task/terminal.json','session/worker/task/collection/terminal.json'))
        reported=task_counts(EXT/'remote/launch/collection/session/worker/task',actual['files'])
        errors.extend(reported['errors'])
        outcome='terminal_custody_review_pending' if complete_markers else 'terminal_partial_outputs_preserved'
        if allocation['scheduler_state'].split()[0]!='COMPLETED' or allocation['scheduler_exit']!='0:0' or any(
                value is not None for name,value in markers.items() if 'failure.json' in name or 'interruption.json' in name):
            outcome='terminal_failed_outputs_preserved'
        if errors:outcome='terminal_custody_with_binding_disagreements'
        disposition=dict(schema='ace-scientific-collection-terminal-custody-v1',utc=utc(),status=outcome,
            job_id=a.job_id,packet_sha256=a.packet_sha256,submission_result_sha256=a.submission_result_sha256,
            custody_bytes_verified=True,current_lineage_verified=not errors,errors=errors,input_pin_mismatches=mismatches,
            remote_root=ROOT,output_root=ROOT+'/collection',remote_files=actual['files'],remote_directories=actual['directories'],
            raw_files=len(actual['files']),raw_bytes=size,original_hardlink_groups=before['original_hardlink_groups'],
            hardlink_custody='All original names copied to separate regular files; original alias metadata retained, no alias dereference/omission.',
            namespace_local_sha256=None if claim_raw is None else sha(claim_raw),
            namespace_remote_sha256=actual['files'].get('namespace.json',{}).get('sha256'),
            recovery_bindings=recovery_bindings,realization_acceptance_sha256=pins['realization-acceptance.json'],
            realization_inventory_transport_sha256=pins['realization-inventory.json'],
            accepted_realization_inputs=upstream_pins,upstream_training_inputs_verified=before['upstream_pins']==upstream_pins and not before['upstream_errors'],
            original_namespace_local_sha256=None if original_claim_raw is None else sha(original_claim_raw),
            original_namespace_remote_sha256=actual['files'].get('original-namespace.json',{}).get('sha256'),
            exact_prior_accounting_matched=accounting_ok,exact_prior_and_current_journals_matched=journal_ok,
            observed_related_jobs=len(before['related_accounting']),observed_submission_journals=len(before['journal_pins']),
            allocation=allocation,closed_cpu_seconds=(baseline['closed_cpu_seconds']+allocation['allocated_cpu_seconds']) if accounting_ok else None,
            closed_gpu_slice_seconds=(baseline['closed_gpu_slice_seconds']+allocation['gpu_slice_seconds']) if accounting_ok and allocation['gpu_slice_seconds'] is not None else None,
            preparation_allowance_cpu_seconds=3600,
            cpu_admission_after_release=(baseline['closed_cpu_seconds']+3600+allocation['allocated_cpu_seconds']) if accounting_ok else None,
            cpu_ceiling_seconds=57600,gpu_ceiling_seconds=7200,output_marker_metadata=markers,
            reported_task_counts=reported,planned_training_worlds=32,planned_records=160,planned_pairs=3584,
            response_attempt_cap=9216,reserved_cost_cap=10240,development_payloads_read=0,
            public_task_terminal_decoded=reported['present'],scientific_payloads_decoded=False,
            scientific_qualified=False,collection_accepted=False,acceptance=False,retry_allowed=False,
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

"""Complete token02 terminal custody, never independent token/scientific acceptance.

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
import shlex
import types

REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LAUNCH=REPO/'results/ace_llm_intervention_scientific_tokenization_20261108/launch02'
LAUNCH_EXT=BASE/'ACE-LLM-intervention-scientific-tokenization-20261108-02'
LOCAL=LAUNCH.parent/'terminal02'
EXT=BASE/'ACE-LLM-intervention-scientific-tokenization-terminal-20261108-02'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_02'
FAILED_TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'
FAILED_TOKEN_JOB='35415334'
BUILDER_SHA='c1bf0ae82b3fe004d76ffa151bf8181a0921bfc2b8f9120bcb16f44b72cb4fbc'
COMMON_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_02'
FAILED_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01'
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
       root_present=root.exists(),failed_token_present=Path(FAILED_TOKEN_ROOT).exists(),common_present=Path(COMMON_ROOT).exists(),failed_present=Path(FAILED_ROOT).exists(),namespace_present=os.path.lexists(namespace),original_hardlink_groups=[],
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
 for label,tree,present in (('launch',root,v['root_present']),('token01',Path(FAILED_TOKEN_ROOT),v['failed_token_present']),('common',Path(COMMON_ROOT),v['common_present']),('failed01',Path(FAILED_ROOT),v['failed_present'])):
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
        errors.append('unexpected allocated generic/GPU resource on CPU token preparation')
    if not tres and (cpus or elapsed):gpu=None;errors.append('missing actual allocation TRES')
    details=observation['results']['details'];requested=False
    if details['returncode']==0:
        rows=[r.split('|') for r in details['stdout'].splitlines() if r.strip()]
        if len(rows)==1 and len(rows[0])>=6:
            r=rows[0];rp=[x.split('=',1) for x in r[4].split(',') if x];req=dict(rp) if all(len(x)==2 for x in rp) else {}
            requested=(r[:4]==[job,'acpu','cpu-normal','10'] and r[5]=='paco0228' and len(req)==len(rp)
                and req.get('cpu')=='3' and req.get('mem') in ('11520M','11.25G')
                and req.get('node')=='1' and not any(k.startswith('gres/') for k in req))
    if not requested:errors.append('requested CPU profile missing/different')
    matched=(requested and row[1]=='ace_llm_scientific_token02' and row[6]=='ucb736_asc2'
        and cpus==3 and tres.get('cpu')=='3' and tres.get('mem') in ('11520M','11.25G')
        and tres.get('node')=='1' and not gres and 0<=elapsed<=600)
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
    # Metadata-only reported summary. Never decode exposure/prepared records/token IDs/plans.
    prefix='launch/preparation/session/worker/artifacts/'
    evidence=files.get(prefix+'terminal.json')
    report=dict(present=evidence is not None,reported=None,errors=[],independent_acceptance=False,
        accounting_authority='Actual scheduler rows remain authoritative even when preparation fails.')
    if evidence is None:return report
    try:
        value=decode(read_metadata(task/'terminal.json',evidence['sha256']))
        need(value['schema']=='ace-scientific-training-preparation-cpu-v1' and value['status']=='cpu_preparation_complete'
             and value['packet_sha256']==files['launch/inputs/packet.json']['sha256'], 'saved preparer terminal binding differs')
        need(type(value['records']) is int and value['records']==160 and type(value['retained_pairs']) is int
             and value['retained_pairs']==3584 and type(value['sft_steps']) is int and value['sft_steps']==160
             and type(value['dpo_steps']) is int and 0<value['dpo_steps']<=128, 'reported fixed exposure counts differ')
        need(value['llm_prepared_file_sha256']==files[prefix+'llm-prepared.json']['sha256']
             and value['numerical_plan_file_sha256']==files[prefix+'numerical-plan.json']['sha256']
             and value['runtime_stage_sha256']==files['launch/preparation/stage.json']['sha256'], 'saved artifact byte pins differ')
        for k in ('training_plan_sha256','numerical_plan_sha256','exposure_sha256'):need(pin(value[k]),'reported plan pin malformed')
        need(all(value[k] is False for k in ('model_weights_loaded','explicit_cuda_api_called','training_execution_qualified','scientific_qualified','retry_allowed'))
             and all(type(value[k]) is int and value[k]==0 for k in ('inference_calls','environment_responses','training_updates')),
             'preparation scope flags differ')
        report['reported']={k:value[k] for k in ('records','retained_pairs','sft_steps','dpo_steps','training_plan_sha256',
            'numerical_plan_sha256','exposure_sha256','llm_prepared_file_sha256','numerical_plan_file_sha256')}
    except (ValueError,TypeError,KeyError,OSError) as exc:report['errors'].append('public preparer terminal metadata disagreement: '+str(exc))
    return report


STREAM_HELPER_SHA='fab01ad840e4e73ef0fc2136fd2eeb12b1033256e80a8f96a392afb2d50b758a'
STREAM_REVIEW_SHA='5058426e9dccaaf210eb2969a8623a31123f13dfefaa43c57e975e5800819371'


def load_stream_helper(source):
    need(sha(source)==STREAM_HELPER_SHA,'independently reviewed stream helper differs')
    module=types.ModuleType('pinned_token_stream_transport')
    module.__file__=str(LAUNCH/'stream_custody.py')
    exec(compile(source,module.__file__,'exec'),vars(module))
    return module


def stream_request(before):
    roots={label:path for label,path,flag in (
        ('launch',ROOT,'root_present'),('token01',FAILED_TOKEN_ROOT,'failed_token_present'),('common',COMMON_ROOT,'common_present'),('failed01',FAILED_ROOT,'failed_present')) if before[flag]}
    singles={label:path for label,path,flag in (
        ('namespace.json',REMOTE_NAMESPACE,'namespace_present'),('original-namespace.json',OLD_REMOTE_NAMESPACE,'original_namespace_present')) if before[flag]}
    return dict(roots=roots,single_files=singles,
        inventory=dict(files=before['files'],directories=sorted(before['directories'])))


def compare_observations(before,after):
    # Same complete comparisons as the original collector, before materialization.
    need(after['files']==before['files'] and sorted(after['directories'])==sorted(before['directories'])
         and after['failed_token_present']==before['failed_token_present']
         and after['common_present']==before['common_present'] and after['failed_present']==before['failed_present']
         and after['root_present']==before['root_present'] and after['namespace_present']==before['namespace_present']
         and after['original_namespace_present']==before['original_namespace_present']
         and after['original_hardlink_groups']==before['original_hardlink_groups']
         and after['parent_accounting']==before['parent_accounting']
         and after['journal_pins']==before['journal_pins']
         and after['upstream_pins']==before['upstream_pins'] and after['upstream_errors']==before['upstream_errors']
         and normalize(after['related_accounting'])==normalize(before['related_accounting']),
         'second remote bytes/journals/terminal accounting differ')


def mirror_attempt(source,destination,copy_file):
    """Preserve every archive/argv/partial receipt as new independent bytes, never overwrite."""
    source,destination=Path(source),Path(destination)
    if not source.exists():return
    need(not source.is_symlink(),'transport attempt symlink forbidden')
    destination.mkdir(mode=0o700,exist_ok=False)
    for path in sorted(source.rglob('*')):
        need(not path.is_symlink(),'transport evidence symlink forbidden')
        target=destination/path.relative_to(source)
        if path.is_dir():target.mkdir(mode=0o700)
        else:copy_file(path,target)
    need(inventory(source)==inventory(destination),'whole transport archive/receipt mirror differs')


def stream_transfer(before,*,observe,save,helper,helper_source,external,local,copy_file,coordinator_bytes=0):
    """Transport only. Caller retains terminal/locks/full downstream acceptance gates."""
    need(sha(helper_source)==STREAM_HELPER_SHA,'stream source pin differs')
    external,local=Path(external),Path(local)
    need(external.is_dir() and local.is_dir() and external!=local,'two previously exclusively claimed capsules required')
    need(not any(os.path.lexists(p/'remote') for p in (external,local)),
         'raw materialization roots must remain absent; never remove/replace a precreated root')
    request=raw(stream_request(before));expected=helper.request_check(decode(request));bound=helper.archive_bound(expected)
    size=sum(v['bytes'] for v in before['files'].values())
    # Conservatively budget both copies AND both retained full archives on each filesystem.
    for parent in (external,local):
        fs=os.statvfs(parent)
        need(fs.f_bavail*fs.f_frsize>2*size+2*bound+coordinator_bytes+2*1024**3,'insufficient complete stream custody space')
    command=['bash',HELPER,'run',shlex.join(['python3','-B','-c',helper_source.decode(),'emit','--request-sha256',sha(request)])]
    save('stream-request.json',decode(request));save('stream-command.json',command)
    save('stream-before.json',before)
    for dest in (external,local):write(dest/'stream-emitter.py',helper_source)
    save('stream-intent.json',dict(helper_sha256=STREAM_HELPER_SHA,helper_review_sha256=STREAM_REVIEW_SHA,
        request_sha256=sha(request),command_sha256=sha(raw(command)),archive_bound_bytes=bound,
        transport_timeout_seconds=1800,copy_roots=[str(external/'remote'),str(local/'remote')],
        acceptance=False,retry_allowed=False))
    attempt=external/'stream-attempt';mirror=local/'stream-attempt';original_error=None
    try:
        helper.capture(command,request,attempt,1800)
        capture_pin=fingerprint(attempt/'capture.json',sole_link=True)['sha256']
        after=observe('inventory_after_transport.json')
        compare_observations(before,after)  # Includes journals/queue gate/upstream/accounting, not only bytes.
        save('stream-after.json',after)
        result=helper.materialize(attempt,capture_pin,external/'stream-before.json',sha(raw(before)),
            external/'stream-after.json',sha(raw(after)),external/'remote',local/'remote')
        need(result['status']=='transport_verified_pending_collector_gates' and result['acceptance'] is False,
             'transport does not establish acceptance')
        actual=inventory(external/'remote')
        need(actual==inventory(local/'remote')==dict(files=before['files'],directories=sorted(before['directories'])),
             'whole raw namespace/empty-directory two-copy custody differs')
        save('stream-transport.json',dict(helper_sha256=STREAM_HELPER_SHA,helper_review_sha256=STREAM_REVIEW_SHA,
            capture_sha256=capture_pin,materialization_sha256=fingerprint(attempt/'materialization.json',sole_link=True)['sha256'],
            archive=fingerprint(attempt/'archive.tar',sole_link=True),acceptance=False))
        return actual,after
    except BaseException as exc:
        original_error=exc
        raise
    finally:
        try:mirror_attempt(attempt,mirror,copy_file)
        except BaseException as secondary:
            try:save('stream-mirror-failure.json',dict(type=type(secondary).__name__,message=str(secondary),
                original_error=None if original_error is None else dict(type=type(original_error).__name__,message=str(original_error)),
                original_archive_preserved=True,acceptance=False,retry_allowed=False))
            except BaseException:pass
            if original_error is None:raise


def preserved_roots_equal(actual,preserved,directories):
    errors=[]
    for label,expected in preserved.items():
        observed={n[len(label)+1:]:v for n,v in actual['files'].items() if n.startswith(label+'/')}
        dirs=sorted(n[len(label)+1:] for n in actual['directories'] if n.startswith(label+'/'))
        if observed!=expected or dirs!=sorted(directories[label]):errors.append('complete immutable '+label+' root changed/missing')
    return errors


def accounting_extension(observation,baseline,job,intent_pin,result_pin):
    prior=[r for r in observation['related_accounting'] if r[0]!=job]
    current=[r for r in observation['related_accounting'] if r[0]==job]
    rows_ok=(len(prior)==43 and len(current)==1 and len(observation['related_accounting'])==44
        and normalize(prior)==normalize(baseline['related_accounting'])
        and normalize(current)==normalize([observation['parent_accounting']])
        and all(r[2].split()[0] in TERMINAL for r in observation['related_accounting']))
    journals={**baseline['journal_pins'],ROOT+'/submission.intent.json':intent_pin,ROOT+'/submission.result.json':result_pin}
    return rows_ok,(len(baseline['journal_pins'])==88 and len(journals)==90 and observation['journal_pins']==journals)


def cost(rows):
    cpu=gpu=0
    for row in normalize(rows):
        t=dict(row[7]);need(t or row[4]==row[5]==0,'unknown historical resource charge')
        g={k:v for k,v in t.items() if k.startswith('gres/')}
        need(all(v.isdigit() and (k=='gres/gpu' or k.startswith('gres/gpu:')) for k,v in g.items()),'unknown historical accelerator charge')
        typed=sum(int(v) for k,v in g.items() if k.startswith('gres/gpu:'))
        count=int(g['gres/gpu']) if 'gres/gpu' in g else typed
        need(not ('gres/gpu' in g and any(k.startswith('gres/gpu:') for k in g)) or count==typed,'inconsistent historical GPU count')
        cpu+=row[4]*row[5];gpu+=count*row[5]
    return cpu,gpu


def no_requeue_record(observation,job,root,files):
    # Current scontrol can age out; retain the worker's original captured scheduler record as fallback.
    captured=files.get('launch/preparation/allocation.json');record=observation['results']['job'];origin='terminal_scontrol'
    if record.get('returncode')!=0 and captured:
        record=decode(read_metadata(Path(root)/'allocation.json',captured['sha256']));origin='captured_allocation'
    fields=dict(re.findall(r'([^\s=]+)=(\S+)',record.get('stdout') or ''))
    okay=(record.get('returncode')==0 and fields.get('JobId')==job and fields.get('Requeue')=='0' and fields.get('Restarts')=='0')
    return dict(verified=okay,source=origin,source_sha256=None if origin=='terminal_scontrol' else captured['sha256'],fields=fields)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('job-id','packet-sha256','submission-result-sha256','local-intent-sha256','packet-commit','collector-sha256','packet-review','packet-review-sha256','collector-repository-path','collector-review','collector-review-sha256','failed-token-terminal-custody','failed-token-custody-inventory-sha256'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();need(re.fullmatch('[1-9][0-9]*',a.job_id) is not None,'exact current job ID required')
    packet_raw=read_metadata(LAUNCH/'packet.json',a.packet_sha256);packet=decode(packet_raw)
    result_raw=read_metadata(LAUNCH/'submission.result.json',a.submission_result_sha256);result=decode(result_raw)
    need(re.fullmatch('[0-9a-f]{40}',a.packet_commit) is not None,'full eventual packet commit required')
    need(sha(Path(__file__).read_bytes())==a.collector_sha256,'independently reviewed collector source pin required')
    need(packet['schema']=='ace-scientific-tokenization-launch-v1' and packet['remote_root']==ROOT
         and result.get('job_id')==a.job_id and result.get('returncode')==0 and result.get('ambiguous') is False
         and pin(result.get('intent_sha256')),'unambiguous original packet/job/result required')
    intent_raw=read_metadata(LAUNCH/'submission.intent.json',a.local_intent_sha256);intent=decode(intent_raw)
    pins={**packet['new_input_files'],'packet.json':a.packet_sha256}
    collector_path=Path(a.collector_repository_path)
    need(not collector_path.is_absolute() and '..' not in collector_path.parts and collector_path.suffix=='.py',
         'safe explicit standalone collector repository path required')
    own=read_metadata(Path(__file__),a.collector_sha256)
    git=subprocess.run(['git','--no-replace-objects','show',a.packet_commit+':'+collector_path.as_posix()],cwd=REPO,capture_output=True,timeout=20)
    need(git.returncode==0 and git.stdout==own,'independent collector/full packet commit differs')
    collector_review=Path(a.collector_review).absolute();cr=read_metadata(collector_review,a.collector_review_sha256)
    git=subprocess.run(['git','--no-replace-objects','show',a.packet_commit+':'+str(collector_review.relative_to(REPO))],cwd=REPO,capture_output=True,timeout=20)
    need(cr.strip() and git.returncode==0 and git.stdout==cr,'independent collector review must match full packet commit')
    need(pins.get('stream_custody.py')==STREAM_HELPER_SHA and pins.get('stream_custody_review.md')==STREAM_REVIEW_SHA,
         'independent accepted helper/review must be committed root packet members')
    need(intent['packet_sha256']==a.packet_sha256 and intent['remote_root']==ROOT and intent['pins']==pins
         and intent['packet_commit']==a.packet_commit,'current local submission lineage differs')
    need(all(type(n) is str and not Path(n).is_absolute() and '..' not in Path(n).parts and pin(h) for n,h in pins.items()),
         'safe complete input pin roster required')
    # Verify immutable local packet/Git bytes without loading any project/author callable.
    for n,h in pins.items():
        data=read_metadata(LAUNCH/n,h)
        git=subprocess.run(['git','--no-replace-objects','show',a.packet_commit+':'+str((LAUNCH/n).relative_to(REPO))],
                           cwd=REPO,capture_output=True,timeout=20)
        need(git.returncode==0 and git.stdout==data==read_metadata(LAUNCH_EXT/n,h),'committed packet mirrors differ')
    review=Path(a.packet_review).absolute();review_raw=read_metadata(review,a.packet_review_sha256)
    git=subprocess.run(['git','--no-replace-objects','show',a.packet_commit+':'+str(review.relative_to(REPO))],
                       cwd=REPO,capture_output=True,timeout=20)
    need(review_raw.strip() and git.returncode==0 and git.stdout==review_raw and intent['review_sha256']==a.packet_review_sha256,
         'independent original packet review/commit required')
    helper_source=read_metadata(LAUNCH/'stream_custody.py',STREAM_HELPER_SHA)
    helper_review=read_metadata(LAUNCH/'stream_custody_review.md',STREAM_REVIEW_SHA)
    builder_raw=read_metadata(LAUNCH/'build_packet.py',BUILDER_SHA)
    need(pins['build_packet.py']==BUILDER_SHA,'exact independently reviewed TOKEN02 core required')
    b=types.ModuleType('captured_token02_metadata');b.__file__=str(LAUNCH/'build_packet.py')
    exec(compile(builder_raw,b.__file__,'exec'),vars(b))
    supplied={n:read_metadata(LAUNCH/b.metadata_name(n),pins[b.metadata_name(n)]) for n in b.METADATA}
    ctx=b.admit(supplied)  # stdlib metadata-only failure/history/amendment checks; never builder main.
    m=ctx['metadata'];baseline=m['token-failure-baseline']
    need(packet['baseline_sha256']==intent['baseline_sha256']==pins['token-failure-baseline.json']
         and packet['token_recovery']==intent['token_recovery']==ctx['token_recovery']
         and packet['resources']==intent['resources']==b.RESOURCES
         and packet['deadlines']==intent['deadlines']==b.DEADLINES
         and packet['source_commit']==intent['source_commit']==b.SOURCE_COMMIT,
         'exact independently admitted TOKEN02 current baseline/authority/profile required')
    stage=decode(read_metadata(LAUNCH/'inputs/packet.json',pins['inputs/packet.json']))
    need(packet['token_science_equality']==intent['token_science_equality']==b.token_science_equality(m['failed-token-stage'],stage),
         'original complete child/source/data equality differs')
    accepted=m['realization-acceptance'];realization_result=m['realization-result'];upstream=m['realization-disposition']
    collection=m['collection-acceptance'];disposition=m['collection-disposition'];collection_result=m['collection-result']
    need(packet['realization_job_id']==accepted['job_id'] and packet['collection_job_id']==collection['job_id']
         and a.job_id not in (accepted['job_id'],collection['job_id'],FAILED_TOKEN_JOB,'35407428'),
         'distinct actual TOKEN02 job required')
    need(packet['collection_acceptance_sha256']==pins['collection-acceptance.json']
         and packet['operational_amendment_sha256']==intent['operational_amendment_sha256']==pins['operational-amendment.json']
         and packet['operational_adoption_review_sha256']==intent['operational_adoption_review_sha256']==pins['operational-adoption-review.md']
         and packet['collection_failure_acceptance_sha256']==intent['collection_failure_acceptance_sha256']==pins['collection-failure-acceptance.json']
         and packet['no_response_proof_sha256']==intent['no_response_proof_sha256']==pins['no-response-proof.json'],
         'unchanged collection/operational predecessor bindings required')
    original_files=ctx['failed_token']['files'];preserved={};preserved_directories={}
    for label,old_label in [('token01','launch'),('common','common'),('failed01','failed01')]:
        prefix=old_label+'/'
        preserved[label]={n[len(prefix):]:v for n,v in original_files.items() if n.startswith(prefix)}
        preserved_directories[label]=[n[len(prefix):] for n in ctx['failed_token']['directories'] if n.startswith(prefix)]
        need(preserved[label],'complete immutable failed/prior root required')
    # Preserve the full independently pinned original capsule: archive, both observations, logs and failures.
    need(a.failed_token_custody_inventory_sha256==pins['token-failure-custody-inventory.json'],
         'explicit original TOKEN01 custody pin must match adopted failure evidence')
    prior_custody=Path(a.failed_token_terminal_custody).resolve(strict=True)
    prior_inventory=inventory(prior_custody)
    custody=decode(read_metadata(prior_custody/'custody-inventory.json',a.failed_token_custody_inventory_sha256))
    view=dict(files={n:v for n,v in prior_inventory['files'].items() if n!='custody-inventory.json'},directories=prior_inventory['directories'])
    matches=[v for k,v in custody['copies'].items() if Path(k).resolve()==prior_custody]
    need(custody['identical'] is True and len(matches)==1 and matches[0]==view,
         'complete original failure capsule does not match independently pinned two-copy inventory')
    need(inventory(prior_custody/'remote')==dict(files=original_files,directories=sorted(ctx['failed_token']['directories'])),
         'original full failure raw capsule differs')
    for name,key in [('disposition.json','token-failure-disposition.json'),('inventory_transport.json','token-failure-inventory.json'),
                     ('inventory_after_transport.json','token-failure-readback.json')]:
        need(prior_inventory['files'][name]['sha256']==pins[key],'original failure capsule metadata differs')
    upstream_pins={REALIZATION_ROOT+'/packet.json':pins['realization-packet.json'],
        REALIZATION_ROOT+'/submission.result.json':pins['realization-result.json'],
        REALIZATION_ROOT+'/submission.intent.json':realization_result['intent_sha256']}
    prefix='launch/realization/session/worker/task/'
    for name in ('terminal.json',*[f'world-{i:03d}.json' for i in range(32)]):
        upstream_pins[REALIZATION_ROOT+'/realization/session/worker/task/'+name]=upstream['remote_files'][prefix+name]['sha256']
    need(upstream_pins[REALIZATION_ROOT+'/realization/session/worker/task/terminal.json']==accepted['realization_terminal_sha256'],
         'original realization terminal binding differs')
    recovery_bindings=dict(recovery_authority_sha256=pins['recovery-authority.json'],original_namespace_claim_sha256=OLD_CLAIM_SHA)
    need(intent['realization_acceptance_sha256']==pins['realization-acceptance.json']
         and intent['realization_job_id']==accepted['job_id']
         and intent['collection_acceptance_sha256']==pins['collection-acceptance.json']
         and intent['existing_namespace_claim_sha256']==packet['namespace_claim_sha256']==pins['namespace-claim.json']
         and intent['namespace_admission_sha256']==pins['namespace-admission.json']
         and all(intent.get(k)==v for k,v in recovery_bindings.items()) and intent['new_namespace_claimed'] is False,
         'current TOKEN02 intent preserves original namespace and upstream bindings')
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
        TERMINAL=TERMINAL,OLD_NAMESPACE=OLD_REMOTE_NAMESPACE,UPSTREAM_PINS=upstream_pins,COMMON_ROOT=COMMON_ROOT,FAILED_ROOT=FAILED_ROOT,FAILED_TOKEN_ROOT=FAILED_TOKEN_ROOT).items())+'\n'+REMOTE_OBSERVATION
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
            for sub in ('coordinator','namespaces'):(dest/sub).mkdir(mode=0o700)
            write(dest/'pull_terminal.py',Path(__file__).read_bytes())
            write(dest/'stream_custody.py',helper_source);write(dest/'stream_custody_review.md',helper_review)
            write(dest/'collector-review.md',cr)
        save('collector-intent.json',dict(utc=utc(),job_id=a.job_id,packet_sha256=a.packet_sha256,
            submission_result_sha256=a.submission_result_sha256,remote_root=ROOT,remote_namespace=REMOTE_NAMESPACE,
            local_namespace=str(LOCAL_NAMESPACE),collector_sha256=sha(Path(__file__).read_bytes()),
            accepted_realization_root=REALIZATION_ROOT,accepted_collection_root=COMMON_ROOT,preserved_failed_root=FAILED_ROOT,
            preserved_failed_token_root=FAILED_TOKEN_ROOT,token_recovery=ctx['token_recovery'],
            collector_repository_path=a.collector_repository_path,collector_review_sha256=a.collector_review_sha256,
            failed_token_custody_inventory_sha256=a.failed_token_custody_inventory_sha256,
            packet_commit=a.packet_commit,local_intent_sha256=a.local_intent_sha256,original_local_namespace=str(OLD_LOCAL_NAMESPACE),
            original_remote_namespace=OLD_REMOTE_NAMESPACE,recovery_bindings=recovery_bindings,
            retry_allowed=False,acceptance=False))
        shutil.copytree(prior_custody,EXT/'upstream-token01-custody',copy_function=copy_file)
        need(inventory(prior_custody)==inventory(EXT/'upstream-token01-custody')==prior_inventory,'original token01 custody changed during copy')
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
        helper=load_stream_helper(helper_source)  # Exact Git/mirror/root pins authenticated above.
        actual,after=stream_transfer(before,observe=observe,save=save,helper=helper,helper_source=helper_source,
            external=EXT,local=LOCAL,copy_file=copy_file,
            coordinator_bytes=sum(v['bytes'] for info in coordinator.values() for v in info['files'].values())+sum(v['bytes'] for v in prior_inventory['files'].values()))
        (LOCAL/'upstream-token01-custody').mkdir(mode=0o700)
        for sub in ('coordinator','upstream-token01-custody'):
            for path in (EXT/sub).iterdir():
                target=LOCAL/sub/path.name
                if path.is_dir():shutil.copytree(path,target,copy_function=copy_file)
                else:copy_file(path,target)
            need(inventory(EXT/sub)==inventory(LOCAL/sub),'second independent byte copy differs')
        errors=[]
        if inventory(prior_custody)!=prior_inventory:errors.append('original TOKEN01 capsule changed during custody')
        errors.extend(preserved_roots_equal(actual,preserved,preserved_directories))
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
        accounting_ok,journal_ok=accounting_extension(before,baseline,a.job_id,result['intent_sha256'],a.submission_result_sha256)
        if not accounting_ok:errors.append('exact43 prior plus current44th accounting row differs')
        if not journal_ok:errors.append('exact88 prior plus current90 journals differ')
        steps=before['results']['steps']
        step_parents=[r.split('|') for r in (steps.get('stdout') or '').splitlines() if r.startswith(a.job_id+'|')]
        if steps['returncode']!=0 or len(step_parents)!=1 or normalize([step_parents[0][:8]])!=normalize([before['parent_accounting']]):
            errors.append('full job/step accounting missing or allocation row disagrees')
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
        allocation['no_requeue']=no_requeue_record(before,a.job_id,EXT/'remote/launch/preparation',actual['files'])
        if not allocation['no_requeue']['verified']:errors.append('actual no-requeue/no-restart evidence missing/different')
        if after['results']['details']!=before['results']['details']:errors.append('scheduler request evidence changed between reads')
        try:
            prior_cost=cost(baseline['related_accounting']);total=cost(before['related_accounting'])
            need(prior_cost==(baseline['closed_cpu_seconds'],baseline['closed_gpu_slice_seconds']),'accepted baseline cost differs from rows')
            need(total==(prior_cost[0]+allocation['allocated_cpu_seconds'],prior_cost[1]+allocation['gpu_slice_seconds']),
                 'complete actual current/history cost differs')
            need(total[0]+3600<=57600 and total[1]<=7200,'actual complete-chain resource ceilings exceeded')
        except (ValueError,TypeError,KeyError) as exc:
            errors.append('actual cost reconciliation unresolved: '+str(exc));accounting_ok=False
        markers={n:actual['files'].get('launch/preparation/'+n) for n in (
            'terminal.json','terminal-candidate.json','failure.json','publication-interruption.json',
            'session/worker/failure.json',
            'session/worker/artifacts/terminal.json','session/worker/artifacts/failure.json',
            'session/worker/artifacts/publication-interruption.json','session/worker/publication-interruption.json')}
        complete_markers=all(markers[n] is not None for n in (
            'terminal.json','session/worker/artifacts/terminal.json'))
        reported=task_counts(EXT/'remote/launch/preparation/session/worker/artifacts',actual['files'])
        errors.extend(reported['errors'])
        outcome='terminal_custody_review_pending' if complete_markers else 'terminal_partial_outputs_preserved'
        if allocation['scheduler_state'].split()[0]!='COMPLETED' or allocation['scheduler_exit']!='0:0' or any(
                value is not None for name,value in markers.items() if 'failure.json' in name or 'interruption.json' in name):
            outcome='terminal_failed_outputs_preserved'
        if errors:outcome='terminal_custody_with_binding_disagreements'
        disposition=dict(schema='ace-scientific-tokenization-terminal-custody-v1',utc=utc(),status=outcome,
            job_id=a.job_id,packet_sha256=a.packet_sha256,submission_result_sha256=a.submission_result_sha256,
            custody_bytes_verified=True,current_lineage_verified=not errors,errors=errors,input_pin_mismatches=mismatches,
            remote_root=ROOT,output_root=ROOT+'/preparation',remote_files=actual['files'],remote_directories=actual['directories'],
            raw_files=len(actual['files']),raw_bytes=size,original_hardlink_groups=before['original_hardlink_groups'],
            hardlink_custody='All original names copied to separate regular files; original alias metadata retained, no alias dereference/omission.',
            namespace_local_sha256=None if claim_raw is None else sha(claim_raw),
            namespace_remote_sha256=actual['files'].get('namespace.json',{}).get('sha256'),
            recovery_bindings=recovery_bindings,realization_acceptance_sha256=pins['realization-acceptance.json'],
            collection_acceptance_sha256=pins['collection-acceptance.json'],collection_disposition_sha256=pins['collection-disposition.json'],
            collection_failure_acceptance_sha256=pins['collection-failure-acceptance.json'],
            operational_amendment_sha256=pins['operational-amendment.json'],
            no_response_proof_sha256=pins['no-response-proof.json'],preserved_upstream_roots=preserved,preserved_upstream_directories=preserved_directories,
            token_recovery=ctx['token_recovery'],failed_token_root=FAILED_TOKEN_ROOT,
            failed_token_custody_inventory_sha256=a.failed_token_custody_inventory_sha256,
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
            reported_task_counts=reported,planned_training_worlds=32,planned_records=160,planned_pairs=3584,development_payloads_read=0,
            public_task_terminal_decoded=reported['present'],scientific_payloads_decoded=False,
            scientific_qualified=False,tokenization_accepted=False,training_execution_qualified=False,acceptance=False,retry_allowed=False,
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
            partial_errors=[]
            if len(claimed)==2:
                for sub in ('remote','coordinator','upstream-token01-custody','stream-attempt'):
                    src=EXT/sub;dst=LOCAL/sub
                    if not src.exists():continue
                    try:
                        for p in sorted(src.rglob('*')):
                            need(not p.is_symlink(),'partial symlink retained only in original copy')
                            q=dst/p.relative_to(src)
                            if p.is_dir():q.mkdir(parents=True,exist_ok=True)
                            elif not q.exists():q.parent.mkdir(parents=True,exist_ok=True);copy_file(p,q)
                            else:need(fingerprint(q)==fingerprint(p),'partial mirror conflict; never overwrite')
                    except BaseException as secondary:partial_errors.append(str(secondary))
            try:save('collector_failure.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),
                retry_allowed=False,acceptance=False,partial_custody_preserved=True,partial_mirror_errors=partial_errors))
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

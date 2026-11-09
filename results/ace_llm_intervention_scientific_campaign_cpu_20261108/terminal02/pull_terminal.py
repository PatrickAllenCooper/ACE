"""Complete scientific CPU campaign terminal custody, never runtime/scientific acceptance.

Standalone sidecar. Requires independent current job/outer packet/result pins.
The original packet commit and the separate collector-source commit are explicit;
new collector/helper/review bytes never relabel the submitted packet or intent.
Preserves the complete raw roots and namespace claims in a verified CURC /projects
backup. Local capsules contain selected evidence and verification records only.
Original local predecessor capsules remain pinned in place without recursive copies.
Only allowlisted public scalar/count evidence is decoded after custody; outcomes and tensors stay opaque.
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
import math
import types

REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LAUNCH=REPO/'results/ace_llm_intervention_scientific_campaign_cpu_20261108/launch01'
LAUNCH_EXT=BASE/'ACE-LLM-intervention-scientific-campaign-cpu-20261108-01'
LOCAL=LAUNCH.parent/'terminal02'
EXT=BASE/'ACE-LLM-intervention-scientific-campaign-cpu-terminal-20261108-02'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01'
TRAIN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01'
CPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_02'
FAILED_TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'
COMMON_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_02'
FAILED_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01'
REALIZATION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02'
LOCAL_NAMESPACE=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json'
REMOTE_NAMESPACE=str(Path(ROOT).parent/'.ace-scientific-screen-20261107.namespace-recovery02.json')
OLD_LOCAL_NAMESPACE=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
OLD_REMOTE_NAMESPACE=str(Path(ROOT).parent/'.ace-scientific-screen-20261107.namespace.json')
OLD_CLAIM_SHA='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
BUILDER_SHA='3893a30ec2e652dac78529602564a4a295ddf9258461530ec7c74557df809025'
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


# Observed GPU training custody seal:72,434,765 bytes; not an acceptance claim.
# Only complete independently pinned upstream seals receive the128MiB bound.
UPSTREAM_CUSTODY_SEAL_MAX_BYTES=128*1024**2


def read_upstream_custody_seal(path,expected):
    path=Path(path)
    need(path.name=='custody-inventory.json' and pin(expected),'independently pinned upstream custody seal required')
    meta=fingerprint(path,sole_link=True)
    need(meta['bytes']<=UPSTREAM_CUSTODY_SEAL_MAX_BYTES,'bounded128MiB upstream custody seal required')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        info=os.fstat(f.fileno())
        need(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'regular independent upstream custody seal required')
        data=f.read(UPSTREAM_CUSTODY_SEAL_MAX_BYTES+1)
    need(len(data)==meta['bytes'] and len(data)<=UPSTREAM_CUSTODY_SEAL_MAX_BYTES
         and sha(data)==meta['sha256']==expected,'upstream custody seal changed or independent pin differs')
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


def require_collector_commit(revision,path,expected):
    """Authenticate new sidecar bytes independently of the original packet commit."""
    need(type(revision) is str and re.fullmatch('[0-9a-f]{40}',revision) is not None,
         'full collector source commit required')
    path=Path(path)
    need(not path.is_absolute() and '..' not in path.parts,'safe collector provenance path required')
    kind=subprocess.run(['git','--no-replace-objects','cat-file','-t',revision],
                        cwd=REPO,capture_output=True,timeout=20)
    need(kind.returncode==0 and kind.stdout.strip()==b'commit','collector source object must be a commit')
    git=subprocess.run(['git','--no-replace-objects','show',revision+':'+path.as_posix()],
                       cwd=REPO,capture_output=True,timeout=20)
    need(git.returncode==0 and git.stdout==expected,'independently pinned collector source commit bytes differ: '+str(path))


REMOTE_OBSERVATION=r'''
from pathlib import Path
import subprocess,hashlib,json,datetime,stat,os,signal,sys,time
class RemoteObservationDeadline(BaseException):
 pass
def remote_deadline(signum,frame):
 raise RemoteObservationDeadline('absolute remote metadata deadline')
signal.signal(signal.SIGALRM,remote_deadline);signal.alarm(1200)
_FINGERPRINT_COUNT=0;_OBSERVATION_STARTED=time.monotonic()
root=Path(ROOT);namespace=Path(NAMESPACE);original_namespace=Path(OLD_NAMESPACE)
def need(ok,message):
 if not ok:raise ValueError(message)
def fingerprint(path):
 global _FINGERPRINT_COUNT
 need(not path.is_symlink(),'remote symlink forbidden')
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode),'nonregular remote file')
  h=hashlib.sha256()
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
  after=os.fstat(f.fileno())
 identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
 need(identity(before)==identity(after)==identity(path.lstat()),'remote file changed during capture')
 _FINGERPRINT_COUNT+=1
 if _FINGERPRINT_COUNT%1000==0:print(json.dumps(dict(phase='terminal_inventory_fingerprint',files_read=_FINGERPRINT_COUNT,elapsed_seconds=time.monotonic()-_OBSERVATION_STARTED)),file=sys.stderr,flush=True)
 return dict(bytes=after.st_size,sha256=h.hexdigest()),after
v=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results={},files={},directories=[],
       root_present=root.exists(),training_present=Path(TRAIN_ROOT).exists(),cpu_present=Path(CPU_ROOT).exists(),failed_token_present=Path(FAILED_TOKEN_ROOT).exists(),common_present=Path(COMMON_ROOT).exists(),failed_present=Path(FAILED_ROOT).exists(),namespace_present=os.path.lexists(namespace),original_hardlink_groups=[],
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
 for label,tree,present in (('launch',root,v['root_present']),('training',Path(TRAIN_ROOT),v['training_present']),('cpu',Path(CPU_ROOT),v['cpu_present']),('failed-token01',Path(FAILED_TOKEN_ROOT),v['failed_token_present']),('common',Path(COMMON_ROOT),v['common_present']),('failed01',Path(FAILED_ROOT),v['failed_present'])):
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
 # Explicit original pinned paths, including24 supplied development descriptor bytes; never enumerate realization or decode payloads.
 for path in UPSTREAM_PINS:
  try:v['upstream_pins'][path]=fingerprint(Path(path))[0]['sha256']
  except (OSError,ValueError) as exc:v['upstream_errors'][path]=dict(type=type(exc).__name__,message=str(exc))

except BaseException as exc:
 v['observation_error']=dict(type=type(exc).__name__,message=str(exc))
print(json.dumps(v,sort_keys=True))
'''


def tres(text):
    pairs=[p.split('=',1) for p in text.split(',') if p]
    need(all(len(p)==2 and all(p) for p in pairs) and len(dict(pairs))==len(pairs),'invalid/duplicate allocation TRES')
    return dict(pairs)


def gpu_count(t,cpus,elapsed):
    if not t:return 0 if cpus==elapsed==0 else None
    g={k:v for k,v in t.items() if k.startswith('gres/')}
    if any(not v.isdigit() or not (k=='gres/gpu' or k.startswith('gres/gpu:')) for k,v in g.items()):return None
    typed=[int(v) for k,v in g.items() if k.startswith('gres/gpu:')]
    aggregate=int(g['gres/gpu']) if 'gres/gpu' in g else None
    if aggregate is not None and typed and aggregate!=sum(typed):return None
    return aggregate if aggregate is not None else sum(typed)


def cost(rows):
    cpu=gpu=0
    for row in normalize(rows):
        count=gpu_count(dict(row[7]),row[4],row[5]);need(count is not None,'unknown historical GPU charge')
        cpu+=row[4]*row[5];gpu+=count*row[5]
    return cpu,gpu


def public_fields(data,allowed):
    """Select top-level JSON values without decoding any other values (notably scores).

    Unknown values are scanned lexically for object boundaries only. Original bytes
    remain authoritative; this does not validate or interpret their contents.
    """
    text=data.decode('utf-8');decoder=json.JSONDecoder();i=0;seen=set();result={}
    def ws(pos):
        while pos<len(text) and text[pos].isspace():pos+=1
        return pos
    i=ws(i);need(i<len(text) and text[i]=='{','public evidence must be a JSON object');i+=1
    while True:
        i=ws(i)
        if i<len(text) and text[i]=='}':i+=1;break
        key,end=decoder.raw_decode(text,i);need(type(key) is str and key not in seen,'duplicate/nonstring public key');seen.add(key)
        i=ws(end);need(i<len(text) and text[i]==':','public evidence colon required');start=ws(i+1)
        j=start;stack=[];quoted=False;escaped=False
        while j<len(text):
            c=text[j]
            if quoted:
                if escaped:escaped=False
                elif c=='\\':escaped=True
                elif c=='"':quoted=False
            elif c=='"':quoted=True
            elif c in '[{':stack.append(c)
            elif c in ']}':
                if not stack:break
                need((stack.pop(),c) in (('[',']'),('{','}')),'unbalanced public evidence')
            elif c==',' and not stack:break
            j+=1
        need(j>start and not quoted and not stack,'incomplete public value')
        if key in allowed:result[key]=decode(text[start:j])
        i=ws(j);need(i<len(text) and text[i] in ',}','public value delimiter required')
        if text[i]=='}':i+=1;break
        i=ws(i+1);need(i<len(text) and text[i]!='}','trailing public comma')
    need(ws(i)==len(text),'trailing public evidence')
    return result


RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,
    memory_mib=7680,gpus=0,requeue=False,wall_seconds=6000)
DEADLINES=dict(startup=180,work=5940,hard=5980)
POLICIES=('uniform_random','menu_coverage','local_information','target_risk','learned_ranker','frozen_llm','sft_llm','dpo_llm')


def cpu_tres(t):
    return (t.get('cpu')=='2' and t.get('node')=='1' and t.get('mem') in ('7680M','7.5G','7.50G')
        and not any(k.startswith('gres/') for k in t))


def resources(observation,job,output,files):
    row=observation['parent_accounting'];normalized=normalize([row])[0];errors=[]
    cpus,elapsed=normalized[4:6];allocated=dict(normalized[7]);gpu=gpu_count(allocated,cpus,elapsed)
    details=observation['results']['details'];requested={};request_identity=False
    if details['returncode']==0:
        rows=[r.split('|') for r in details['stdout'].splitlines() if r.strip()]
        if len(rows)==1 and len(rows[0])>=6:
            r=rows[0]
            try:requested=tres(r[4])
            except ValueError as exc:errors.append(str(exc))
            request_identity=r[:4]==[job,'acpu','cpu-normal','100'] and r[5]=='paco0228'
    fields={};allocation_pin=files.get('launch/cpu/allocation.json',{}).get('sha256')
    if allocation_pin:
        try:
            record=decode(read_metadata(output/'allocation.json',allocation_pin))
            need(record['returncode']==0,'saved allocation command failed')
            fields=dict(re.findall(r'([^\s=]+)=(\S+)',record['stdout']))
        except (ValueError,KeyError,OSError) as exc:errors.append('saved allocation malformed: '+str(exc))
    if not fields and observation['results']['job']['returncode']==0:
        fields=dict(re.findall(r'([^\s=]+)=(\S+)',observation['results']['job']['stdout']))
    no_requeue=fields.get('JobId')==job and fields.get('Requeue')=='0' and fields.get('Restarts')=='0'
    matched=(request_identity and cpu_tres(requested) and cpu_tres(allocated) and no_requeue
        and normalized[1]=='ace_llm_scientific_campaign_cpu01' and normalized[6]=='ucb736_asc2'
        and cpus==2 and gpu==0 and 0<=elapsed<=6000)
    if not matched:errors.append('CPU campaign identity/request/allocation/no-requeue/wall differs or unavailable')
    if gpu is None:errors.append('unknown accelerator charge; never infer zero')
    return dict(resource_contract_matched=matched,errors=errors,expected=RESOURCES,requested_tres=requested,
        allocated_tres=allocated,allocated_cpus=cpus,elapsed_seconds=elapsed,allocated_cpu_seconds=cpus*elapsed,
        gpu_count=gpu,gpu_slice_seconds=None if gpu is None else gpu*elapsed,
        scheduler_state=row[2],scheduler_exit=row[3],account=row[6],alloc_tres=row[7],
        no_requeue_verified=no_requeue,allocation_fields=fields,allocation_source_sha256=allocation_pin)


def charged_journal(path,expected_pin):
    """Recorded charge/response counts only; never decode values, targets or actions."""
    before=fingerprint(path,sole_link=True);need(before['sha256']==expected_pin,'journal byte pin differs')
    reserves={};responses=set();known_failed_returns=set();failure_events=0;errors=[];lines=0
    with Path(path).open('rb') as stream:
        while True:
            raw_line=stream.readline(8*1024**2+1)
            if not raw_line:break
            lines+=1
            if len(raw_line)>8*1024**2 or lines>10000:
                errors.append('bounded public journal parsing exceeded; remaining charges unknown');break
            try:
                need(raw_line.endswith(b'\n'),'partial journal line; charges unresolved')
                row=public_fields(raw_line,{'event','request_id','cost_units','response_returned_to_driver'})
                event=row.get('event');need(type(event) is str,'journal event absent')
                if event not in ('reserve','response','failure'):continue
                rid=row['request_id'];need(type(rid) is int and rid>=0,'public request id malformed')
                if event=='reserve':
                    units=row['cost_units'];need(type(units) is int and units>0 and rid not in reserves,'malformed/duplicate charge')
                    reserves[rid]=units
                elif event=='response':
                    need(rid in reserves and rid not in responses and rid not in known_failed_returns
                         and type(row['cost_units']) is int and row['cost_units']==reserves[rid], 'unjoined or duplicate response charge')
                    responses.add(rid)
                else:
                    need(rid in reserves and rid not in responses and rid not in known_failed_returns,'unjoined/duplicate failure')
                    failure_events+=1
                    if row.get('response_returned_to_driver') is True:known_failed_returns.add(rid)
            except (ValueError,TypeError,KeyError,UnicodeError) as exc:errors.append('line '+str(lines)+': '+str(exc))
    need(fingerprint(path,sole_link=True)==before,'captured journal changed during public count')
    return dict(sha256=expected_pin,lines_observed=lines,recorded_reserved_requests=len(reserves),
        known_reserved_units_lower_bound=sum(reserves.values()),recorded_valid_responses=len(responses),
        recorded_failure_events=failure_events,recorded_failed_returns=len(known_failed_returns),
        unresolved_reservations=len(set(reserves)-responses-known_failed_returns),errors=errors,
        ledger_fully_validated=False,private_response_values_decoded=False)


def public_cpu_evidence(output,files):
    """Reported fits and charged journals, after custody; never numeric/utility acceptance."""
    prefix='launch/cpu/';selected={};errors=[]
    allowed={'schema','status','scope','job_id','packet_sha256','admission_sha256','runtime_stage_sha256','stage_sha256',
        'phase_terminal_sha256','campaign_sha256','plan_sha256','exposure_sha256','numeric_training_updates','completed_cpu_cells',
        'declared_cells','llm_training_updates','llm_model_or_adapter_loaded','tokenizer_calls','worlds_generated','gpu_services_started',
        'private_scores_computed','scientific_qualified','retry_allowed','external_acceptance_required','independent_acceptance_required',
        'completed_steps','sft_steps','dpo_steps','environment_responses','training_receipt_sha256','elapsed_seconds','supervisor_cpu_seconds',
        'returncode','group_still_exists','term_sent','kill_sent','pid','pgid','phase','policy_id','monotonic_seconds','training_updates'}
    names=['supervisor.json','worker.cleanup.json','worker.resources.json','worker.started.json','worker/artifacts/terminal.json',
        'worker/artifacts/cpu-phase/terminal.json','worker/artifacts/cpu-phase/ranker/terminal.json']
    names+=sorted(n[len(prefix):] for n in files if n.startswith(prefix) and re.fullmatch(r'worker/artifacts/phase-[^/]+\.json',n[len(prefix):]))
    need(len(names)<10000,'public metadata roster exceeds bound')
    for name in names:
        item=files.get(prefix+name)
        if item is None:continue
        try:
            fields=public_fields(read_metadata(output/name,item['sha256']),allowed)
            need(all(type(v) in (str,int,float,bool,type(None)) for v in fields.values()),'allowlisted public field must be scalar')
            selected[name]=dict(sha256=item['sha256'],reported=fields)
        except (ValueError,KeyError,TypeError,OSError,UnicodeError) as exc:errors.append(name+': '+str(exc))
    relative='worker/artifacts/campaign/'
    expected=[relative+f'policy-{p:02d}/world-{w:02d}/episode/journal.jsonl' for p in range(5) for w in range(24)]
    counts={};missing=[]
    for name in expected:
        item=files.get(prefix+name)
        if item is None:missing.append(name);continue
        try:counts[name]=charged_journal(output/name,item['sha256'])
        except (ValueError,TypeError,KeyError,OSError) as exc:errors.append(name+': '+str(exc))
    observed=[n[len(prefix):] for n in files if n.startswith(prefix+relative) and n.endswith('/episode/journal.jsonl')]
    unexpected=sorted(set(observed)-set(expected))
    if unexpected:errors.append('unexpected policy/world journals retained; excluded from authorized-five-arm count and not inferred zero')
    return dict(acceptance=False,reported_metadata=selected,charge_journals=counts,missing_expected_journals=missing,
        unexpected_journals=unexpected,public_count_errors=errors,planned_cpu_cells=120,declared_campaign_cells=192,
        authorized_cpu_policies=list(POLICIES[:5]),observed_charge_journals=len(counts),
        recorded_reserved_requests=sum(c['recorded_reserved_requests'] for c in counts.values()),
        known_reserved_units_lower_bound=sum(c['known_reserved_units_lower_bound'] for c in counts.values()),
        recorded_valid_responses=sum(c['recorded_valid_responses'] for c in counts.values()),
        recorded_failure_events=sum(c['recorded_failure_events'] for c in counts.values()),
        unresolved_reservations=sum(c['unresolved_reservations'] for c in counts.values()),
        total_campaign_response_cost_units=None,complete_response_accounting_qualified=False,
        accounting_note='Counts describe only durable original reserve/response/failure records. Missing, malformed or partial journals never imply zero work; full independent charged-accounting audit remains required.',
        collector_new_scientific_responses=0,collector_training_updates=0,private_outcomes_decoded=False,checkpoint_tensors_decoded=False,
        training_qualified=False,scientific_qualified=False)


def failed_token_label(files,packet_pin):
    labels=[n.split('/')[0] for n,v in files.items() if n.count('/')==1 and n.endswith('/packet.json') and v['sha256']==packet_pin]
    need(len(labels)==1,'unique original failed TOKEN01 raw root required')
    return labels[0]


def accounting_extension(observation,baseline,job,intent_pin,result_pin):
    prior=[r for r in observation['related_accounting'] if r[0]!=job]
    current=[r for r in observation['related_accounting'] if r[0]==job]
    rows_ok=(len(prior)==45 and len(current)==1 and len(observation['related_accounting'])==46
        and normalize(prior)==normalize(baseline['related_accounting'])
        and normalize(current)==normalize([observation['parent_accounting']])
        and all(r[2].split()[0] in TERMINAL for r in observation['related_accounting']))
    journals={**baseline['journal_pins'],ROOT+'/submission.intent.json':intent_pin,
              ROOT+'/submission.result.json':result_pin}
    journals_ok=(len(baseline['journal_pins'])==92 and len(journals)==94 and observation['journal_pins']==journals)
    return rows_ok,journals_ok


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


def validate_backup_capture(receipt,expected,selected):
    """The byte transport authenticates the archive. This joins its exact local view."""
    need(type(receipt) is dict and receipt.get('full_backup_verified') is True,
         'complete verified CURC backup required')
    need(receipt.get('full_inventory')==expected,'complete independent CURC backup differs')
    need(receipt.get('local_inventory')==selected,'compact selected byte inventory differs')
    need(set(selected)=={'files','directories'} and type(selected['files']) is dict
         and type(selected['directories']) is list,'exact selected inventory schema required')
    need(selected['directories']==sorted(set(selected['directories'])),'selected directories duplicate/unsorted')
    for name,info in selected['files'].items():
        need(type(name) is str and not Path(name).is_absolute()
             and all(p not in ('','.','..') for p in name.split('/')),'unsafe selected path')
        need(expected['files'].get(name)==info,'local selected evidence not in complete backup')
    need(set(selected['directories'])<=set(expected['directories']),'local directory outside complete backup')
    return expected


def mirror_partial_evidence(external,local,copy_file):
    """Retain secondary mirror errors without replacing the original exception."""
    errors=[]
    for sub in ('remote','coordinator','backup-operations'):
        try:
            src=external/sub;dst=local/sub
            if not src.exists():continue
            dst.mkdir(mode=0o700,exist_ok=True)
            for p in sorted(src.rglob('*')):
                need(not p.is_symlink(),'partial symlink retained only in original copy')
                q=dst/p.relative_to(src)
                if p.is_dir():q.mkdir(parents=True,exist_ok=True)
                elif not q.exists():q.parent.mkdir(parents=True,exist_ok=True);copy_file(p,q)
                else:need(fingerprint(q)==fingerprint(p),'partial mirror conflict; never overwrite')
        except BaseException as secondary:errors.append(str(secondary))
    return errors


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('job-id','packet-sha256','submission-result-sha256','local-intent-sha256','packet-commit','collector-source-commit','collector-sha256','collector-repository-path','collector-review','collector-review-sha256','training-terminal-custody','training-custody-inventory-sha256','training-acceptance-sha256','baseline-sha256','packet-review','packet-review-sha256'):ap.add_argument('--'+name,required=True)
    for name in ('backup-root','backup-helper','backup-helper-sha256','backup-helper-repository-path'):
        ap.add_argument('--'+name,required=True)
    a=ap.parse_args();need(re.fullmatch('[1-9][0-9]*',a.job_id) is not None,'exact current job ID required')
    packet_raw=read_metadata(LAUNCH/'packet.json',a.packet_sha256);packet=decode(packet_raw)
    result_raw=read_metadata(LAUNCH/'submission.result.json',a.submission_result_sha256);result=decode(result_raw)
    need(re.fullmatch('[0-9a-f]{40}',a.packet_commit) is not None,'full eventual packet commit required')
    need(sha(Path(__file__).read_bytes())==a.collector_sha256,'independently reviewed collector source pin required')
    need(packet['schema']=='ace-scientific-cpu-campaign-launch-v1' and packet['remote_root']==ROOT
         and result.get('job_id')==a.job_id and result.get('returncode')==0 and result.get('ambiguous') is False
         and pin(result.get('intent_sha256')),'unambiguous original packet/job/result required')
    intent_raw=read_metadata(LAUNCH/'submission.intent.json',a.local_intent_sha256);intent=decode(intent_raw)
    pins={**packet['new_input_files'],'packet.json':a.packet_sha256}
    collector_path=Path(a.collector_repository_path)
    need(not collector_path.is_absolute() and '..' not in collector_path.parts and collector_path.suffix=='.py',
         'explicit safe collector repository path required')
    own=read_metadata(Path(__file__),a.collector_sha256)
    require_collector_commit(a.collector_source_commit,collector_path,own)
    helper_raw=read_metadata(a.backup_helper,a.backup_helper_sha256)
    helper_path=Path(a.backup_helper_repository_path)
    need(not helper_path.is_absolute() and '..' not in helper_path.parts and helper_path.suffix=='.py', 'safe backup helper Git path required')
    require_collector_commit(a.collector_source_commit,helper_path,helper_raw)
    backup=types.ModuleType('remote_custody');backup.__file__=str(Path(a.backup_helper).resolve())
    exec(compile(helper_raw,backup.__file__,'exec'),backup.__dict__)
    backup.bind_source(helper_raw,a.backup_helper_sha256)
    collector_review=Path(a.collector_review).absolute();cr=read_metadata(collector_review,a.collector_review_sha256)
    need(cr.strip(),'independent collector review required')
    require_collector_commit(a.collector_source_commit,collector_review.relative_to(REPO),cr)
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
    # This exact reviewed stdlib coordinator performs metadata-only admission; no project imports.
    builder_raw=read_metadata(LAUNCH/'build_packet.py',BUILDER_SHA)
    need(pins['build_packet.py']==BUILDER_SHA,'reviewed history coordinator differs')
    builder=types.ModuleType('cpu_campaign_history_coordinator')
    builder.__file__=str(LAUNCH/'build_packet.py')
    exec(compile(builder_raw,builder.__file__,'exec'),builder.__dict__)
    supplied={n:read_metadata(LAUNCH/builder.metadata_name(n),pins[builder.metadata_name(n)]) for n in builder.METADATA}
    history=builder.admit(supplied)
    need(packet['token_recovery']==intent['token_recovery']==history['token_recovery']
         and packet['failed_token_root']==intent['failed_token_root']==FAILED_TOKEN_ROOT
         and packet['failed_token_job_id']==intent['failed_token_job_id']==builder.FAILED_TOKEN_JOB
         and packet['original_input_pins']==history['upstream_pins']
         and packet['preserved_directories']==history['preserved_directories'], 'current full token history binding differs')
    baseline_pin=a.baseline_sha256;need(baseline_pin==pins['baseline.json'],'independent current baseline pin differs');baseline=decode(read_metadata(LAUNCH/'baseline.json',baseline_pin))
    need(packet['baseline_sha256']==intent['baseline_sha256']==baseline_pin and baseline['terminal_jobs']==45
         and baseline['submission_journals']==92 and len(baseline['related_accounting'])==45 and len(baseline['journal_pins'])==92
         and baseline['preparation_allowance_cpu_seconds']==3600 and baseline['outstanding_reservation']==0,
         'independently pinned actual45-job/92-journal baseline required')
    normalize(baseline['related_accounting'])
    need(packet['resources']==intent['resources']==RESOURCES and packet['deadlines']==intent['deadlines']==DEADLINES,
         'exact CPU campaign resource/deadline contract required')
    need(packet['source_commit']==intent['source_commit']=='a9346022f145cc29439325b6f291597b2e6ec037'
         and pins['inputs/prepare_llm_intervention_scientific_campaign_cpu.py']=='ba439d819f49f6832c97181e4ff727c4813f29031e7b67f8085670f845b0f893'
         and pins['freeze.json']=='15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d'
         and pins['operational-amendment.json']==packet['operational_amendment_sha256']==intent['operational_amendment_sha256']
         == 'fa94858dfc19c8b15c82acde167ad8fe9991d0c863c32444d017743079617855', 'original frozen source/amendment differs')
    need(packet['policy_ids']==list(POLICIES) and packet['cpu_policy_ids']==list(POLICIES[:5]) and packet['worlds']==24 and packet['cells']==192
         and packet['cpu_output']==ROOT+'/cpu','whole192-cell/five-CPU-arm contract differs')
    accepted=decode(read_metadata(LAUNCH/'gpu-training-acceptance.json',a.training_acceptance_sha256))
    need(a.training_acceptance_sha256==pins['gpu-training-acceptance.json']==packet['training_acceptance_sha256']==intent['training_acceptance_sha256'],
         'independent original training acceptance pin differs')
    result_before=decode(read_metadata(LAUNCH/'training-result.json',pins['training-result.json']))
    upstream=decode(read_metadata(LAUNCH/'training-disposition.json',pins['training-disposition.json']))
    expected=decode(read_metadata(LAUNCH/'expected-training.json',pins['expected-training.json']))
    need(accepted['schema']=='ace-scientific-training-gpu-acceptance-v1' and accepted['status']=='scientific_training_gpu_accepted'
         and type(accepted['required_findings']) is int and accepted['required_findings']==0 and accepted['scope']=='policy_train'
         and accepted['scientific_utility_qualified'] is False and all(accepted[k]==v for k,v in expected.items())
         and all(accepted[k] is True for k in ('training_gpu_qualified','checkpoint_reload_inference_qualified','complete_custody_verified',
             'runtime_qualified','owned_process_cleanup_verified','scheduler_verified')), 'independent completed training acceptance required')
    need(accepted['job_id']==result_before['job_id']==packet['training_job_id']==intent['training_job_id']
         and result_before['returncode']==0 and result_before['ambiguous'] is False
         and upstream['job_id']==accepted['job_id'] and upstream['remote_root']==TRAIN_ROOT
         and upstream['packet_sha256']==pins['training-packet.json']
         and upstream['submission_result_sha256']==pins['training-result.json']
         and upstream['custody_bytes_verified'] is upstream['current_lineage_verified'] is True
         and upstream['errors']==[] and upstream['input_pin_mismatches']==[]
         and a.job_id not in (accepted['job_id'],'35407428'), 'current/accepted original training lineage differs')
    need(baseline['input_pins']['predecessor']==pins['previous-baseline.json']
         and baseline['input_pins']['acceptance']==a.training_acceptance_sha256
         and baseline['input_pins']['disposition']==pins['training-disposition.json'], 'actual45/92 accepted baseline provenance differs')
    transport=decode(read_metadata(LAUNCH/'training-inventory.json',pins['training-inventory.json']))
    need(transport['returncode']==0,'original accepted training inventory transport failed')
    observation=decode(transport['stdout']);original_files=observation['files']
    need(observation['terminal_gate'] is True and not observation.get('observation_error')
         and observation['parent_accounting'][0]==accepted['job_id']
         and len(original_files)==upstream['raw_files'] and sum(v['bytes'] for v in original_files.values())==upstream['raw_bytes'],
         'complete original training raw inventory required')
    need(original_files['launch/packet.json']['sha256']==pins['training-packet.json']
         and original_files['launch/submission.result.json']['sha256']==pins['training-result.json']
         and original_files['launch/submission.intent.json']['sha256']==result_before['intent_sha256'], 'original training packet/journal map differs')
    failed_label=failed_token_label(original_files,pins['failed-token-packet.json'])
    preserved={};preserved_directories={}
    for label,old_label in [('training','launch'),('cpu','cpu'),('failed-token01',failed_label),('common','common'),('failed01','failed01')]:
        prefix=old_label+'/'
        preserved_directories[label]=sorted(n[len(prefix):] for n in observation['directories'] if n.startswith(prefix))
        preserved[label]={n[len(prefix):]:v for n,v in original_files.items() if n.startswith(prefix)}
        need(preserved[label] and all(not Path(n).is_absolute() and '..' not in Path(n).parts and pin(v['sha256'])
             and type(v['bytes']) is int and v['bytes']>=0 for n,v in preserved[label].items()), 'safe complete accepted upstream root map required')
    # Reuse the reviewed terminal capsule's inventory, not a newly manufactured acceptance.
    prior_custody=Path(a.training_terminal_custody).resolve(strict=True)
    custody=decode(read_upstream_custody_seal(prior_custody/'custody-inventory.json',a.training_custody_inventory_sha256))
    prior_custody_before=inventory(prior_custody)
    actual_capsule=dict(files={n:v for n,v in prior_custody_before['files'].items() if n!='custody-inventory.json'},directories=prior_custody_before['directories'])
    matches=[v for k,v in custody['copies'].items() if Path(k).resolve()==prior_custody]
    need(custody['identical'] is True and len(matches)==1 and matches[0]==actual_capsule,'complete independently pinned original training capsule required')
    for name,h in [('disposition.json',pins['training-disposition.json']),('inventory_transport.json',pins['training-inventory.json'])]:
        need(prior_custody_before['files'][name]['sha256']==h,'training capsule metadata differs')
    need(inventory(prior_custody/'remote')['files']==original_files,'complete accepted training raw capsule differs')
    failed_capsule=prior_custody/'upstream-token01-custody'
    failed_custody=decode(read_upstream_custody_seal(failed_capsule/'custody-inventory.json',pins['token-failure-custody-inventory.json']))
    failed_view=inventory(failed_capsule)
    failed_without_receipt=dict(files={n:v for n,v in failed_view['files'].items() if n!='custody-inventory.json'},directories=failed_view['directories'])
    need(failed_custody['identical'] is True and len(failed_custody['copies'])==2
         and all(view==failed_without_receipt for view in failed_custody['copies'].values()),
         'complete original failed TOKEN01 archive/receipts/copies required inside training custody')

    # Only the core's explicit24-descriptor+terminal readback, never an enumeration of realization worlds.
    rt=decode(read_metadata(LAUNCH/'realization-terminal.json',pins['realization-terminal.json']))
    ra=decode(read_metadata(LAUNCH/'realization-acceptance.json',pins['realization-acceptance.json']))
    need(ra['status']=='accepted' and ra['complete_custody_verified'] is True and ra['required_findings']==0
         and ra['realization_terminal_sha256']==pins['realization-terminal.json'] and ra['freeze_sha256']==pins['freeze.json']
         and ra['namespace_admission_sha256']==pins['namespace-admission.json'] and len(rt['entries'])==56,
         'accepted original realization metadata required')
    realization_pins={REALIZATION_ROOT+'/realization/session/worker/task/terminal.json':pins['realization-terminal.json']}
    for i,entry in enumerate(rt['entries'][32:]):
        need(entry['world_id']==202611070100+i and entry['scope']=='development' and entry['status']=='complete'
             and entry['path']==f'world-{32+i:03d}.json' and pin(entry['file_sha256']),'exact original24 descriptor identities required')
        realization_pins[REALIZATION_ROOT+'/realization/session/worker/task/'+entry['path']]=entry['file_sha256']
        need(pins[f'inputs/descriptor-{i:03d}.json']==entry['file_sha256'],'original supplied descriptor byte pin differs')
    upstream_pins=packet['original_input_pins']
    need({p:h for p,h in upstream_pins.items() if p.startswith(REALIZATION_ROOT+'/')}==realization_pins,
         'exact original realization readback roster required')
    for p,h in upstream_pins.items():
        need(pin(h) and '..' not in Path(p).parts,'safe explicit upstream pin required')
        if p in realization_pins:continue
        matches=[(label,root) for label,root in [('training',TRAIN_ROOT),('cpu',CPU_ROOT),('failed-token01',FAILED_TOKEN_ROOT),('common',COMMON_ROOT),('failed01',FAILED_ROOT)] if p.startswith(root+'/')]
        need(len(matches)==1,'unexpected upstream readback scope')
        label,root=matches[0];need(preserved[label][p[len(root)+1:]]['sha256']==h,'original input/readback map differs')
    recovery_bindings=dict(recovery_authority_sha256=pins['recovery-authority.json'],original_namespace_claim_sha256=OLD_CLAIM_SHA)
    need(intent['existing_namespace_claim_sha256']==packet['namespace_claim_sha256']==pins['namespace-claim.json']
         and intent['namespace_admission_sha256']==pins['namespace-admission.json']
         and all(intent.get(k)==v for k,v in recovery_bindings.items()) and intent['new_namespace_claimed'] is False,
         'current CPU intent/namespace/acceptance joins differ')
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
            transport=dict(command=command,returncode=None,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr),utc=utc(),
                           retry_allowed=False,partial_custody_preserved=True)
        if source is not None:transport['python_source']=source
        save(name,transport);need(transport['returncode']==0,'failed/partial custody retained: '+name)
        return transport
    source='\n'.join(k+'='+repr(v) for k,v in dict(ROOT=ROOT,NAMESPACE=REMOTE_NAMESPACE,JOB=a.job_id,
        TERMINAL=TERMINAL,OLD_NAMESPACE=OLD_REMOTE_NAMESPACE,UPSTREAM_PINS=upstream_pins,TRAIN_ROOT=TRAIN_ROOT,COMMON_ROOT=COMMON_ROOT,FAILED_ROOT=FAILED_ROOT,CPU_ROOT=CPU_ROOT,FAILED_TOKEN_ROOT=FAILED_TOKEN_ROOT).items())+'\n'+REMOTE_OBSERVATION
    def observe(name):
        transport=run_saved(name,['bash',HELPER,'run','python3 -'],source=source,timeout=1230)
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
            write(dest/'remote_custody.py',helper_raw)
        save('collector-intent.json',dict(utc=utc(),job_id=a.job_id,packet_sha256=a.packet_sha256,
            submission_result_sha256=a.submission_result_sha256,remote_root=ROOT,remote_namespace=REMOTE_NAMESPACE,
            local_namespace=str(LOCAL_NAMESPACE),collector_sha256=sha(Path(__file__).read_bytes()),
            custody_mode='curc_projects_full_backup_with_compact_local_evidence',
            backup_root=a.backup_root,backup_helper_sha256=a.backup_helper_sha256,
            backup_helper_repository_path=a.backup_helper_repository_path,complete_local_raw_copy=False,
            token_recovery=history['token_recovery'],preserved_failed_token_root=FAILED_TOKEN_ROOT,
            accepted_training_root=TRAIN_ROOT,accepted_token_root=CPU_ROOT,accepted_collection_root=COMMON_ROOT,preserved_failed_root=FAILED_ROOT,
            collector_repository_path=a.collector_repository_path,collector_review_sha256=a.collector_review_sha256,
            training_custody_inventory_sha256=a.training_custody_inventory_sha256,
            packet_commit=a.packet_commit,collector_source_commit=a.collector_source_commit,
            remote_metadata_deadline_seconds=1200,local_observation_timeout_seconds=1230,
            local_intent_sha256=a.local_intent_sha256,original_local_namespace=str(OLD_LOCAL_NAMESPACE),
            original_remote_namespace=OLD_REMOTE_NAMESPACE,recovery_bindings=recovery_bindings,
            retry_allowed=False,acceptance=False))
        for dest in claimed:
            write(dest/'collector-review.md',cr)
        save('predecessor-reference.json',dict(path=str(prior_custody),custody_inventory_sha256=a.training_custody_inventory_sha256,
            disposition_sha256=pins['training-disposition.json'],raw_inventory_sha256=pins['training-inventory.json'],
            complete_existing_capsule_authenticated=True,recursive_copy_created=False,original_preserved=True,
            scope='Reference to existing complete accepted training custody. This compact capsule does not contain its bytes.'))
        # Keep complete local and external coordinator histories, not just their successful receipts.
        coordinator={}
        for label,original in (('local-launch',LAUNCH),('external-launch',LAUNCH_EXT)):
            before=inventory(original)
            coordinator[label]=before
            save('coordinator/'+label+'-reference.json',dict(path=str(original),inventory=before,recursive_copy_created=False,original_preserved=True))
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
        roots={label:path for label,path,present in (
            ('launch',ROOT,before['root_present']),('training',TRAIN_ROOT,before['training_present']),
            ('cpu',CPU_ROOT,before['cpu_present']),('failed-token01',FAILED_TOKEN_ROOT,before['failed_token_present']),
            ('common',COMMON_ROOT,before['common_present']),('failed01',FAILED_ROOT,before['failed_present']),
            ('namespace.json',REMOTE_NAMESPACE,before['namespace_present']),
            ('original-namespace.json',OLD_REMOTE_NAMESPACE,before['original_namespace_present'])) if present}
        selection,_=backup.selected_inventory(backup.normalized(before))
        # Two selected copies plus bounded operation/observation/capsule metadata.
        # Existing predecessor bytes are references, not new disk allocation.
        selected_bytes=sum(v['bytes'] for v in selection['files'].values())
        required_local=2*selected_bytes+32*len(raw(before))+4*len(raw(coordinator))+512*1024**2
        capacity={}
        for parent in (BASE,LOCAL.parent):
            fs=os.statvfs(parent);available=fs.f_bavail*fs.f_frsize
            need(available>required_local,'insufficient aggregate compact-custody headroom')
            capacity[str(parent)]=dict(available_bytes=available,required_increment_bytes=required_local,
                selected_bytes_per_copy=selected_bytes,device=parent.stat().st_dev)
        save('local-capacity.json',capacity)
        backup_receipt=backup.collect_backup(observation=before,roots=roots,backup_root=a.backup_root,
            local_root=EXT/'remote',helper_path=HELPER,log_dir=EXT/'backup-operations')
        selected=inventory(EXT/'remote')
        actual=validate_backup_capture(backup_receipt,
            dict(files=before['files'],directories=sorted(before['directories'])),selected)
        save('remote-custody.json',backup_receipt)
        after=observe('inventory_after_transport.json')
        need(after['files']==before['files'] and sorted(after['directories'])==sorted(before['directories'])
             and after['failed_token_present']==before['failed_token_present']
             and after['training_present']==before['training_present'] and after['cpu_present']==before['cpu_present'] and after['common_present']==before['common_present'] and after['failed_present']==before['failed_present']
             and after['root_present']==before['root_present'] and after['namespace_present']==before['namespace_present']
             and after['original_namespace_present']==before['original_namespace_present']
             and after['original_hardlink_groups']==before['original_hardlink_groups']
             and after['parent_accounting']==before['parent_accounting']
             and after['journal_pins']==before['journal_pins']
             and after['upstream_pins']==before['upstream_pins'] and after['upstream_errors']==before['upstream_errors']
             and normalize(after['related_accounting'])==normalize(before['related_accounting']),
             'second remote bytes/journals/terminal accounting differ')
        for sub in ('remote','backup-operations'):
            (LOCAL/sub).mkdir(mode=0o700,exist_ok=True)
            for path in (EXT/sub).iterdir():
                target=LOCAL/sub/path.name
                if path.is_dir():shutil.copytree(path,target,copy_function=copy_file)
                else:copy_file(path,target)
            need(inventory(EXT/sub)==inventory(LOCAL/sub),'second independent byte copy differs')
        errors=[]
        if inventory(prior_custody)!=prior_custody_before:errors.append('original accepted training custody changed')
        for label,expected in preserved.items():
            actual_map={n[len(label)+1:]:v for n,v in actual['files'].items() if n.startswith(label+'/')}
            actual_dirs=sorted(n[len(label)+1:] for n in actual['directories'] if n.startswith(label+'/'))
            if actual_map!=expected or actual_dirs!=preserved_directories[label]:errors.append('complete accepted '+label+' root changed/missing')
        if before['upstream_pins']!=upstream_pins or before['upstream_errors']:
            errors.append('accepted original input/descriptor bytes changed/missing')
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
            if not (claim.get('realization_root')==REALIZATION_ROOT
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
        if not accounting_ok:errors.append('exact45 prior plus current46th accounting row differs')
        if not journal_ok:errors.append('exact92 prior plus current94 journals differ')
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
        allocation=resources(before,a.job_id,EXT/'remote/launch/cpu',actual['files'])
        if after['results']['details']!=before['results']['details']:
            errors.append('scheduler request evidence changed between reads')
        try:
            observed_cost=cost(before['related_accounting']);prior_cost=cost(baseline['related_accounting'])
            need(prior_cost==(baseline['closed_cpu_seconds'],baseline['closed_gpu_slice_seconds']), 'baseline total/row mismatch')
            need(observed_cost==(prior_cost[0]+allocation['allocated_cpu_seconds'],prior_cost[1]+allocation['gpu_slice_seconds']), 'actual full-chain cost mismatch')
            need(observed_cost[0]+3600<=57600 and observed_cost[1]<=7200,'actual cumulative resource ceiling exceeded')
        except (ValueError,TypeError,KeyError) as exc:
            errors.append('actual accounting unresolved: '+str(exc));accounting_ok=False
        errors.extend(allocation['errors'])
        markers={n:actual['files'].get('launch/cpu/'+n) for n in (
            'supervisor.json','failure.json','worker.cleanup.json','worker.resources.json','worker.started.json',
            'worker.lifecycle-errors.json','worker/work-start.json','worker/artifacts/terminal.json',
            'worker/artifacts/failure.json','worker/artifacts/cpu-phase/terminal.json',
            'worker/artifacts/cpu-phase/campaign-bundle.json','worker/artifacts/cpu-phase/ranker/terminal.json')}
        failure_members=sorted(n for n in actual['files'] if n.startswith('launch/cpu/') and
            (Path(n).name.endswith('failure.json') or Path(n).name in ('publication-interruption.json','worker.lifecycle-errors.json')))
        complete_markers=all(markers[n] is not None for n in ('supervisor.json','worker/artifacts/terminal.json','worker/artifacts/cpu-phase/terminal.json'))
        reported=public_cpu_evidence(EXT/'remote/launch/cpu',actual['files'])
        save('public-cpu-evidence.json',reported)
        outcome='terminal_custody_review_pending' if complete_markers else 'terminal_partial_outputs_preserved'
        if allocation['scheduler_state'].split()[0]!='COMPLETED' or allocation['scheduler_exit']!='0:0' or failure_members:
            outcome='terminal_failed_outputs_preserved'
        if errors:outcome='terminal_custody_with_binding_disagreements'
        disposition=dict(schema='ace-scientific-cpu-campaign-terminal-custody-v1',utc=utc(),status=outcome,
            job_id=a.job_id,packet_sha256=a.packet_sha256,submission_result_sha256=a.submission_result_sha256,
            packet_commit=a.packet_commit,collector_source_commit=a.collector_source_commit,
            remote_metadata_deadline_seconds=1200,local_observation_timeout_seconds=1230,
            custody_bytes_verified=True,current_lineage_verified=not errors,errors=errors,input_pin_mismatches=mismatches,
            remote_root=ROOT,output_root=ROOT+'/cpu',remote_files=actual['files'],remote_directories=actual['directories'],
            raw_files=len(actual['files']),raw_bytes=size,original_hardlink_groups=before['original_hardlink_groups'],
            custody_mode='curc_projects_full_backup_with_compact_local_evidence',
            remote_custody_receipt_sha256=sha(raw(backup_receipt)),backup_helper_sha256=a.backup_helper_sha256,
            complete_local_raw_copy=False,local_raw_files=len(selected['files']),local_raw_bytes=sum(v['bytes'] for v in selected['files'].values()),
            hardlink_custody='Every original name has independent bytes in the verified CURC projects backup. Local evidence is an explicit subset. Original alias metadata retained.',
            namespace_local_sha256=None if claim_raw is None else sha(claim_raw),
            namespace_remote_sha256=actual['files'].get('namespace.json',{}).get('sha256'),
            recovery_bindings=recovery_bindings,training_acceptance_sha256=a.training_acceptance_sha256,
            training_custody_inventory_sha256=a.training_custody_inventory_sha256,collection_acceptance_sha256=accepted['collection_acceptance_sha256'],cpu_preparation_acceptance_sha256=accepted['cpu_preparation_acceptance_sha256'],
            token_recovery=history['token_recovery'],failed_token_root=FAILED_TOKEN_ROOT,
            operational_amendment_sha256=pins['operational-amendment.json'],preserved_upstream_roots=preserved,preserved_upstream_directories=preserved_directories,
            original_input_pins=upstream_pins,upstream_inputs_verified=before['upstream_pins']==upstream_pins and not before['upstream_errors'],
            original_namespace_local_sha256=None if original_claim_raw is None else sha(original_claim_raw),
            original_namespace_remote_sha256=actual['files'].get('original-namespace.json',{}).get('sha256'),
            exact_prior_accounting_matched=accounting_ok,exact_prior_and_current_journals_matched=journal_ok,
            observed_related_jobs=len(before['related_accounting']),observed_submission_journals=len(before['journal_pins']),
            allocation=allocation,closed_cpu_seconds=(baseline['closed_cpu_seconds']+allocation['allocated_cpu_seconds']) if accounting_ok else None,
            closed_gpu_slice_seconds=(baseline['closed_gpu_slice_seconds']+allocation['gpu_slice_seconds']) if accounting_ok and allocation['gpu_slice_seconds'] is not None else None,
            preparation_allowance_cpu_seconds=3600,
            cpu_admission_after_release=(baseline['closed_cpu_seconds']+3600+allocation['allocated_cpu_seconds']) if accounting_ok else None,
            cpu_ceiling_seconds=57600,gpu_ceiling_seconds=7200,output_marker_metadata=markers,
            public_cpu_evidence_sha256=sha(raw(reported)),failure_members=failure_members,
            collector_new_scientific_responses=0,collector_training_updates=0,development_payloads_decoded=False,scored_outcomes_decoded=False,
            checkpoint_tensors_decoded=False,scientific_qualified=False,cpu_campaign_accepted=False,acceptance=False,retry_allowed=False,
            collector_wall_seconds=time.monotonic()-started,collector_process_cpu_seconds=time.process_time()-cpu,
            cost_scope='Local collector process only; subprocess/remote costs not inferred. Actual Slurm allocation costs are separate, not utilization.',
            acceptance_scope='Custody and actual accounting only; retained claims, public counts and exit0 do not qualify runtime, fits, utility or science.')
        save('disposition.json',disposition)
        need(not errors,'full bytes retained but bindings disagree; independent reconciliation required')
        inventories={str(dest):inventory(dest) for dest in claimed}
        need(inventories[str(EXT)]==inventories[str(LOCAL)],'complete two-copy capsule differs')
        save('custody-inventory.json',dict(copies=inventories,identical=True,
             excludes='custody-inventory.json itself',acceptance=False,
             custody_mode='curc_projects_full_backup_with_compact_local_evidence',complete_local_raw_copy=False,
             remote_custody_receipt_sha256=sha(raw(backup_receipt)),
             scope='Two copies of selected local evidence and verification records only. Full raw bytes reside in the separately verified CURC archive.'))
        need(inventory(EXT)==inventory(LOCAL),'final capsule readback differs')
        sealed=True
        print(raw({k:v for k,v in disposition.items() if k not in ('remote_files','remote_directories')}).decode())
    except BaseException as exc:
        if claimed and not sealed:
            partial_errors=mirror_partial_evidence(EXT,LOCAL,copy_file) if len(claimed)==2 else []
            try:save('collector_failure.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),
                packet_commit=a.packet_commit,collector_source_commit=a.collector_source_commit,
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

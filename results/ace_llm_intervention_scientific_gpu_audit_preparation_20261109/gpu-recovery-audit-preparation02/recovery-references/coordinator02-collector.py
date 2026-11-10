"""Complete scientific GPU campaign terminal custody, never runtime/scientific acceptance.

Standalone sidecar. Requires independent current job/outer packet/result pins.
Copies the whole root and both separate namespace claims, retains all terminal
failures/partial outputs, and performs a second remote inventory/readback.
Only allowlisted public telemetry scalars are decoded after custody; score values and tensors stay opaque.
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
import shlex
import base64

REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LAUNCH=REPO/'results/ace_llm_intervention_scientific_campaign_gpu_20261108/launch02'
LAUNCH_EXT=BASE/'ACE-LLM-intervention-scientific-campaign-gpu-20261109-02'
LOCAL=LAUNCH.parent/'terminal02'
EXT=BASE/'ACE-LLM-intervention-scientific-campaign-gpu-terminal-20261109-02'
OLDROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02'
OUTPUT=OLDROOT+'/gpu'
CPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01'
TRAIN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01'
TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_02'
FAILED_TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'
COMMON_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_02'
FAILED_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01'
REALIZATION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02'
LOCAL_NAMESPACE=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json'
REMOTE_NAMESPACE=str(Path(ROOT).parent/'.ace-scientific-screen-20261107.namespace-recovery02.json')
OLD_LOCAL_NAMESPACE=BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json'
OLD_REMOTE_NAMESPACE=str(Path(ROOT).parent/'.ace-scientific-screen-20261107.namespace.json')
OLD_CLAIM_SHA='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
BUILDER_SHA='85691a9c7d52e40b334d9e42ea2da67762a955953c115de8a7d58be385935c5a'
POLICIES=('frozen_llm','sft_llm','dpo_llm')
UPSTREAM_ROOTS=dict(cpu=CPU_ROOT,training=TRAIN_ROOT,token=TOKEN_ROOT,common=COMMON_ROOT,failed01=FAILED_ROOT)
UPSTREAM_ROOTS['failed-token01']=FAILED_TOKEN_ROOT
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


FULL_MAP_PACKET_MEMBERS = frozenset(('gpu-failure-acceptance.json', 'gpu-failure-disposition.json', 'gpu-failure-inventory.json', 'gpu-failure-inventory-after.json', 'gpu-failure-seal.json', 'cpu-history-expected.json', 'inputs/cpu-history-expected.json', 'cpu-inventory.json', 'cpu-disposition.json', 'cpu-acceptance.json', 'gpu-inventory.json', 'gpu-inventory-after.json', 'gpu-custody-inventory.json', 'gpu-disposition.json', 'gpu-acceptance.json', 'storage-receipt.json', 'storage-seal.json', 'storage-after.json', 'inputs/cpu-acceptance.json', 'inputs/gpu-acceptance.json'))

def packet_member_maximum(name):
    return 128*1024**2 if name in FULL_MAP_PACKET_MEMBERS else 32*1024**2


def read_metadata(path,expected=None,maximum=32*1024**2):
    meta=fingerprint(path,sole_link=True);need(type(maximum) is int and 0<=maximum<=128*1024**2 and meta['bytes']<=maximum,'bounded coordinator metadata required')
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


REMOTE_OBSERVATION="import signal\ndef deadline(signum,frame):raise TimeoutError('bounded full terminal observation')\nsignal.signal(signal.SIGALRM,deadline);signal.alarm(1200)\n\nfrom pathlib import Path\nimport subprocess,hashlib,json,datetime,stat,os,sys,time,base64\n_READ_COUNT=0;_READ_STARTED=time.monotonic()\nroot=Path(ROOT);namespace=Path(NAMESPACE);original_namespace=Path(OLD_NAMESPACE)\ndef need(ok,message):\n if not ok:raise ValueError(message)\ndef fingerprint(path):\n global _READ_COUNT\n need(not path.is_symlink(),'remote symlink forbidden')\n with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:\n  before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode),'nonregular remote file')\n  h=hashlib.sha256()\n  for block in iter(lambda:f.read(1024**2),b''):h.update(block)\n  after=os.fstat(f.fileno())\n identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)\n need(identity(before)==identity(after)==identity(path.lstat()),'remote file changed during capture')\n _READ_COUNT+=1\n if _READ_COUNT%1000==0:print(json.dumps(dict(phase='terminal_inventory_verification',files_read=_READ_COUNT,elapsed_seconds=time.monotonic()-_READ_STARTED)),file=sys.stderr,flush=True)\n return dict(bytes=after.st_size,sha256=h.hexdigest()),after\nv=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results={},files={},directories=[],\n       root_present=root.exists(),failed_gpu_root_present=Path(OLDROOT).exists(),cpu_present=Path(CPU_ROOT).exists(),training_present=Path(TRAIN_ROOT).exists(),token_present=Path(TOKEN_ROOT).exists(),failed_token_present=Path(FAILED_TOKEN_ROOT).exists(),common_present=Path(COMMON_ROOT).exists(),failed_present=Path(FAILED_ROOT).exists(),namespace_present=os.path.lexists(namespace),original_hardlink_groups=[],\n       original_namespace_present=os.path.lexists(original_namespace),upstream_pins={},upstream_errors={})\ncommands={\n 'related_accounting':['sacct','-X','-u','paco0228','-S','2026-11-02','--format',\n   'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'],\n 'steps':['sacct','-j',JOB,'--format',\n   'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180,MaxRSS,TotalCPU','-n','-P'],\n 'details':['sacct','-X','-j',JOB,'--format','JobIDRaw,Partition,QOS,TimelimitRaw,ReqTRES%180,User','-n','-P'],\n 'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],\n 'job':['scontrol','show','job',JOB,'-o']}\nfor name,command in commands.items():\n try:\n  p=subprocess.run(command,capture_output=True,text=True,timeout=20)\n  v['results'][name]=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)\n except subprocess.TimeoutExpired as exc:\n  txt=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x\n  v['results'][name]=dict(command=command,returncode=None,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr))\ntry:\n need(v['results']['related_accounting']['returncode']==0 and v['results']['queue']['returncode']==0,\n      'scheduler observation unavailable')\n rows=[l.split('|') for l in v['results']['related_accounting']['stdout'].splitlines() if '|ace_llm' in l]\n selected=[r for r in rows if r[0]==JOB]\n need(len(selected)==1 and selected[0][2].split()[0] in TERMINAL,'requested job not conclusively terminal')\n queue=[l.split('|') for l in v['results']['queue']['stdout'].splitlines() if l.strip()]\n need(all(len(r)==5 and r[0]!=JOB and not r[1].lower().startswith('ace') for r in queue),'current/related ACE job active')\n v['parent_accounting']=selected[0];v['related_accounting']=rows;v['terminal_gate']=True\n links={}\n for label,tree,present in (('control',root,v['root_present']),('launch',Path(OLDROOT),v['failed_gpu_root_present']),('cpu',Path(CPU_ROOT),v['cpu_present']),('training',Path(TRAIN_ROOT),v['training_present']),('token',Path(TOKEN_ROOT),v['token_present']),('failed-token01',Path(FAILED_TOKEN_ROOT),v['failed_token_present']),('common',Path(COMMON_ROOT),v['common_present']),('failed01',Path(FAILED_ROOT),v['failed_present'])):\n  if not present:continue\n  need(tree.is_dir() and tree==tree.resolve(),'ordinary remote root required')\n  v['directories'].append(label)\n  for path in sorted(tree.rglob('*')):\n   need(not path.is_symlink(),'remote tree symlink forbidden')\n   name=label+'/'+path.relative_to(tree).as_posix()\n   if path.is_dir():v['directories'].append(name);continue\n   info,s=fingerprint(path);v['files'][name]=info\n   if s.st_nlink>1:links.setdefault((s.st_dev,s.st_ino,s.st_nlink),[]).append(name)\n for name,path,present in (('namespace.json',namespace,v['namespace_present']),\n                           ('original-namespace.json',original_namespace,v['original_namespace_present'])):\n  if not present:continue\n  need(path.parent==path.parent.resolve(),'registry parent alias')\n  info,s=fingerprint(path);v['files'][name]=info\n  if s.st_nlink>1:links.setdefault((s.st_dev,s.st_ino,s.st_nlink),[]).append(name)\n intent=root/'submission.intent.json'\n if intent.exists():\n  need(intent.stat().st_size<=1024**2,'bounded current control intent required')\n  data=intent.read_bytes();need(len(data)<=1024**2 and hashlib.sha256(data).hexdigest()==v['files']['control/submission.intent.json']['sha256'],'current intent changed after fingerprint')\n  v['current_control_intent_base64']=base64.b64encode(data).decode('ascii')\n v['original_hardlink_groups']=sorted([dict(paths=sorted(paths),link_count=k[2]) for k,paths in links.items()],\n                                     key=lambda group:group['paths'])\n v['journal_pins']={str(p):fingerprint(p)[0]['sha256'] for d in root.parent.glob('ace_llm_intervention*')\n                    for p in d.glob('submission.*.json')}\n # Explicit accepted upstream byte pins; never decode descriptors or outcomes.\n for path in UPSTREAM_PINS:\n  try:v['upstream_pins'][path]=fingerprint(Path(path))[0]['sha256']\n  except TimeoutError:raise\n  except (OSError,ValueError) as exc:v['upstream_errors'][path]=dict(type=type(exc).__name__,message=str(exc))\n\nexcept BaseException as exc:\n v['observation_error']=dict(type=type(exc).__name__,message=str(exc))\nprint(json.dumps(v,sort_keys=True))\n"


RESOURCES=dict(account='ucb736_asc2',partition='aa100',qos='gpu-testing',requested_cpus=4,
    max_allocated_cpus=4,memory_mib=15360,wall_seconds=2880,gpus=1,gpu_type='a100_3g.20gb',requeue=False)
LIMITS=dict(session_reserve_seconds=600,session_cap_seconds=1200,work_seconds=2820,final_seconds=2860)


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


def resource_tres(t):
    return (t.get('cpu')=='4' and t.get('node')=='1' and t.get('mem') in ('15360M','15G','15.00G')
        and t.get('gres/gpu:a100_3g.20gb')=='1' and gpu_count(t,4,1)==1
        and all(k in ('gres/gpu','gres/gpu:a100_3g.20gb') for k in t if k.startswith('gres/')))


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
            request_identity=r[:4]==[job,'aa100','gpu-testing','48'] and r[5]=='paco0228'
    fields={};allocation_pin=files.get('launch/gpu/allocation.json',{}).get('sha256')
    if allocation_pin:
        try:fields=decode(read_metadata(output/'allocation.json',allocation_pin))['fields']
        except (ValueError,KeyError,OSError) as exc:errors.append('saved allocation malformed: '+str(exc))
    if not fields and observation['results']['job']['returncode']==0:
        fields=dict(re.findall(r'([^\s=]+)=(\S+)',observation['results']['job']['stdout']))
    no_requeue=fields.get('JobId')==job and fields.get('Requeue')=='0' and fields.get('Restarts')=='0'
    matched=(request_identity and resource_tres(requested) and resource_tres(allocated) and no_requeue
        and normalized[1]=='ace_llm_scientific_campaign_gpu02' and normalized[6]=='ucb736_asc2'
        and cpus==4 and gpu==1 and 0<=elapsed<=2880)
    if not matched:errors.append('actual/requested GPU identity/resource/no-requeue/wall contract differs or unavailable')
    if gpu is None:errors.append('unknown GPU charge; never infer zero')
    return dict(resource_contract_matched=matched,errors=errors,expected=RESOURCES,requested_tres=requested,
        allocated_tres=allocated,allocated_cpus=cpus,elapsed_seconds=elapsed,allocated_cpu_seconds=cpus*elapsed,
        gpu_count=gpu,gpu_slice_seconds=None if gpu is None else gpu*elapsed,requested_mig_memory_gib=20,
        scheduler_state=row[2],scheduler_exit=row[3],account=row[6],alloc_tres=row[7],
        no_requeue_verified=no_requeue,allocation_fields=fields,allocation_source_sha256=allocation_pin,
        memory_note='15GiB host request and typed20GB MIG are distinct; framework total device bytes reported separately.')


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


def public_gpu_evidence(output,files):
    """Only producer public telemetry; all generations, private outcomes and tensors remain opaque."""
    prefix='launch/gpu/';selected={};errors=[];services={}
    scalars={'utc','outer_elapsed_seconds','service_elapsed_seconds','policy_id','registered_pid',
        'positive_generations','first_verified_monotonic','last_gpu_monotonic','last_generation_monotonic',
        'started_monotonic','closed_monotonic','process_group_closed','post_exit_gpu_empty','cuda_interval_ms','error'}
    for pid in POLICIES:
        stem='controller/phase/'+pid+'/'
        names=sorted(n[len(prefix):] for n in files if n.startswith(prefix+stem) and
            (re.fullmatch(r'telemetry-\d+\.json',n[len(prefix+stem):]) or n[len(prefix+stem):]=='summary.json'))
        need(len(names)<=20000,'bounded public metadata inventory required; retain raw originals')
        positive_memory=[];positive_generations=0;cuda_interval=None;registered=set();samples=0
        for name in names:
            item=files[prefix+name]
            try:
                v=public_fields(read_metadata(output/name,item['sha256']),scalars|{'gpu_process_sample','uncertainties'})
                for k in scalars&set(v):need(type(v[k]) in (str,int,float,bool,type(None)),'public scalar required')
                need(v.get('policy_id')==pid,'public service identity differs')
                process=v.get('registered_pid')
                if process is not None:need(type(process) is int and process>0,'registered PID invalid');registered.add(process)
                count=v.get('positive_generations')
                if count is not None:
                    need(type(count) is int and 0<=count<=576,'bounded recorded generation count required')
                    positive_generations=max(positive_generations,count)
                if name.endswith('/summary.json'):
                    interval=v.get('cuda_interval_ms')
                    if type(interval) in (int,float) and math.isfinite(interval) and interval>0:cuda_interval=interval
                sample=v.pop('gpu_process_sample',None)
                if sample is not None:
                    samples+=1
                    need(type(sample) is dict,'public GPU sample required')
                    # Retain only known sample fields in summary; full original JSON stays opaque in custody.
                    sample={k:sample[k] for k in ('returncode','owned_rows') if k in sample}
                    v['gpu_process_sample']=sample
                    if sample.get('returncode')==0 and type(sample.get('owned_rows')) is list:
                        for row in sample['owned_rows']:
                            if not isinstance(row,str):continue
                            parts=[part.strip() for part in row.split(',')]
                            if len(parts)!=3 or type(process) is not int or parts[0]!=str(process):continue
                            try:memory=float(parts[2])
                            except ValueError:continue
                            if math.isfinite(memory) and memory>0:positive_memory.append(memory)
                selected[name]=dict(sha256=item['sha256'],reported=v)
            except (ValueError,KeyError,TypeError,OSError) as exc:errors.append(name+': '+str(exc))
        if len(registered)>1:errors.append(pid+': recorded owned PID changed')
        services[pid]=dict(registered_pids=sorted(registered),nvidia_samples=samples,
            positive_owned_memory_rows=len(positive_memory),owned_memory_mib_max=max(positive_memory) if positive_memory else None,
            recorded_positive_generations=positive_generations,reported_cuda_interval_ms=cuda_interval,
            independent_signal_types_present=bool(positive_memory) and positive_generations>0 and len(registered)==1,
            utilization_percent=None)
    return dict(acceptance=False,reported_metadata=selected,services=services,errors=errors,
        all_three_services_have_reported_work_signals=all(v['independent_signal_types_present'] for v in services.values()),
        utilization_percent=None,utilization_note='Owned NVIDIA process memory plus recorded positive generations are distinct reported signals; not SM utilization percent. Missing/unreliable signals remain uncertain. Independent runtime and cleanup audit still required.',
        generation_records_decoded=False,scored_outcomes_decoded=False,adapter_tensors_decoded=False,runtime_qualified=False)


def upstream_disagreements(actual,preserved,preserved_directories):
    """Original accepted bytes stay fixed; GPU appends only the three frozen campaign attempts."""
    errors=[]
    def allowed(label,name):
        if label=='launch':return name=='gpu' or name.startswith('gpu/')
        return label=='cpu' and any(name==p or name.startswith(p+'/') for p in
            ('cpu/worker/artifacts/campaign/policy-05','cpu/worker/artifacts/campaign/policy-06','cpu/worker/artifacts/campaign/policy-07'))
    for label,expected in preserved.items():
        current={n[len(label)+1:]:v for n,v in actual['files'].items() if n.startswith(label+'/')}
        dirs={n[len(label)+1:] for n in actual['directories'] if n.startswith(label+'/')}
        prior_dirs=set(preserved_directories[label])
        if any(current.get(n)!=v for n,v in expected.items()):errors.append(label+': original accepted bytes missing/changed')
        if any(not allowed(label,n) for n in set(current)-set(expected)):errors.append(label+': unexpected added files')
        if not prior_dirs<=dirs:errors.append(label+': original accepted directory missing')
        if any(not allowed(label,n) for n in dirs-prior_dirs):errors.append(label+': unexpected added directories')
    return errors




def verify_failed_capsule(path,expected_pin):
    custody=decode(read_metadata(Path(path)/'custody-inventory.json',expected_pin));view=inventory(path)
    actual=dict(files={n:v for n,v in view['files'].items() if n!='custody-inventory.json'},directories=view['directories'])
    need(custody['identical'] is True and len(custody['copies'])==2 and all(v==actual for v in custody['copies'].values()),
         'complete original failed TOKEN01 archive and receipt capsule required')
    return view


def accounting_extension(observation,baseline,job,intent_pin,result_pin):
    prior=[r for r in observation['related_accounting'] if r[0]!=job]
    current=[r for r in observation['related_accounting'] if r[0]==job]
    rows_ok=(len(prior)==47 and len(current)==1 and len(observation['related_accounting'])==48
        and normalize(prior)==normalize(baseline['related_accounting'])
        and normalize(current)==normalize([observation['parent_accounting']])
        and all(r[2].split()[0] in TERMINAL for r in observation['related_accounting']))
    journals={**baseline['journal_pins'],ROOT+'/submission.intent.json':intent_pin,
              ROOT+'/submission.result.json':result_pin}
    journals_ok=(len(baseline['journal_pins'])==96 and len(journals)==98 and observation['journal_pins']==journals)
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








def compare_observations(before,after):
    # Same complete comparisons as the original collector, before materialization.
    need(after['files']==before['files'] and sorted(after['directories'])==sorted(before['directories'])
         and all(after[k]==before[k] for k in ('cpu_present','training_present','token_present','failed_token_present'))
         and after['common_present']==before['common_present'] and after['failed_present']==before['failed_present']
         and after['root_present']==before['root_present']
         and after['failed_gpu_root_present']==before['failed_gpu_root_present']
         and after.get('current_control_intent_base64')==before.get('current_control_intent_base64') and after['namespace_present']==before['namespace_present']
         and after['original_namespace_present']==before['original_namespace_present']
         and after['original_hardlink_groups']==before['original_hardlink_groups']
         and after['parent_accounting']==before['parent_accounting']
         and after['journal_pins']==before['journal_pins']
         and after['upstream_pins']==before['upstream_pins'] and after['upstream_errors']==before['upstream_errors']
         and normalize(after['related_accounting'])==normalize(before['related_accounting']),
         'second remote bytes/journals/terminal accounting differ')






def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('job-id','packet-sha256','submission-result-sha256','local-intent-sha256','packet-commit','collector-sha256','collector-repository-path','collector-review','collector-review-sha256','cpu-terminal-custody','cpu-custody-inventory-sha256','packet-review','packet-review-sha256','backup-root'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();need(re.fullmatch('[1-9][0-9]*',a.job_id) is not None,'exact current job ID required')
    packet_raw=read_metadata(LAUNCH/'packet.json',a.packet_sha256);packet=decode(packet_raw)
    result_raw=read_metadata(LAUNCH/'submission.result.json',a.submission_result_sha256);result=decode(result_raw)
    need(re.fullmatch('[0-9a-f]{40}',a.packet_commit) is not None,'full eventual packet commit required')
    need(sha(Path(__file__).read_bytes())==a.collector_sha256,'independently reviewed collector source pin required')
    need(packet['schema']=='ace-scientific-campaign-gpu-recovery-launch-v2' and packet['remote_root']==ROOT
         and result.get('job_id')==a.job_id and result.get('returncode')==0 and result.get('ambiguous') is False
         and pin(result.get('intent_sha256')),'unambiguous original packet/job/result required')
    intent_raw=read_metadata(LAUNCH/'submission.intent.json',a.local_intent_sha256);intent=decode(intent_raw)
    pins={**packet['new_input_files'],'packet.json':a.packet_sha256}
    collector_path=Path(a.collector_repository_path)
    need(not collector_path.is_absolute() and '..' not in collector_path.parts and collector_path.suffix=='.py',
         'explicit safe collector repository path required')
    own=read_metadata(Path(__file__),a.collector_sha256)
    git=subprocess.run(['git','--no-replace-objects','show',a.packet_commit+':'+collector_path.as_posix()],cwd=REPO,capture_output=True,timeout=20)
    need(git.returncode==0 and git.stdout==own,'standalone collector must equal independently pinned full packet commit bytes')
    collector_review=Path(a.collector_review).absolute();cr=read_metadata(collector_review,a.collector_review_sha256)
    git=subprocess.run(['git','--no-replace-objects','show',a.packet_commit+':'+str(collector_review.relative_to(REPO))],cwd=REPO,capture_output=True,timeout=20)
    need(cr.strip() and git.returncode==0 and git.stdout==cr,'independent collector review must be in packet commit')
    need(intent['packet_sha256']==a.packet_sha256 and intent['remote_root']==ROOT and intent['pins']==pins
         and intent['packet_commit']==a.packet_commit,'current local submission lineage differs')
    need(all(type(n) is str and not Path(n).is_absolute() and '..' not in Path(n).parts and pin(h) for n,h in pins.items()),
         'safe complete input pin roster required')
    # Verify immutable local packet/Git bytes without loading any project/author callable.
    for n,h in pins.items():
        data=read_metadata(LAUNCH/n,h,packet_member_maximum(n))
        git=subprocess.run(['git','--no-replace-objects','show',a.packet_commit+':'+str((LAUNCH/n).relative_to(REPO))],
                           cwd=REPO,capture_output=True,timeout=20)
        need(git.returncode==0 and git.stdout==data==read_metadata(LAUNCH_EXT/n,h,packet_member_maximum(n)),'committed packet mirrors differ')
    review=Path(a.packet_review).absolute();review_raw=read_metadata(review,a.packet_review_sha256)
    git=subprocess.run(['git','--no-replace-objects','show',a.packet_commit+':'+str(review.relative_to(REPO))],
                       cwd=REPO,capture_output=True,timeout=20)
    need(review_raw.strip() and git.returncode==0 and git.stdout==review_raw and intent['review_sha256']==a.packet_review_sha256,
         'independent original packet review/commit required')
    # Reuse the exact corrected coordinator's stdlib metadata-only admission.
    # All packet/Git/mirror bytes above have already been independently authenticated.
    builder_raw=read_metadata(LAUNCH/'build_packet.py',BUILDER_SHA)
    need(pins['build_packet.py']==BUILDER_SHA,'reviewed corrected campaign coordinator required')
    b=types.ModuleType('captured_campaign_metadata');b.__file__=str(LAUNCH/'build_packet.py')
    exec(compile(builder_raw,b.__file__,'exec'),vars(b))
    supplied={n:read_metadata(LAUNCH/b.metadata_name(n),pins[b.metadata_name(n)],b.metadata_maximum(n)) for n in b.METADATA}
    need(pins['cpu_history_contract.py']==b.CPU_HISTORY_SHA,'captured CPU history sidecar differs')
    ctx=b.authenticate(supplied)  # Pure original metadata; no builder main, science or source/model import.
    need(packet['token_recovery']==intent['token_recovery']==ctx['token_recovery']
         and packet['failed_token_root']==intent['failed_token_root']==FAILED_TOKEN_ROOT
         and packet['failed_token_job_id']==intent['failed_token_job_id']==b.FAILED_TOKEN_JOB
         and packet['history_directories']==ctx['history_directories'],'original complete TOKEN02 history differs')
    baseline_pin=pins['gpu-failure-baseline.json'];baseline=ctx['metadata']['gpu-failure-baseline']
    need(packet['gpu_recovery']==intent['gpu_recovery']==ctx['gpu_recovery'] and packet['prior_jobs']==47
         and packet['prior_journals']==96 and packet['paths']['gpu_output']==OUTPUT
         and packet['operational_source_commit']==intent['operational_source_commit']==ctx['metadata']['operational-source-manifest']['source_commit'],
         'GPU02 exact failure/recovery/control/output provenance differs')
    accepted=ctx['metadata']['cpu-acceptance'];upstream=ctx['metadata']['cpu-disposition']
    cpu_result=ctx['metadata']['cpu-result'];cpu_packet=ctx['metadata']['cpu-packet']
    collection=ctx['metadata']['collection-acceptance']
    need(packet['baseline_sha256']==intent['baseline_sha256']==baseline_pin
         and packet['cpu_job_id']==intent['cpu_job_id']==accepted['job_id']
         and packet['cpu_acceptance_sha256']==intent['cpu_acceptance_sha256']==pins['cpu-acceptance.json'],
         'accepted46/94 CPU campaign/intent bindings differ')
    need(packet['resources']==intent['resources']==RESOURCES and packet['limits']==intent['limits']==LIMITS,
         'unchanged GPU campaign resource/limit contract required')
    need(packet['source_commit']==intent['source_commit']==b.SOURCE_COMMIT
         and packet['operational_amendment_sha256']==pins['operational-amendment.json']
         and packet['gpu_packet_sha256']==pins['inputs/packet.json']
         and packet['cpu_packet_sha256']==pins['cpu-packet.json']
         and packet['worlds']==24 and packet['cells']==192
         and packet['gpu_policy_ids']==list(POLICIES) and packet['new_training_updates']==0,
         'exact frozen scientific campaign packet required')
    need(a.job_id not in [ctx['metadata'][n+'-result']['job_id'] for n in ('collection','token','training','cpu')]+['35407428',b.FAILED_TOKEN_JOB,b.FAILED_GPU_JOB],
         'distinct current campaign job required')
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
    try:
        for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
            lock=BASE/name;need(lock==lock.resolve(),'coordinator lock alias')
            f=os.fdopen(os.open(lock,os.O_RDWR|os.O_NOFOLLOW),'r+')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:f.close();raise
            locks.append(f)
        need(all(not os.path.lexists(p) for p in (LOCAL,EXT,EXT.with_name(EXT.name+'-storage-collector-operations'))),'terminal custody already claimed; never retry')
        for dest in (EXT,LOCAL):
            dest.mkdir(mode=0o700);claimed.append(dest)
            for sub in ('coordinator','namespaces'):(dest/sub).mkdir(mode=0o700)
            write(dest/'pull_terminal.py',Path(__file__).read_bytes())
        b.start_storage_operations(ctx,EXT.with_name(EXT.name+'-storage-collector-operations'),(EXT,LOCAL),
            dict(phase='collector',job_id=a.job_id,packet_sha256=a.packet_sha256,
                 collector_sha256=a.collector_sha256,packet_commit=a.packet_commit))
        original_files=ctx['observations']['cpu']['files']
        # Bound compact CPU capsule; original full archive and failed ancestors remain explicit.
        gate=b.configure_packet_storage(ctx,LAUNCH,pins,packet)
        need(intent['storage_binding']==packet['storage_binding'],'original storage intent differs')
        cpu_custody=Path(a.cpu_terminal_custody).resolve(strict=True)
        need(cpu_custody==gate.capsules[0] and a.cpu_custody_inventory_sha256==pins['storage-seal.json'],
             'independent original selected CPU capsule/seal differs')
        cpu_custody_before=inventory(cpu_custody)
        # This existing pure byte validator checks complete120 CPU cells, opaque bundle/attempt/source bytes,
        # matching numerical plan and all five CPU attempts. It never decodes private outcomes or imports project code.
        opaque_cpu_inputs=b.admit_cpu_bytes(ctx,cpu_custody/'remote');del opaque_cpu_inputs
        upstream_pins=ctx['upstream_pins']
        need(packet['original_cpu_inputs']==upstream_pins,'complete original CPU/history input roster differs')
        preserved={};preserved_directories={}
        for label,stage in [('cpu','cpu'),('training','training'),('token','token'),('common','collection')]:
            observed=ctx['observations'][stage]['files']
            preserved[label]={n[len('launch/'):]:v for n,v in observed.items() if n.startswith('launch/')}
            preserved_directories[label]=[n[len('launch/'): ] for n in ctx['observations'][stage]['directories'] if n.startswith('launch/')]
            need(preserved[label],'complete upstream root required: '+label)
        # The accepted collection transport must preserve the failed first collection's whole original root.
        preserved['failed01']={n[len('failed01/'):]:v for n,v in ctx['observations']['collection']['files'].items() if n.startswith('failed01/')}
        preserved_directories['failed01']=[n[len('failed01/'): ] for n in ctx['observations']['collection']['directories'] if n.startswith('failed01/')]
        need(preserved['failed01'],'complete failed collection01 root required')
        failed=b.decode(ctx['metadata']['token-failure-inventory']['stdout'])
        preserved['failed-token01']={n[len('launch/'):]:v for n,v in failed['files'].items() if n.startswith('launch/')}
        preserved_directories['failed-token01']=[n[len('launch/'):] for n in failed['directories'] if n.startswith('launch/')]
        need(preserved['failed-token01'],'whole original failed TOKEN01 root required')

        preserved['launch']=ctx['failed_gpu_inventory']['files']
        preserved_directories['launch']=ctx['failed_gpu_inventory']['directories']
        for path,h in upstream_pins.items():
            found=[(label,root) for label,root in UPSTREAM_ROOTS.items() if path.startswith(root+'/')]
            need(len(found)==1,'explicit upstream input root differs')
            label,root=found[0]
            need(preserved[label][path[len(root)+1:]]['sha256']==h,'upstream input/full original root differs')
        # Backend and review are independently captured/Git-bound outer storage members.
        backend=gate.backend;helper_source=gate.supplied['remote_custody.py']
        helper_review=gate.supplied['storage-backend-review.md']
        recovery_bindings=dict(recovery_authority_sha256=pins['recovery-authority.json'],original_namespace_claim_sha256=OLD_CLAIM_SHA)
        need(intent['existing_namespace_claim_sha256']==packet['namespace_claim_sha256']==pins['namespace-claim.json']
             and intent['namespace_admission_sha256']==pins['namespace-admission.json']
             and all(intent.get(k)==v for k,v in recovery_bindings.items()) and intent['new_namespace_claimed'] is False,
             'current GPU intent namespace/acceptance joins differ')
        source='\n'.join(k+'='+repr(v) for k,v in dict(ROOT=ROOT,OLDROOT=OLDROOT,NAMESPACE=REMOTE_NAMESPACE,JOB=a.job_id,
            TERMINAL=TERMINAL,OLD_NAMESPACE=OLD_REMOTE_NAMESPACE,UPSTREAM_PINS=upstream_pins,COMMON_ROOT=COMMON_ROOT,FAILED_ROOT=FAILED_ROOT,CPU_ROOT=CPU_ROOT,TRAIN_ROOT=TRAIN_ROOT,TOKEN_ROOT=TOKEN_ROOT,FAILED_TOKEN_ROOT=FAILED_TOKEN_ROOT).items())+'\n'+REMOTE_OBSERVATION
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
        for dest in claimed:
            write(dest/'remote_custody.py',helper_source);write(dest/'storage-backend-review.md',helper_review)
        save('collector-intent.json',dict(utc=utc(),job_id=a.job_id,packet_sha256=a.packet_sha256,
            submission_result_sha256=a.submission_result_sha256,remote_root=ROOT,output_root=OUTPUT,original_failed_root=OLDROOT,
            gpu_recovery=ctx['gpu_recovery'],historical_postcustody_required=True,remote_namespace=REMOTE_NAMESPACE,
            local_namespace=str(LOCAL_NAMESPACE),collector_sha256=sha(Path(__file__).read_bytes()),
            accepted_cpu_root=CPU_ROOT,accepted_collection_root=COMMON_ROOT,preserved_failed_root=FAILED_ROOT,
            collector_repository_path=a.collector_repository_path,collector_review_sha256=a.collector_review_sha256,
            cpu_custody_inventory_sha256=a.cpu_custody_inventory_sha256,
            backup_root=a.backup_root,backup_helper_sha256=gate.helper_sha256,
            storage_binding=packet['storage_binding'],complete_local_raw_copy=False,
            packet_commit=a.packet_commit,local_intent_sha256=a.local_intent_sha256,original_local_namespace=str(OLD_LOCAL_NAMESPACE),
            original_remote_namespace=OLD_REMOTE_NAMESPACE,recovery_bindings=recovery_bindings,
            retry_allowed=False,acceptance=False))
        for dest in claimed:
            write(dest/'collector-review.md',cr)
        save('predecessor-reference.json',dict(path=str(cpu_custody),
            custody_inventory_sha256=a.cpu_custody_inventory_sha256,
            disposition_sha256=pins['cpu-disposition.json'],raw_inventory_sha256=pins['cpu-inventory.json'],
            complete_existing_capsule_authenticated=True,recursive_copy_created=False,original_preserved=True,
            custody_mode='curc_projects_full_backup_with_compact_local_evidence'))
        coordinator={}
        for label,original in (('local-launch',LAUNCH),('external-launch',LAUNCH_EXT)):
            view=inventory(original);coordinator[label]=view
            save('coordinator/'+label+'-reference.json',dict(path=str(original),inventory=view,
                recursive_copy_created=False,original_preserved=True))
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
        all_roots=dict(control=ROOT,launch=OLDROOT,**UPSTREAM_ROOTS,**{'namespace.json':REMOTE_NAMESPACE,'original-namespace.json':OLD_REMOTE_NAMESPACE})
        roots={label:path for label,path in all_roots.items() if label in before['directories'] or label in before['files']}
        actual,after,backup_receipt=b.compact_adapter().collect_selected(backend=backend,
            helper_sha256=gate.helper_sha256,before=before,roots=roots,backup_root=a.backup_root,
            external=EXT,local=LOCAL,helper_path=HELPER,save=save,observe=observe,
            compare=compare_observations,copy_file=copy_file)
        b.recheck_storage(ctx)
        for label,original in (('local-launch',LAUNCH),('external-launch',LAUNCH_EXT)):
            need(inventory(original)==coordinator[label],'original coordinator history changed')
        errors=[]
        if inventory(cpu_custody)!=cpu_custody_before:errors.append('original accepted CPU custody changed')
        errors.extend(upstream_disagreements(actual,preserved,preserved_directories))
        if before['upstream_pins']!=upstream_pins or before['upstream_errors']:
            errors.append('accepted original CPU/collection input bytes changed/missing')
        if actual['files'].get('original-namespace.json',{}).get('sha256')!=OLD_CLAIM_SHA:
            errors.append('original remote namespace claim missing/changed')
        if original_claim_raw is None or sha(original_claim_raw)!=OLD_CLAIM_SHA:
            errors.append('original local registry missing/changed')
        if original_claim_raw is not None and (not os.path.lexists(OLD_LOCAL_NAMESPACE)
                or fingerprint(OLD_LOCAL_NAMESPACE)['sha256']!=sha(original_claim_raw)):
            errors.append('original local registry changed during collection')
        if not before['root_present']:errors.append('new control root missing')
        if not before['failed_gpu_root_present']:errors.append('original failed/output root missing')
        mismatches=[n for n,h in pins.items() if actual['files'].get('control/'+n,{}).get('sha256')!=h]
        if mismatches:errors.append('remote original input/packet pin disagreement')
        if actual['files'].get('control/submission.result.json',{}).get('sha256')!=a.submission_result_sha256:
            errors.append('remote current result pin differs')
        if actual['files'].get('control/submission.intent.json',{}).get('sha256')!=result['intent_sha256']:
            errors.append('remote current intent pin differs')
        expected_claim=intent.get('existing_namespace_claim_sha256')
        if claim_raw is None or not pin(expected_claim) or sha(claim_raw)!=expected_claim:
            errors.append('separate local namespace claim missing/different')
        if actual['files'].get('namespace.json',{}).get('sha256')!=expected_claim:
            errors.append('separate remote namespace claim missing/different')
        try:
            remote_intent_raw=base64.b64decode(before['current_control_intent_base64'],validate=True)
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
        if not accounting_ok:errors.append('exact47 prior plus current48th accounting row differs')
        if not journal_ok:errors.append('exact96 prior plus current98 journals differ')
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
        allocation=resources(before,a.job_id,EXT/'remote/launch/gpu',actual['files'])
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
        marker_names=['terminal.json','failure.json','controller/phase/terminal.json','controller/phase/failure.json',
            'controller/phase/cleanup-hold-intent.json','controller/failure.json']
        for policy in POLICIES:
            marker_names += [f'controller/phase/{policy}/summary.json',f'controller/phase/{policy}/fallback-cleanup.json',
                f'controller/phase/{policy}/post-exit-gpu.json']
            marker_names += [f'services/{policy}/{sub}/{name}.json' for sub in ('worker','launch') for name in ('terminal','failure')]
        markers={n:actual['files'].get('launch/gpu/'+n) for n in marker_names}
        failure_markers={n:v for n,v in actual['files'].items() if (n.startswith('launch/gpu/') or
            any(n.startswith('cpu/cpu/worker/artifacts/campaign/policy-'+i+'/') for i in ('05','06','07')))
            and (n.endswith('/failure.json') or n.endswith('/publication-interruption.json') or n.endswith('/cleanup-failure.json'))}
        complete_markers=all(markers[n] is not None for n in ('terminal.json','controller/phase/terminal.json'))
        reported=public_gpu_evidence(EXT/'remote/launch/gpu',actual['files'])
        # Missing GPU work/cleanup evidence is unknown, not proof of idleness or success.
        save('public-gpu-evidence.json',reported)
        outcome='terminal_custody_review_pending' if complete_markers else 'terminal_partial_outputs_preserved'
        if allocation['scheduler_state'].split()[0]!='COMPLETED' or allocation['scheduler_exit']!='0:0' or failure_markers:
            outcome='terminal_failed_outputs_preserved'
        if errors:outcome='terminal_custody_with_binding_disagreements'
        disposition=dict(schema='ace-scientific-campaign-gpu-terminal-custody-v1',utc=utc(),status=outcome,
            job_id=a.job_id,packet_sha256=a.packet_sha256,submission_result_sha256=a.submission_result_sha256,
            custody_bytes_verified=True,current_lineage_verified=not errors,errors=errors,input_pin_mismatches=mismatches,
            remote_root=ROOT,output_root=OUTPUT,original_failed_root=OLDROOT,
            raw_root_labels={'control':ROOT,'launch':OLDROOT},gpu_recovery=ctx['gpu_recovery'],
            original_failed_collector_status=ctx['metadata']['gpu-failure-disposition']['status'],
            historical_postcustody_required=True,historical_postcustody_byte_comparison_performed=True,remote_files=actual['files'],remote_directories=actual['directories'],
            raw_files=len(actual['files']),raw_bytes=size,original_hardlink_groups=before['original_hardlink_groups'],
            custody_mode='curc_projects_full_backup_with_compact_local_evidence',
            remote_custody_receipt_sha256=sha(raw(backup_receipt)),backup_helper_sha256=gate.helper_sha256,
            complete_local_raw_copy=False,local_raw_files=len(backup_receipt['local_inventory']['files']),
            local_raw_bytes=sum(v['bytes'] for v in backup_receipt['local_inventory']['files'].values()),
            hardlink_custody='All original names verified in CURC USTAR archive. Local evidence is explicitly selected; original alias metadata retained.',
            namespace_local_sha256=None if claim_raw is None else sha(claim_raw),
            namespace_remote_sha256=actual['files'].get('namespace.json',{}).get('sha256'),
            recovery_bindings=recovery_bindings,cpu_acceptance_sha256=pins['cpu-acceptance.json'],
            cpu_custody_inventory_sha256=a.cpu_custody_inventory_sha256,
            training_acceptance_sha256=pins['training-acceptance.json'],token_acceptance_sha256=pins['token-acceptance.json'],
            collection_acceptance_sha256=pins['collection-acceptance.json'],
            token_recovery=ctx['token_recovery'],failed_token_root=FAILED_TOKEN_ROOT,
            operational_amendment_sha256=pins['operational-amendment.json'],preserved_upstream_roots=preserved,preserved_upstream_directories=preserved_directories,
            authorized_campaign_append_roots=['cpu/cpu/worker/artifacts/campaign/policy-'+n for n in ('05','06','07')],
            authorized_old_gpu_root_append='launch/gpu',control_raw_locally_selected=False,
            control_logs_custody='Full authenticated remote archive; exact demand fetch if audit requires original log bytes',
            original_input_pins=upstream_pins,upstream_inputs_verified=before['upstream_pins']==upstream_pins and not before['upstream_errors'],
            original_namespace_local_sha256=None if original_claim_raw is None else sha(original_claim_raw),
            original_namespace_remote_sha256=actual['files'].get('original-namespace.json',{}).get('sha256'),
            exact_prior_accounting_matched=accounting_ok,exact_prior_and_current_journals_matched=journal_ok,
            observed_related_jobs=len(before['related_accounting']),observed_submission_journals=len(before['journal_pins']),
            allocation=allocation,closed_cpu_seconds=(baseline['closed_cpu_seconds']+allocation['allocated_cpu_seconds']) if accounting_ok else None,
            closed_gpu_slice_seconds=(baseline['closed_gpu_slice_seconds']+allocation['gpu_slice_seconds']) if accounting_ok and allocation['gpu_slice_seconds'] is not None else None,
            preparation_allowance_cpu_seconds=3600,
            cpu_admission_after_release=(baseline['closed_cpu_seconds']+3600+allocation['allocated_cpu_seconds']) if accounting_ok else None,
            cpu_ceiling_seconds=57600,gpu_ceiling_seconds=7200,output_marker_metadata=markers,failure_marker_metadata=failure_markers,
            public_gpu_evidence_sha256=sha(raw(reported)),development_payloads_read=0,scored_outcomes_decoded=False,
            adapter_tensors_decoded=False,scientific_qualified=False,runtime_qualified=False,training_execution_qualified=False,acceptance=False,retry_allowed=False,
            collector_wall_seconds=time.monotonic()-started,collector_process_cpu_seconds=time.process_time()-cpu,
            cost_scope='Local collector process only; subprocess/remote costs not inferred. Actual Slurm allocation costs are separate, not utilization.',
            acceptance_scope='Custody and actual accounting only; retained claims, work signals and exit0 do not qualify runtime, training, utility or science.')
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
                for sub in ('remote','coordinator','backup-operations'):
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

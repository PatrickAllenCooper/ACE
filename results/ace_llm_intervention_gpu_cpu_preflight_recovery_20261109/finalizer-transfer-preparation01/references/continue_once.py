"""Exclusive GPU02 pre-submission transfer continuation; no automatic retry.

Authoring this file does not execute it. Only main() can claim an actual operation.
All scientific files remain opaque. Original packet/collector contracts are reused.
"""
import argparse
import ast
import base64
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
import types

REPO = Path('/Users/pat/code/ACE')
BASE = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LOCAL = REPO/'results/ace_llm_intervention_scientific_campaign_gpu_20261108/launch02'
EXT = BASE/'ACE-LLM-intervention-scientific-campaign-gpu-20261109-02'
ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02'
PROJECT_PARENT = '/projects/paco0228/ACE_evidence'
OP = 'transfer-continuation01'
PARTIAL = 'gpu-failure-disposition.json'
PARTIAL_BYTES = 2392064
PARTIAL_SHA = '7b8939e8509c6849203e0eb3daad0817aba743dc163fab1580de2505269cf66e'
PACKET_COMMIT = '463e5b1140e8aa957e07636c603c3057da81e7d6'
PROOF_COMMIT = 'b476bc40bda7abb789416b37df6910a2196334aa'
PACKET_SHA = '25fb597c6f475fa0bbfe924062bc439bb197ca0b646cecdae2be1a080010d684'
BUILDER_SHA = '85691a9c7d52e40b334d9e42ea2da67762a955953c115de8a7d58be385935c5a'
LAUNCHER_SHA = '7bcd8c3db86446d92e954011a2351ebd773018b92fd7879607f8c2e9a9c9fe93'
HELPER = '/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
PROOFS = {
 'transfer': ('results/ace_llm_intervention_scientific_campaign_gpu_20261108/launch02/transfer-root.json','3088f9388b98f0e1fd97e6f1b08fc8932cbd3866fb5661c6b61c04fdb5202044'),
 'failure': ('results/ace_llm_intervention_scientific_campaign_gpu_20261108/launch02/launcher_failure.json','6564aec94d50be6173baad2788db08ee2ccc7289eae8f7983166fcb6b714719a'),
 'observation': ('results/ace_llm_intervention_scientific_campaign_gpu_20261108/transport-failure01/stdout.log','2ab4613717329ee35e2a7a58c79cfbe196bd9eb7d579ef5617b2b8270e8b3da4'),
 'quiescence': ('results/ace_llm_intervention_scientific_campaign_gpu_20261108/transport-quiescence02/stdout.log','39331aa1e9429f25b370f4e63dd0dfb5b6130bcdb6d6ac835aa9336cf065d797'),
 'local_processes': ('results/ace_llm_intervention_scientific_campaign_gpu_20261108/transport-failure01/local-transfer-processes.json','5d52efa07705fffbb40ef0ca5cbee9c96d3eca40d78ca420199d264ecca7545a'),
}
REVIEW_PATH = 'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109/packet-review02/review.md'
REVIEW_SHA = '743f7ec49c47d12fef784dfbe0823c93aa7c287e7ff8edf15d639a98f70a6ef2'
DESIGN_REVIEW_PATH = 'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109/transport-recovery-design-review01/review.md'
DESIGN_REVIEW_SHA = '9231ec51d64fb87df5c99055225ba0f56783fcf4cf8498ccde0d4e93b10bcc75'
MAX_META = 128*1024**2
MEMBER_MAX = 256*1024**2

class TerminationRequested(BaseException):
    pass

@contextlib.contextmanager
def termination_guard():
    def terminate(signum,frame):
        raise TerminationRequested('SIGTERM requested bounded owned cleanup')
    previous=signal.signal(signal.SIGTERM,terminate)
    try:yield
    finally:signal.signal(signal.SIGTERM,previous)

@contextlib.contextmanager
def cleanup_signals(primary):
    """Defer SIGTERM and SIGINT until cleanup finishes; preserve the primary."""
    pending=[];error=primary
    previous={sig:signal.signal(sig,lambda signum,frame:pending.append(signum))
              for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        yield
    except BaseException as exc:
        error=exc;raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
        if pending and error is not None and hasattr(error,'add_note'):
            error.add_note('Additional SIGTERM/SIGINT deferred during owned cleanup')
    if pending and error is None:raise TerminationRequested('SIGTERM/SIGINT deferred until owned cleanup completed')

def preservation_state():
    return dict(phase='not_started',projects_readback=None,local_readbacks={},original_move='not_requested')

def failure_record(exc,preservation):
    saved=decode(raw(preservation))
    verified=saved['projects_readback'] is not None and len(saved['local_readbacks'])==2
    return dict(utc=utc(),type=type(exc).__name__,message=str(exc),retry_allowed=False,
                partial_preserved=True if verified else (False if saved['phase']=='not_started' else None),
                partial_preservation=saved,remote_outcome_ambiguous=True)

def confirm_owned_group_closed(pgid, seconds=5):
    deadline=time.monotonic()+seconds;observations=[];count=0
    while True:
        count+=1
        try:os.killpg(pgid,0);state='present'
        except ProcessLookupError:
            return dict(closed=True,probes=count,observations=observations+['ESRCH'])
        except OSError as exc:state=type(exc).__name__+': '+str(exc)
        if len(observations)<12:observations.append(state)
        if time.monotonic()>=deadline:return dict(closed=False,probes=count,observations=observations)
        time.sleep(.02)

def need(ok, message):
    if not ok: raise ValueError(message)

def raw(v): return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(v): return hashlib.sha256(v).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def decode(data):
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out,'duplicate JSON key');out[k]=v
        return out
    def bad(v): raise ValueError('nonfinite JSON: '+v)
    return json.loads(data,object_pairs_hook=pairs,parse_constant=bad)

def identity(s): return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)

def read(path, expected=None, maximum=MEMBER_MAX):
    p=Path(path);need(p==p.resolve(strict=True),'canonical no-alias file required: '+str(p))
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=maximum,'bounded single-link regular file required')
        data=f.read(maximum+1);after=os.fstat(f.fileno())
    need(len(data)<=maximum and identity(before)==identity(after)==identity(p.lstat()),'file changed during read')
    need(expected is None or sha(data)==expected,'independent byte pin differs: '+str(p))
    return data

def write(path, data):
    p=Path(path);need(p.parent==p.parent.resolve(strict=True),'write parent alias')
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
    need(read(p,sha(data),max(MEMBER_MAX,len(data)))==data,'exclusive readback differs')

def new_dir(path):
    p=Path(path);need(p.parent==p.parent.resolve(strict=True),'directory parent alias')
    p.mkdir(mode=0o700);os.chmod(p,0o700)
    need(p==p.resolve(strict=True) and p.stat().st_uid==os.getuid(),'new owned directory required')
    return p

def safe_name(name):
    need(type(name) is str and name and not name.startswith('/') and all(p not in ('','.','..') for p in name.split('/')),'unsafe member')
    return name

def owned_parents(root, name):
    p=Path(root)
    for part in Path(safe_name(name)).parts[:-1]:
        p=p/part
        if not os.path.lexists(p):new_dir(p)
        need(p.is_dir() and not p.is_symlink() and p==p.resolve(),'staging parent alias')
    return Path(root)/name

def stat_map(root, excluded=()):
    root=Path(root);need(root==root.resolve() and root.is_dir() and root.stat().st_uid==os.getuid(),'canonical owned root required')
    files={};dirs=[]
    for p in sorted(root.rglob('*')):
        n=p.relative_to(root).as_posix()
        if any(n==x or n.startswith(x+'/') for x in excluded):continue
        s=p.lstat();need(not stat.S_ISLNK(s.st_mode),'tree alias')
        if stat.S_ISDIR(s.st_mode):dirs.append(n);continue
        data=read(p)
        files[n]=dict(bytes=len(data),sha256=sha(data),mtime_ns=s.st_mtime_ns)
    return dict(files=files,directories=dirs)

def assert_partial_tree(root, observation, *, excluded=()):
    actual=stat_map(root,excluded)
    wanted={n:{k:v[k] for k in ('bytes','sha256','mtime_ns')} for n,v in observation['files'].items()}
    need(actual==dict(files=wanted,directories=observation['directories']),'original partial tree changed; no reuse')
    return actual

def validate_proofs(proofs, pins, sizes):
    obs,q,t,f=map(proofs.__getitem__,('observation','quiescence','transfer','failure'))
    need(obs['root']==q['root']==ROOT and obs['mismatches']==[PARTIAL] and not obs['unexpected'] and not obs['changed_during_observation'],'wrong or changed partial observation')
    need(obs['submission_intent_exists'] is obs['submission_result_exists'] is obs['output_exists'] is False,'existing submission/output')
    need(q['same_membership_and_stat_as_prior'] is True and q['partial_sha256']==PARTIAL_SHA and q['fuser']==dict(returncode=1,stdout='',stderr=''),'conclusive saved quiescence required')
    need(t['returncode'] is None and t['timeout'] is t['ambiguous'] is True and t['retry_allowed'] is False,'wrong transfer failure')
    need(f['retry_allowed'] is False and 'transfer-root.json' in f['message'],'wrong launcher failure boundary')
    need(proofs['local_processes']==dict(matches=[],returncode=0),'saved owned local transfers not closed')
    p=obs['files'][PARTIAL]
    need(p['bytes']==PARTIAL_BYTES and p['sha256']==PARTIAL_SHA and p['expected_sha256']==pins[PARTIAL] and sizes[PARTIAL]>PARTIAL_BYTES,'exact known truncated member required')
    need(set(obs['files'])-{'readiness.json'}<=set(pins),'unexpected original partial members')
    need(set(obs['missing'])==set(pins)-set(obs['files']),'missing roster differs')
    for n,v in obs['files'].items():
        if n not in (PARTIAL,'readiness.json'):
            need(v['sha256']==pins[n] and v['bytes']==sizes[n] and v['matches_expected'] is True,'nonpartial input differs')
    need(obs['directories']==['inputs','logs'] and not any(n.startswith('inputs/') for n in obs['files']),'wrong failed transfer phase')
    return {n:dict(bytes=sizes[n],sha256=pins[n]) for n in sorted(set(obs['missing'])|{PARTIAL})}

def quiescent(root, observation):
    p=subprocess.run(['fuser','-v',*[str(Path(root)/n) for n in sorted(observation['files'])]],capture_output=True,text=True,timeout=20)
    result=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    need(result==dict(returncode=1,stdout='',stderr=''),'live file user or inconclusive fuser; no mutation')
    return result

def no_local_transfer():
    p=subprocess.run(['ps','-U',str(os.getuid()),'-o','pid=,comm=,args='],capture_output=True,text=True,timeout=15)
    need(p.returncode==0,'local owned-process query failed')
    matches=[]
    for line in p.stdout.splitlines():
        fields=line.strip().split(None,2)
        if len(fields)!=3:continue
        command=Path(fields[1]).name;args=fields[2]
        if command=='scp' or (command=='ssh' and ('sftp' in args or re.search(r'\bscp\s',args))):matches.append(line)
    need(not matches,'owned transfer process still present; leave it untouched')
    return dict(utc=utc(),matches=matches,returncode=0)

def preserve_partial(root, backup, evidence, observation, lineage):
    """Remote use only after quiescence/full reconciliation; artificial tests use temp trees."""
    assert_partial_tree(root,observation)
    data=read(Path(root)/PARTIAL,PARTIAL_SHA,PARTIAL_BYTES)
    need(len(data)==PARTIAL_BYTES,'partial byte length differs')
    for parent in (Path(backup).parent,Path(evidence).parent):
        fs=os.statvfs(parent);need(fs.f_bavail*fs.f_frsize>=len(data)*3+16*1024**2,'partial preservation headroom')
    new_dir(evidence);new_dir(backup)
    record=dict(schema='ace-gpu02-transfer-partial-preservation-v1',utc=utc(),original_path=str(Path(root)/PARTIAL),bytes=len(data),sha256=sha(data),projects_backup=str(Path(backup)/'partial.original'),lineage=lineage,acceptance=False)
    write(Path(evidence)/'intent.json',raw(record));write(Path(backup)/'intent.json',raw(record))
    write(Path(backup)/'partial.original',data)
    need(read(Path(backup)/'partial.original',PARTIAL_SHA,PARTIAL_BYTES)==data,'projects backup readback differs')
    write(Path(evidence)/'projects-readback.json',raw(record));write(Path(backup)/'readback.json',raw(record))
    assert_partial_tree(root,observation,excluded=(Path(evidence).name,))
    return dict(record=record,record_sha256=sha(raw(record)),partial_base64=base64.b64encode(data).decode())

def relocate_partial(root,evidence,backup,observation,ack):
    need(ack['partial_sha256']==PARTIAL_SHA and ack['partial_bytes']==PARTIAL_BYTES and len(ack['local_copies'])==2,'both selected local readbacks required')
    assert_partial_tree(root,observation,excluded=(Path(evidence).name,))
    original=Path(root)/PARTIAL;target=Path(evidence)/'partial.renamed'
    data=read(original,PARTIAL_SHA,PARTIAL_BYTES)
    need(read(Path(backup)/'partial.original',PARTIAL_SHA,PARTIAL_BYTES)==data,'durable partial changed before rename')
    need(not os.path.lexists(target),'exclusive rename target already exists')
    # Parent is this attempt's owned0700 directory; both coordinator locks are held.
    before=identity(original.lstat());os.rename(original,target)
    need(read(target,PARTIAL_SHA,PARTIAL_BYTES)==data and target.stat().st_ino==before[1] and not os.path.lexists(original),'partial rename verification failed')
    write(Path(evidence)/'local-readback-ack.json',raw(ack))

def receive_missing(stream, root, evidence, members, observation):
    """Read exact ordered opaque bytes, then publish ONLY absent original members."""
    staged=new_dir(Path(evidence)/'staged');count=0;total=0;start=time.monotonic()
    print(json.dumps(dict(phase='missing_transfer',state='start',members=len(members),bytes=sum(v['bytes'] for v in members.values()))),file=sys.stderr,flush=True)
    last=start
    for name,info in sorted(members.items()):
        need(not os.path.lexists(Path(root)/safe_name(name)),'refuse existing publication destination')
        p=owned_parents(staged,name);h=hashlib.sha256();remaining=info['bytes']
        with p.open('xb') as out:
            while remaining:
                block=stream.read(min(1024**2,remaining));need(block,'truncated transfer retained')
                out.write(block);h.update(block);remaining-=len(block);total+=len(block)
                now=time.monotonic()
                if now-last>=5:
                    print(json.dumps(dict(phase='missing_transfer',files=count,bytes=total,elapsed_seconds=now-start)),file=sys.stderr,flush=True);last=now
            out.flush();os.fsync(out.fileno())
        need(h.hexdigest()==info['sha256'] and len(read(p,info['sha256']))==info['bytes'],'staged member pin differs')
        count+=1
    need(stream.read(1)==b'','extra transfer bytes')
    # No publication occurs until the entire bounded transfer has authenticated.
    for name,info in sorted(members.items()):
        target=Path(root)/safe_name(name)
        need(target.parent==target.parent.resolve() and target.parent.is_dir(),'original destination parent differs')
        write(target,read(staged/name,info['sha256']))
    expected={n:{k:v[k] for k in ('bytes','sha256','mtime_ns')} for n,v in observation['files'].items() if n!=PARTIAL}
    after=stat_map(root,excluded=(Path(evidence).name,))
    need(all(after['files'].get(n)==v for n,v in expected.items()),'previously matching file changed')
    need(set(after['files'])==set(expected)|set(members) and after['directories']==observation['directories'],'unexpected completion membership')
    return dict(completed_members=members,total_bytes=total,matching_original_members_unchanged=True,acceptance=False)

def completion_admission(req):
    members=req['members'];obs=req['observation'];pins=req['context']['pins']
    need(type(members) is dict and set(members)==set(obs['missing'])|{PARTIAL},'exact missing-only completion roster required')
    for n,v in members.items():
        safe_name(n)
        need(set(v)=={'bytes','sha256'} and type(v['bytes']) is int and 0<=v['bytes']<=MEMBER_MAX and v['sha256']==pins[n],'exact bounded missing member pin required')
    need(sum(v['bytes'] for v in members.values())<=1024**3,'bounded original packet completion payload')
    return sum(v['bytes'] for v in members.values())

def common_from_source(source):
    need(sha(source)==LAUNCHER_SHA,'original launcher bytes differ')
    values=[ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE_COMMON' for t in n.targets)]
    need(len(values)==1 and type(values[0]) is str,'original common source missing')
    return values[0]

def remote_entry(req, stream):
    need(req['context']['root']==ROOT,'fixed continuation root differs')
    backup=Path(req['projects_backup']);need(backup.parent==Path(PROJECT_PARENT) and backup.name.startswith('gpu02-transfer-continuation-') and backup.parent==backup.parent.resolve() and backup.parent.stat().st_uid==os.getuid(),'explicit trusted projects backup parent required')
    env={'C':req['context']};exec(compile(common_from_source(base64.b64decode(req['launcher_source'],validate=True)),'<exact-original-REMOTE_COMMON>','exec'),env)
    root=Path(ROOT);evidence=root/OP;phase=req['phase'];obs=req['observation']
    try:
        if phase=='preserve':
            need(not os.path.lexists(evidence) and not os.path.lexists(backup),'continuation already claimed; no retry')
            fresh=env['reconcile']();assert_partial_tree(root,obs);fuser=quiescent(root,obs)
            fs=os.statvfs(root);free=fs.f_bavail*fs.f_frsize
            need(free>=4*1024**3+2*completion_admission(req),'scratch output and staged completion headroom')
            result=preserve_partial(root,backup,evidence,obs,req['lineage'])
            write(evidence/'fresh-before-preservation.json',raw(dict(reconciliation=fresh,fuser=fuser,scratch_free_bytes=free)))
        else:
            need(read(evidence/'projects-readback.json',req['preservation_sha256'])==read(backup/'readback.json',req['preservation_sha256']),'preservation identity differs')
            need(decode(read(evidence/'intent.json'))['lineage']==req['lineage'],'continuation lineage changed')
            if phase=='complete':
                # One absolute deadline covers recheck, transfer, staged readback and publication.
                signal.alarm(600)
                completion_admission(req)
                need(not os.path.lexists(evidence/'complete.intent.json'),'completion already attempted; no retry')
                quiescent(root,obs);assert_partial_tree(root,obs,excluded=(OP,))
                write(evidence/'complete.intent.json',raw(dict(utc=utc(),members=req['members'],lineage=req['lineage'],retry_allowed=False)))
                relocate_partial(root,evidence,backup,obs,req['local_ack'])
                result=receive_missing(stream,root,evidence,req['members'],obs)
                env['verify']();write(evidence/'complete.json',raw(result))
            elif phase=='preflight':
                need(sha(read(evidence/'complete.json'))==req['completion_sha256'],'completed transfer pin differs')
                result=dict(fresh=env['reconcile'](),verified=env['verify'](),utc=utc())
                fs=os.statvfs(root);need(fs.f_bavail*fs.f_frsize>=4*1024**3,'original output headroom required')
                need(sha(read(root/'readiness.json'))==obs['files']['readiness.json']['sha256'],'original readiness changed')
                write(evidence/'pre-submission.json',raw(result))
            elif phase=='submit':
                need(not os.path.lexists(root/'submission.intent.json') and not os.path.lexists(root/'submission.result.json') and not os.path.lexists(root/'gpu'),'submission marker exists; never retry')
                need(sha(read(evidence/'complete.json'))==req['completion_sha256'] and sha(read(evidence/'pre-submission.json'))==req['preflight_sha256'],'transfer/fresh readiness identity differs')
                need(sha(read(root/'readiness.json'))==obs['files']['readiness.json']['sha256'],'original readiness differs')
                fresh=env['reconcile']();env['verify']();env['failed_gpu_unchanged']()
                fs=os.statvfs(root);need(fs.f_bavail*fs.f_frsize>=4*1024**3,'original output headroom required')
                intent=req['submission_intent'];need(intent['transfer_continuation']==req['submission_lineage'],'intent continuation binding differs')
                intent=dict(intent,remote_utc=utc(),fresh_readiness=fresh)
                write(root/'submission.intent.json',raw(intent))
                result=dict(utc=utc(),job_id=None,intent_sha256=sha(read(root/'submission.intent.json')))
                try:
                    p=subprocess.run(['sbatch','--parsable',str(root/'job.sbatch')],capture_output=True,text=True,timeout=25)
                    value=p.stdout.strip();valid=re.fullmatch(r'[1-9][0-9]*(;alpine)?',value) is not None
                    result.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,job_id=value.split(';')[0] if p.returncode==0 and valid else None,ambiguous=not(p.returncode==0 and valid))
                except BaseException as exc:
                    txt=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x
                    result.update(ambiguous=True,type=type(exc).__name__,message=str(exc),stdout=txt(getattr(exc,'stdout',None)),stderr=txt(getattr(exc,'stderr',None)))
                write(root/'submission.result.json',raw(result))
            else:raise ValueError('unknown fixed continuation phase')
        print(raw(result).decode(),flush=True)
    except BaseException as exc:
        if evidence.exists():
            try:write(evidence/(phase+'.failure.json'),raw(dict(type=type(exc).__name__,message=str(exc),retry_allowed=False,acceptance=False)))
            except BaseException:pass
        raise
    finally:signal.alarm(0)

BOOTSTRAP = r'''import sys,hashlib,json
def need(ok,message):
 if not ok:raise ValueError(message)
n=int(sys.argv[2]);need(0<n<=1048576,'bounded driver source')
source=sys.stdin.buffer.read(n);need(len(source)==n and hashlib.sha256(source).hexdigest()==sys.argv[1],'independent driver source pin')
header=sys.stdin.buffer.read(8);need(len(header)==8,'request frame truncated')
n=int.from_bytes(header,'big');need(0<n<=134217728,'bounded request')
data=sys.stdin.buffer.read(n);need(len(data)==n,'truncated request')
v={'__name__':'captured_transfer_continuation','__file__':'<pinned-continuation>'}
exec(compile(source,v['__file__'],'exec'),v)
v['remote_entry'](v['decode'](data),sys.stdin.buffer)
'''

def transport(req, source, outputs, *, members=None):
    """One bounded local process group; all streams/partials retained, never retry."""
    phase=req['phase'];limit=630 if phase=='complete' else 1230
    command=['bash',HELPER,'run',shlex.join(['python3','-I','-B','-c',BOOTSTRAP,sha(source),str(len(source))])]
    packet=raw(req);need(len(packet)<=MAX_META,'bounded request metadata')
    for out in outputs:write(out/(phase+'.request.json'),packet)
    logs=outputs[0];start=time.monotonic();errors=[];proc=None;threads=[];primary=None
    result=dict(command=command,utc=utc(),returncode=None,ambiguous=True,retry_allowed=False)
    def send():
        try:
            proc.stdin.write(source+len(packet).to_bytes(8,'big')+packet);proc.stdin.flush()
            for name,info in sorted((members or {}).items()):
                data=read(LOCAL/name,info['sha256']);need(len(data)==info['bytes'],'local member size changed')
                proc.stdin.write(data);proc.stdin.flush()
            proc.stdin.close()
        except BaseException as exc:errors.append(('writer',repr(exc)))
    def drain(pipe,path,cap):
        try:
            total=0
            with path.open('xb') as f:
                while True:
                    data=os.read(pipe.fileno(),65536)
                    if not data:break
                    total+=len(data);need(total<=cap,'bounded remote diagnostic stream')
                    f.write(data);f.flush()
                os.fsync(f.fileno())
        except BaseException as exc:errors.append(('reader',repr(exc)))
    try:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        threads=[threading.Thread(target=send,daemon=True),threading.Thread(target=drain,args=(proc.stdout,logs/(phase+'.stdout'),MAX_META),daemon=True),threading.Thread(target=drain,args=(proc.stderr,logs/(phase+'.stderr'),16*1024**2),daemon=True)]
        for t in threads:t.start()
        while proc.poll() is None:
            need(not errors,'transport stream error: '+repr(errors))
            if time.monotonic()-start>=limit:raise TimeoutError('absolute continuation transport deadline')
            time.sleep(.05)
        for t in threads:t.join(timeout=2)
        need(not any(t.is_alive() for t in threads) and not errors,'transport stream completion uncertain: '+repr(errors))
        result.update(returncode=proc.returncode,ambiguous=proc.returncode!=0)
    except BaseException as exc:
        primary=exc
        result.update(type=type(exc).__name__,message=str(exc))
        raise
    finally:
      with cleanup_signals(primary):
        if proc is not None:
            # Owned local group only; remote source has its own absolute alarm.
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            except OSError as exc:result['local_group_cleanup_error']=repr(exc)
            try:proc.wait(timeout=5)
            except BaseException as exc:result['local_wait_error']=repr(exc)
            for t in threads:t.join(timeout=5)
            result['stream_threads_closed']=not any(t.is_alive() for t in threads)
            result['owned_group_probe']=confirm_owned_group_closed(proc.pid)
            result['owned_process_group_closed']=result['owned_group_probe']['closed']
        result['elapsed_seconds']=time.monotonic()-start;result['stream_errors']=errors
        publication_errors=[]
        actions=[(out/(phase+'.transport.json'),raw(result)) for out in outputs]
        for suffix in ('stdout','stderr'):
            p=logs/(phase+'.'+suffix)
            try:
                if p.exists():actions.append((outputs[1]/p.name,read(p,maximum=MAX_META)))
            except BaseException as exc:publication_errors.append(repr(exc))
        for path,data in actions:
            try:write(path,data)
            except BaseException as exc:publication_errors.append(repr(exc))
        if publication_errors:
            if primary is None:raise RuntimeError('transport evidence publication failed: '+repr(publication_errors))
            if hasattr(primary,'add_note'):primary.add_note('secondary evidence publication errors: '+repr(publication_errors))
    need(result['returncode']==0 and result['stream_threads_closed'] and result['owned_process_group_closed'] is True and not any(k.endswith('_error') for k in result),'failed or ambiguous transport; never retry')
    return decode(read(logs/(phase+'.stdout'),maximum=MAX_META))

def committed(revision,path):
    need(re.fullmatch('[0-9a-f]{40}',revision) is not None,'full Git commit required')
    p=subprocess.run(['git','--no-replace-objects','show',revision+':'+str(Path(path).relative_to(REPO))],cwd=REPO,capture_output=True,timeout=30)
    need(p.returncode==0,'Git object unavailable');return p.stdout

def authenticate_packet():
    packet=decode(read(LOCAL/'packet.json',PACKET_SHA));pins={**packet['new_input_files'],'packet.json':PACKET_SHA};sizes={}
    for name,h in pins.items():
        safe_name(name);data=read(LOCAL/name,h)
        need(data==read(EXT/name,h)==committed(PACKET_COMMIT,LOCAL/name),'original committed packet copies differ')
        sizes[name]=len(data)
    need(read(REPO/REVIEW_PATH,REVIEW_SHA)==committed(PACKET_COMMIT,REPO/REVIEW_PATH),'original committed packet review differs')
    builder=read(LOCAL/'build_packet.py',BUILDER_SHA);launcher=read(LOCAL/'launch_once.py',LAUNCHER_SHA)
    b=types.ModuleType('captured_original_gpu02_builder');b.__file__=str(LOCAL/'build_packet.py');exec(compile(builder,b.__file__,'exec'),vars(b))
    need(packet['schema']==b.OUTER_VERSION and packet['remote_root']==b.ROOT==ROOT and packet['source_commit']==b.SOURCE_COMMIT and packet['resources']==b.RESOURCES and packet['limits']==b.LIMITS,'original frozen packet differs')
    proofs={}
    for role,(name,h) in PROOFS.items():
        data=read(REPO/name,h);need(data==committed(PROOF_COMMIT,REPO/name),'committed failure/probe bytes differ');proofs[role]=decode(data)
        if role in ('transfer','failure'):need(data==read(EXT/Path(name).name,h),'original failed local/external copies differ')
    members=validate_proofs(proofs,pins,sizes)
    need(sha(read(LOCAL/PARTIAL)[:PARTIAL_BYTES])==PARTIAL_SHA,'truncation is not an exact original prefix')
    need(read(LOCAL/'readiness.json',proofs['observation']['files']['readiness.json']['sha256'])==read(EXT/'readiness.json'),'original readiness copies differ')
    return b,packet,pins,proofs,members,launcher

def no_local_intents():
    need(not any(os.path.lexists(p/n) for p in (LOCAL,EXT) for n in ('submission.intent.json','submission.result.json')),'original canonical submission already claimed')

def run_main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('self-sha256','driver-commit','driver-review','driver-review-sha256','projects-backup'):
        ap.add_argument('--'+name,required=True)
    a=ap.parse_args();source=read(Path(__file__).resolve(),a.self_sha256,1024**2)
    need(source==committed(a.driver_commit,Path(__file__).resolve()),'new continuation source commit differs')
    review_path=Path(a.driver_review).resolve(strict=True);review=read(review_path,a.driver_review_sha256)
    need(review.strip() and review==committed(a.driver_commit,review_path),'independent committed continuation review required')
    need(read(REPO/DESIGN_REVIEW_PATH,DESIGN_REVIEW_SHA)==committed(a.driver_commit,REPO/DESIGN_REVIEW_PATH),'new driver commit must contain the pinned operational design review')
    locks=[];outputs=[];primary=None;preservation=preservation_state()
    try:
        for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
            p=BASE/name;need(p==p.resolve(strict=True),'lock alias')
            fd=os.open(p,os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK)
            try:
                st=os.fstat(fd);need(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_uid==os.getuid(),'owned regular lock')
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:os.close(fd);raise
            locks.append(fd)
        no_local_intents();local_processes=no_local_transfer()
        b,packet,pins,proofs,members,launcher=authenticate_packet()
        for dest in (LOCAL,EXT):need(not os.path.lexists(dest/OP),'continuation already attempted; no reuse')
        for dest in (LOCAL,EXT):
            fs=os.statvfs(dest);need(fs.f_bavail*fs.f_frsize>=512*1024**2,'local operation evidence headroom')
            outputs.append(new_dir(dest/OP))
        def save(name,value):
            for out in outputs:write(out/name,raw(value))
        lineage=dict(schema='ace-gpu02-transfer-continuation-v1',packet_sha256=PACKET_SHA,packet_commit=PACKET_COMMIT,packet_review_sha256=REVIEW_SHA,
            driver_sha256=a.self_sha256,driver_commit=a.driver_commit,driver_review_sha256=a.driver_review_sha256,
            design_review_sha256=DESIGN_REVIEW_SHA,
            proof_commit=PROOF_COMMIT,proof_pins={k:v[1] for k,v in PROOFS.items()},remote_root=ROOT,projects_backup=a.projects_backup,
            operation_name=OP,transfer_seconds=600,transport_seconds=630,acceptance=False,retry_allowed=False)
        save('intent.json',lineage);save('local-transfer-processes.json',local_processes)
        for out in outputs:write(out/'continue_once.py',source);write(out/'independent-review.md',review)
        supplied={n:b.read(LOCAL/b.metadata_name(n),pins[b.metadata_name(n)],b.metadata_maximum(n)) for n in b.METADATA}
        ctx=b.authenticate(supplied);m=ctx['metadata']
        b.start_storage_operations(ctx,outputs[1]/'storage-operations',tuple(outputs),dict(phase='transfer-continuation',packet_sha256=PACKET_SHA,packet_commit=PACKET_COMMIT))
        b.configure_packet_storage(ctx,LOCAL,pins,packet);cpu_data=b.admit_cpu_bytes(ctx,Path(packet['cpu_remote_custody']))
        need(packet['baseline_sha256']==ctx['pins']['gpu-failure-baseline'] and packet['cpu_job_id']==m['cpu-acceptance']['job_id'] and packet['operational_amendment_sha256']==ctx['pins']['operational-amendment'] and packet['namespace_claim_sha256']==ctx['pins']['namespace-claim'],'original baseline/namespace binding differs')
        need(packet['token_recovery']==ctx['token_recovery'] and packet['failed_token_root']==b.FAILED_TOKEN_ROOT and packet['failed_token_job_id']==b.FAILED_TOKEN_JOB and packet['history_directories']==ctx['history_directories'],'original complete history differs')
        need(packet['gpu_recovery']==ctx['gpu_recovery'] and packet['prior_jobs']==47 and packet['prior_journals']==96 and packet['operational_source_commit']==m['operational-source-manifest']['source_commit'],'original GPU02 recovery differs')
        claim=ctx['pins']['namespace-claim'];need(b.read(b.LOCAL_REGISTRY,claim)==supplied['namespace-claim'],'namespace changed');b.read(b.ORIGINAL_LOCAL_REGISTRY,b.ORIGINAL_CLAIM_SHA)
        stage=decode(read(LOCAL/'inputs/packet.json',pins['inputs/packet.json']));expected,data,_=b.make_packet(ctx,cpu_data,supplied)
        need(raw(stage)==b.raw(expected) and packet['gpu_packet_sha256']==sha(raw(expected)),'original inner packet differs')
        for n,value in data.items():need(read(LOCAL/'inputs'/n,stage['files'][n])==value,'original input reconstruction differs')
        need(read(LOCAL/'inputs/admission.json',pins['inputs/admission.json'])==b.raw(b.admission(ctx,stage)) and packet['original_cpu_inputs']==ctx['upstream_pins'],'original admission/input roster differs')
        context=dict(root=b.ROOT,registry=b.REMOTE_REGISTRY,pins=pins,baseline=m['gpu-failure-baseline'],failed_gpu_root=b.OLDROOT,gpu_output=b.OUTPUT,
            failed_gpu_inventory=ctx['failed_gpu_inventory'],gpu_recovery=ctx['gpu_recovery'],cpu_directories=sorted(n[7:] for n in ctx['observations']['cpu']['directories'] if n.startswith('launch/')),
            archive=m['cpu-packet']['archive'],manifest=m['cpu-packet']['manifest'],namespace_claim_sha256=claim,cpu_inputs=ctx['upstream_pins'],history_directories=ctx['history_directories'],
            model_directory=m['cpu-packet']['model_directory'],model_files=m['cpu-packet']['model_files'],original_registry=b.ORIGINAL_REMOTE_REGISTRY,original_namespace_claim_sha256=b.ORIGINAL_CLAIM_SHA,
            cpu_root=b.CPU_ROOT,cpu_roster={n.removeprefix('launch/'):v['sha256'] for n,v in ctx['observations']['cpu']['files'].items() if n.startswith('launch/')})
        req=dict(context=context,launcher_source=base64.b64encode(launcher).decode(),observation=proofs['observation'],lineage=lineage,projects_backup=a.projects_backup,members=members)
        preservation['phase']='remote_preservation_requested_outcome_unknown'
        preserved=transport(dict(req,phase='preserve'),source,outputs)
        partial=base64.b64decode(preserved['partial_base64'],validate=True);need(len(partial)==PARTIAL_BYTES and sha(partial)==PARTIAL_SHA and sha(raw(preserved['record']))==preserved['record_sha256'],'remote preserved partial identity differs')
        preservation.update(phase='projects_verified_local_copies_in_progress',projects_readback=dict(path=preserved['record']['projects_backup'],bytes=PARTIAL_BYTES,sha256=PARTIAL_SHA,record_sha256=preserved['record_sha256']))
        for out in outputs:
            write(out/'partial.original',partial)
            preservation['local_readbacks'][str(out/'partial.original')]=dict(bytes=len(read(out/'partial.original',PARTIAL_SHA)),sha256=PARTIAL_SHA)
        preservation['phase']='projects_and_both_local_copies_verified'
        ack=dict(partial_sha256=PARTIAL_SHA,partial_bytes=PARTIAL_BYTES,local_copies=decode(raw(preservation['local_readbacks'])),preservation_sha256=preserved['record_sha256'])
        save('partial-local-readback.json',ack)
        req.update(preservation_sha256=preserved['record_sha256'],local_ack=ack,members=members)
        preservation['original_move']='requested_outcome_unknown'
        completed=transport(dict(req,phase='complete'),source,outputs,members=members)
        need(completed['completed_members']==members and completed['matching_original_members_unchanged'] is True,'remote completion disagrees')
        preservation['original_move']='verified_by_completed_remote_phase'
        req['completion_sha256']=sha(raw(completed));save('completion.json',completed)
        preflight=transport(dict(req,phase='preflight'),source,outputs);req['preflight_sha256']=sha(raw(preflight));save('pre-submission.json',preflight)
        for n,h in pins.items():need(read(LOCAL/n,h)==read(EXT/n,h),'original local packet changed')
        need(b.read(b.LOCAL_REGISTRY,claim)==supplied['namespace-claim'],'namespace changed');b.read(b.ORIGINAL_LOCAL_REGISTRY,b.ORIGINAL_CLAIM_SHA)
        no_local_intents()
        continuation=dict(lineage,preservation_sha256=req['preservation_sha256'],local_readback_sha256=sha(raw(ack)),completion_sha256=req['completion_sha256'],preflight_sha256=req['preflight_sha256'])
        intent=dict(utc=b.utc(),packet_commit=PACKET_COMMIT,source_commit=b.SOURCE_COMMIT,storage_binding=b.storage_packet_binding(ctx),packet_sha256=PACKET_SHA,review_sha256=REVIEW_SHA,pins=pins,remote_root=b.ROOT,
            token_recovery=ctx['token_recovery'],failed_token_root=b.FAILED_TOKEN_ROOT,failed_token_job_id=b.FAILED_TOKEN_JOB,baseline_sha256=ctx['pins']['gpu-failure-baseline'],gpu_recovery=ctx['gpu_recovery'],
            operational_source_commit=m['operational-source-manifest']['source_commit'],cpu_acceptance_sha256=ctx['pins']['cpu-acceptance'],cpu_job_id=m['cpu-acceptance']['job_id'],existing_namespace_claim_sha256=claim,
            namespace_admission_sha256=ctx['pins']['namespace-admission'],recovery_authority_sha256=ctx['pins']['recovery-authority'],original_namespace_claim_sha256=b.ORIGINAL_CLAIM_SHA,
            resources=b.RESOURCES,limits=b.LIMITS,admission=packet['admission'],blind_retry_forbidden=True,new_namespace_claimed=False,replacement_worlds=False,transfer_continuation=continuation)
        for out in (LOCAL,EXT):write(out/'submission.intent.json',raw(intent))
        result=transport(dict(req,phase='submit',submission_intent=intent,submission_lineage=continuation),source,outputs)
        for out in (LOCAL,EXT):write(out/'submission.result.json',raw(result))
        need(result.get('job_id') is not None and result.get('returncode')==0 and result.get('ambiguous') is False,'ambiguous submission retained; never retry')
        save('terminal.json',dict(status='one_submission_recorded',job_id=result['job_id'],lineage=continuation,acceptance=False))
        print(raw(result).decode())
    except BaseException as exc:
        primary=exc
        for out in outputs:
            try:write(out/'failure.json',raw(failure_record(exc,preservation)))
            except BaseException:pass
        raise
    finally:
      with cleanup_signals(primary):
        errors=[]
        for fd in reversed(locks):
            for fn in (lambda:fcntl.flock(fd,fcntl.LOCK_UN),lambda:os.close(fd)):
                try:fn()
                except BaseException as exc:errors.append(repr(exc))
        if errors and primary is None:raise RuntimeError('owned lock release errors: '+repr(errors))
        if errors and hasattr(primary,'add_note'):primary.add_note('owned lock release errors: '+repr(errors))

def main():
    with termination_guard():return run_main()

if __name__=='__main__':main()

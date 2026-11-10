"""One authorized metadata-only capture; no build/submit/imported producer."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,resource,signal,subprocess,time
ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
def put(name,obj):
    with (ROOT/name).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def call(name,command,timeout,source=None):
    started=time.monotonic();proc=None
    result=dict(command=command,utc=utc(),timeout_seconds=timeout,returncode=None)
    put(name+'-intent.json',result)
    try:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE if source is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        stdout,stderr=proc.communicate(source,timeout=timeout)
        result['returncode']=proc.returncode
    except subprocess.TimeoutExpired as exc:
        result['timeout']=True;result['ambiguity_preserved']=True
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        stdout,stderr=proc.communicate(timeout=5)
        result['returncode']=proc.returncode
    finally:
        result['elapsed_seconds']=time.monotonic()-started
    for suffix,data in [('stdout',stdout),('stderr',stderr)]:
        with (ROOT/(name+'.'+suffix)).open('xb') as f:f.write(data)
        result[suffix+'_sha256']=hashlib.sha256(data).hexdigest()
    put(name+'-transport.json',result)
    if result['returncode']!=0 or result.get('timeout'):raise RuntimeError('Retained failed metadata transport; no automatic retry: '+name)
    return result
handles=[];started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
try:
    for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
        path=BASE/name
        if path!=path.resolve():raise ValueError('Lock path alias')
        f=os.fdopen(os.open(path,os.O_RDWR|os.O_NOFOLLOW),'r+');handles.append(f)
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=(ROOT/'readiness-source.py').read_bytes()
    if hashlib.sha256(source).hexdigest()!='f207ec822caf8e3e406fe1cd82ecbb25ed06dae1bcb6be02bf7197624b0e2240':raise ValueError('Captured source changed')
    put('command.json',dict(utc=utc(),held_locks=[str(BASE/n) for n in ('ace-delivery-controller.lock','ace-foundation-controller.lock')],
        source_sha256=hashlib.sha256(source).hexdigest(),local_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remote_command=['bash',HELPER,'run','python3 -I -B -'],remote_deadline_seconds=150,local_timeout_seconds=180,
        authorization='user metadata-only fresh CPU readiness plus existing dedicated write/read/cleanup workflow',
        coordination='Leibniz reports no active/planned alias check, no remote operation and no held locks'))
    call('master-check',['bash',HELPER,'check'],15)
    call('readiness',['bash',HELPER,'run','python3 -I -B -'],180,source)
except BaseException as exc:
    put('failure.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise
finally:
    for f in reversed(handles):
        fcntl.flock(f,fcntl.LOCK_UN);f.close()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    put('run.json',dict(utc=utc(),wall_seconds=time.monotonic()-started,
        child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        max_rss_bytes=after.ru_maxrss,locks_released=True,build_invoked=False,submission_invoked=False))

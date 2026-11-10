"""Preserve the single authorized committed launcher invocation; no retries."""
from pathlib import Path
import datetime,hashlib,json,resource,subprocess,time
ROOT=Path(__file__).resolve().parent
EXT=Path((ROOT/'external-path.txt').read_text().strip())
def put(n,value):
    data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    for root in (ROOT,EXT):
        with (root/n).open('xb') as f:f.write(data)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
data=(ROOT/'command.json').read_bytes()
if hashlib.sha256(data).hexdigest()!='ff1be298cb90a26abcf750468b5dd7fe5693179d4391fc7ca4775cf9b8f7df8a':raise ValueError('Exact command changed')
cfg=json.loads(data)
put('invocation.intent.json',dict(utc=utc(),command_sha256=hashlib.sha256(data).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),authorized_once=True,retry_allowed=False))
start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);result={}
try:
    with (ROOT/'stdout.log').open('xb') as out,(ROOT/'stderr.log').open('xb') as err:
        p=subprocess.Popen(cfg['command'],cwd=cfg['cwd'],stdin=subprocess.DEVNULL,stdout=out,stderr=err)
        put('process.json',dict(pid=p.pid,utc=utc(),command_sha256=hashlib.sha256(data).hexdigest()))
        result['returncode']=p.wait()
except BaseException as exc:
    result.update(type=type(exc).__name__,message=str(exc),ambiguous=True);raise
finally:
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(utc=utc(),wall_seconds=time.monotonic()-start,
        child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        max_rss_bytes=after.ru_maxrss,command_sha256=hashlib.sha256(data).hexdigest(),retry_allowed=False)
    for name in ('stdout.log','stderr.log'):
        if (ROOT/name).exists():
            value=(ROOT/name).read_bytes()
            with (EXT/name).open('xb') as f:f.write(value)
            result[name+'_sha256']=hashlib.sha256(value).hexdigest()
    put('run.json',result)
raise SystemExit(result['returncode'])

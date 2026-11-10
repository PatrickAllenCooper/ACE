"""Retain the single authorized build invocation; never submit or retry."""
from pathlib import Path
import datetime,hashlib,json,resource,subprocess,time
ROOT=Path(__file__).resolve().parent
EXT=Path((ROOT/'external-path.txt').read_text().strip())
def raw(v):return (json.dumps(v,indent=2,sort_keys=True)+'\n').encode()
def put(n,v):
    data=raw(v)
    for p in (ROOT/n,EXT/n):
        with p.open('xb') as f:f.write(data)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
command=(ROOT/'command.json').read_bytes()
if hashlib.sha256(command).hexdigest()!='d8637a67030f406b9ef8404a7221320b38f3ea19c57fd94519d95b0d6d6a264f':raise ValueError('Exact reviewed-composite command changed')
cfg=json.loads(command)
put('invocation.intent.json',dict(utc=utc(),command_sha256=hashlib.sha256(command).hexdigest(),
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),submission_authorized=False,automatic_retry=False))
started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);result={}
try:
    with (ROOT/'stdout.log').open('xb') as out,(ROOT/'stderr.log').open('xb') as err:
        p=subprocess.run(cfg['argv'],cwd=cfg['cwd'],stdin=subprocess.DEVNULL,stdout=out,stderr=err)
    result['returncode']=p.returncode
except BaseException as exc:
    result.update(type=type(exc).__name__,message=str(exc),ambiguous=True)
    raise
finally:
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(utc=utc(),wall_seconds=time.monotonic()-started,
        child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,max_rss_bytes=after.ru_maxrss,
        command_sha256=hashlib.sha256(command).hexdigest(),submission_invoked=False,automatic_retry=False)
    for name in ('stdout.log','stderr.log'):
        if (ROOT/name).exists():
            data=(ROOT/name).read_bytes()
            with (EXT/name).open('xb') as f:f.write(data)
            result[name+'_sha256']=hashlib.sha256(data).hexdigest()
    put('run.json',result)
raise SystemExit(result['returncode'])

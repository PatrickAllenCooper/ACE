"""Local parent-command evidence wrapper; runs only the supplied pinned command."""
from pathlib import Path
import datetime
import hashlib
import json
import resource
import subprocess
import sys
import time

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
root=Path(__file__).resolve().parent
label=sys.argv[1]
if not label.replace('-','').isalnum():raise ValueError('unsafe command label')
command_path=root/(label+'.command.json')
config=json.loads(command_path.read_bytes())
attempt=root/label;attempt.mkdir(mode=0o700)
intent=dict(utc=utc(),command=config['command'],config_sha256=hashlib.sha256(command_path.read_bytes()).hexdigest(),
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),automatic_retry=False)
(attempt/'intent.json').write_text(json.dumps(intent,sort_keys=True,indent=2)+'\n')
start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
result=dict(started_utc=utc(),returncode=None,automatic_retry=False)
try:
    with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
        p=subprocess.run(config['command'],cwd=config['cwd'],stdout=out,stderr=err)
    result['returncode']=p.returncode
except BaseException as exc:
    result.update(type=type(exc).__name__,message=str(exc));raise
finally:
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(ended_utc=utc(),elapsed_seconds=time.monotonic()-start,
                  child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                  maxrss_platform_units=after.ru_maxrss,platform=sys.platform,
                  command_config_sha256=intent['config_sha256'])
    with (attempt/'run.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(result))
sys.exit(result['returncode'])

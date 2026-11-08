"""Local metadata-only GPU packet construction after resolved actual acceptance inputs."""
from pathlib import Path
import contextlib,fcntl,hashlib,json,resource,subprocess,time,datetime,os
HERE=Path(__file__).resolve().parent
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
SOURCE_SHA='6227e6a2da68bb53b2dc255066678e42f5b7adf56edaeb3dbeed10227ea806e1'
def raw(v):return (json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
def write(p,b):
 with p.open('xb') as f:f.write(b)
assert hashlib.sha256((HERE/'build_packet.py').read_bytes()).hexdigest()==SOURCE_SHA
config=json.loads((HERE/'argument-map.resolved.json').read_text())
args=['python3','-B',str(HERE/'build_packet.py')]
for name,item in sorted(config['metadata'].items()):
 assert type(item['path']) is str and type(item['sha256']) is str and len(item['sha256'])==64
 args+=['--'+name,item['path'],'--'+name+'-sha256',item['sha256']]
for name,value in config['inventory_relative_paths'].items():args+=['--'+name,value]
item=config['additional_required']['prepared'];args+=['--prepared',item['path'],'--prepared-sha256',item['sha256'],'--plan-sha256',config['additional_required']['plan_sha256']]
with contextlib.ExitStack() as stack:
 held=[]
 for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
  f=stack.enter_context((BASE/name).open('a+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(str(BASE/name))
 out=HERE/'build01';out.mkdir()
 write(out/'command.json',raw(args));write(out/'controller.json',raw(dict(pid=os.getpid(),held=held,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
 start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
 with (out/'stdout.log').open('xb') as so,(out/'stderr.log').open('xb') as se:r=subprocess.run(args,cwd='/tmp',stdout=so,stderr=se)
 after=resource.getrusage(resource.RUSAGE_CHILDREN)
 info=dict(returncode=r.returncode,elapsed_seconds=time.monotonic()-start,child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,peak_child_rss_bytes=after.ru_maxrss,scope='metadata-only build, including Git children; no models, GPU allocation, training or simulator responses')
 write(out/'execution.json',raw(info));print(json.dumps(info));raise SystemExit(r.returncode)

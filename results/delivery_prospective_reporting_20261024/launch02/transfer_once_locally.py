from pathlib import Path
import json,hashlib,subprocess,base64,shlex
p=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-reporting-launch-20261024-02');pins=json.loads((p/'input_pins.json').read_text());reg=json.loads((p/'report_freeze.json').read_text())
s='''import sys,json,base64,os,stat,hashlib
value=json.load(sys.stdin);root=value['root'];fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);info=os.fstat(fd)
if info.st_ino!=value['inode'] or info.st_uid!=os.geteuid() or info.st_mode&0o077:raise ValueError('output identity differs')
if set(os.listdir(fd))!={'output_claim.json'}:raise ValueError('unexpected existing output; reconcile')
for name,item in value['files'].items():
 if '/' in name or name in ('.','..'):raise ValueError('basename required')
 raw=base64.b64decode(item['data'],validate=True)
 if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('transport payload changed')
 if name=='output_claim.json':
  h=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
  with os.fdopen(h,'rb') as f:
   if f.read()!=raw:raise ValueError('claim differs')
 else:
  h=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
  with os.fdopen(h,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
actual={}
for name in value['files']:
 h=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 with os.fdopen(h,'rb') as f:actual[name]=hashlib.sha256(f.read()).hexdigest()
if actual!={n:v['sha256'] for n,v in value['files'].items()}:raise ValueError('remote readback differs')
if set(os.listdir(fd))!=set(actual):raise ValueError('remote membership differs')
print(json.dumps({'pins':actual,'inode':info.st_ino,'files':len(actual)}))
'''
(p/'transfer_remote.py').write_text(s)
payload={'root':reg['output'],'inode':reg['output_inode'],'files':{n:{'sha256':pin,'data':base64.b64encode((p/n).read_bytes()).decode()} for n,pin in pins.items()}}
command='/scratch/alpine/paco0228/ACE/envs/neuronbench_py311/bin/python -I -B -c '+shlex.quote(s)
r=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run',command],input=json.dumps(payload),text=True,capture_output=True,timeout=90)
(p/'transfer.stdout').write_text(r.stdout);(p/'transfer.stderr').write_text(r.stderr)
(p/'transfer_transport.json').write_text(json.dumps({'returncode':r.returncode,'payload_bytes':len(json.dumps(payload).encode())})+'\n')
if r.returncode:raise RuntimeError('transfer failed; preserve/reconcile')
a=json.loads(r.stdout);assert a['pins']==pins;(p/'remote_input_pins.json').write_text(json.dumps(a,indent=2)+'\n');print({'files':len(pins),'verified':True})

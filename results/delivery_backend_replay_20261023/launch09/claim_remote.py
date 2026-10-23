import os,json,stat,hashlib,subprocess
root='/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261023_09'
claim=b'{\n  "schema": "ace-exclusive-replay-output-v1",\n  "at": "2026-10-23T22:40:35.611099+00:00",\n  "nonce": "daa3b083-7486-4211-a049-5db8f946ee06",\n  "output": "/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261023_09",\n  "authorization_sha256": "192dab9c468f6ccb0b23c45cbf606085a5a21a5a5f4e7b5f2158b09246066f82",\n  "controller_pid": 45379\n}\n'
queue=subprocess.check_output(['squeue','-h','-u','paco0228','-o','%i|%j|%T|%Z'],text=True,timeout=20)
ace=[r for r in queue.splitlines() if r.split('|')[1].lower().startswith('ace_') or '/ACE/' in r.split('|')[-1] or r.split('|')[-1].endswith('/ACE')]
if ace:raise ValueError('ACE queue is not empty')
os.mkdir(root,0o700)
fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
i=os.fstat(fd)
if i.st_uid!=os.geteuid() or i.st_mode&0o077:raise ValueError('output owner/mode differs')
h=os.open('output_claim.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
with os.fdopen(h,'wb') as f:f.write(claim);f.flush();os.fsync(f.fileno())
os.fsync(fd)
print(json.dumps({'inode':i.st_ino,'mode':oct(i.st_mode),'uid':i.st_uid,'queue':ace,'claim_sha256':hashlib.sha256(claim).hexdigest()},indent=2))

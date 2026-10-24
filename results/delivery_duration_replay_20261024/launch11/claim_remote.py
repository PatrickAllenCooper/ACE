import os,json,stat,hashlib,subprocess
root='/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261024_11'
claim=b'{\n  "schema": "ace-exclusive-replay-output-v1",\n  "at": "2026-10-24T18:15:54.282021+00:00",\n  "nonce": "53f815c5-5bfa-48b7-8ea8-83541625ccf7",\n  "output": "/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261024_11",\n  "authorization_sha256": "c10b5b8c2b830361ed8dcce6ec7e0f253dd79ac8df534e68a3f7531a6538a78c",\n  "controller_pid": 92730,\n  "readiness_sha256": "4f353ad2bc5111dd5af15ec44902a0e778402d1df2b22779480b836b2bf9f5d0"\n}\n'
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

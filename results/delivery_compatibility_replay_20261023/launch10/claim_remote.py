import os,json,stat,hashlib,subprocess
root='/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261023_10'
claim=b'{\n  "schema": "ace-exclusive-replay-output-v1",\n  "at": "2026-10-24T00:46:18.410065+00:00",\n  "nonce": "c197b56f-56a6-4a74-9a03-38214889d30d",\n  "output": "/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261023_10",\n  "authorization_sha256": "abd4c5125a5e7a8523e46839ece8c3aae6cdee758dc08dcb89d6ffc8b55a5c8f",\n  "controller_pid": 97219,\n  "readiness_sha256": "6071a11bef56a00a339de04ecf0cadb78c5280d7f6c50de633ae7fa8aff9397b"\n}\n'
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

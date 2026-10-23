import os,json,stat,hashlib,subprocess
root='/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261023_07'
claim=b'{\n  "schema": "ace-exclusive-replay-output-v1",\n  "at": "2026-10-23T16:49:48.789773+00:00",\n  "nonce": "7e969263-8d88-4301-b921-de56cbca0c7e",\n  "output": "/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261023_07",\n  "authorization_sha256": "350372308cc93fc199ea5c00a1c838c80b94594b0273de7443fa3ddef3aabe91",\n  "controller_pid": 59431\n}\n'
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

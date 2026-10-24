import os,json,stat,hashlib,subprocess
root='/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261024_12'
claim=b'{\n  "schema": "ace-exclusive-replay-output-v1",\n  "at": "2026-10-24T19:28:43.766407+00:00",\n  "nonce": "ef874d0f-add7-4a04-a415-06ef3ada8d3d",\n  "output": "/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261024_12",\n  "authorization_sha256": "5a30ada2f990701c415ad51f97c2299e46226dbc839263fd1069d93cae592a5d",\n  "controller_pid": 22207,\n  "readiness_sha256": "e601330bef56a80eb603c85f720e3dd46146852ab21d96c2ca06d705f7157c1b"\n}\n'
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

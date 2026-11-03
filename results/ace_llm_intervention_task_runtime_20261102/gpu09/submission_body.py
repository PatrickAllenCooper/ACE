root=pathlib.Path(manifest['remote_root']);inputs=root/'input'
if {x.name for x in inputs.iterdir()}!=set(manifest['files']):raise RuntimeError('input membership mismatch')
for name,pin in manifest['files'].items():
 if hashlib.sha256((inputs/name).read_bytes()).hexdigest()!=pin:raise RuntimeError('input hash mismatch '+name)
queue=subprocess.run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%C|%b'],capture_output=True,text=True,check=True).stdout
acct=subprocess.run(['sacct','-X','-u','paco0228','-S','2026-11-02','--name','ace_llm_check09','--format','JobIDRaw,JobName,State,Submit','-n','-P'],capture_output=True,text=True,check=True).stdout
if any('ace_llm' in line for line in queue.splitlines()) or 'ace_llm_check09' in acct:raise RuntimeError('matching prior/current job requires reconciliation')
probe=root/'submission-readiness-probe';value=str(uuid.uuid4()).encode()
with probe.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
if probe.read_bytes()!=value:raise RuntimeError('write/read failed')
probe.unlink()
stat=root.stat()
if stat.st_uid!=os.getuid():raise RuntimeError('unexpected output owner')
intent.update({'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authenticated_inputs':len(manifest['files']),'squeue':queue,'matching_sacct':acct,'write_read_cleanup':True,'root_uid':stat.st_uid,'df':subprocess.run(['df','-h',str(root)],capture_output=True,text=True,check=True).stdout})
with (root/'submission.intent.json').open('x') as f:json.dump(intent,f,indent=2);f.flush();os.fsync(f.fileno())
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'intent':intent,'input_manifest':manifest}
try:
 result=subprocess.run(['sbatch','--parsable',str(inputs/'gpu09.sbatch')],capture_output=True,text=True,timeout=30)
 job=result.stdout.strip().split(';')[0]
 receipt.update({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'job_id':job if result.returncode==0 and job.isdigit() else None})
except BaseException as exc:
 receipt.update({'job_id':None,'ambiguous':True,'exception_type':type(exc).__name__,'message':str(exc)})
with (root/'submission.result.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt))
if receipt['job_id'] is None:raise RuntimeError('submission failed or ambiguous, never blind retry')

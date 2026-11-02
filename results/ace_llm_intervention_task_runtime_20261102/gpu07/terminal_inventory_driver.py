import pathlib,json,hashlib,datetime,subprocess
root=pathlib.Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_precision_gpu_20261102_07')
acct=subprocess.run(['sacct','-j','35209701','--format','JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Account,Partition,QOS,ReqMem,Timelimit,TotalCPU,MaxRSS,Start,End','-P'],capture_output=True,text=True,check=True).stdout
if not any(x in acct for x in ('COMPLETED|','FAILED|','TIMEOUT|','OUT_OF_MEMORY|')):raise RuntimeError('not terminal')
objects={}
for p in sorted(root.rglob('*')):
 if p.is_symlink():raise RuntimeError('symlink')
 if p.is_file():
  raw=p.read_bytes();objects[p.relative_to(root).as_posix()]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'files':objects,'file_count':len(objects),'bytes':sum(x['bytes'] for x in objects.values()),'sacct':acct},indent=2))

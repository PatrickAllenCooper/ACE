import pathlib,hashlib,json,subprocess,datetime,os
root=pathlib.Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261104_10')
files={}
for p in sorted(root.rglob('*')):
 if p.is_symlink():raise ValueError('symlink '+str(p))
 if p.is_file():
  b=p.read_bytes();files[str(p.relative_to(root))]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
commands={
 'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R'],
 'accounting':['sacct','-j','35252306','-P','-n','--format=JobIDRaw,JobID,JobName,Account,Partition,QOS,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,AllocTRES,Start,End,TotalCPU,MaxRSS'],
 'related':['sacct','-X','-u','paco0228','-S','2026-11-02','-P','-n','--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES']}
results={}
for name,args in commands.items():
 p=subprocess.run(args,capture_output=True,text=True,timeout=30);results[name]={'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 if p.returncode:raise ValueError(results[name])
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':os.uname().nodename,'root':str(root),'files':files,'scheduler':results},sort_keys=True,indent=2))

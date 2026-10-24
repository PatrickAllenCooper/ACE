import datetime,hashlib,json,pathlib,subprocess,stat
p=pathlib.Path('/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261024_12')
files=[]
for f in sorted(p.rglob('*')):
 if f.is_symlink():raise ValueError('unexpected symlink')
 if f.is_file():
  raw=f.read_bytes();files.append({'path':str(f.relative_to(p)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 elif not f.is_dir():raise ValueError('nonregular output')
commands={
 'sacct':['sacct','-j','34863006','--noheader','--parsable2','--format=JobIDRaw,JobID,State,ExitCode,Start,End,ElapsedRaw,CPUTimeRAW,TotalCPU,AllocCPUS,ReqMem,MaxRSS,AllocTRES,Account,Partition,QOS,NodeList'],
 'scontrol':['scontrol','show','job','34863006','-o'],
 'squeue':['squeue','-h','-u','paco0228','-n','ace_delB_replay','-o','%i|%j|%T|%R']}
records={}
for k,c in commands.items():
 r=subprocess.run(c,capture_output=True,text=True,timeout=25);records[k]={'argv':c,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(p),'files':files,'file_count':len(files),'bytes':sum(f['bytes'] for f in files),'commands':records},indent=2))

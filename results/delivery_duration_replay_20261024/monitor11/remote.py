import subprocess,json,datetime,pathlib
p=pathlib.Path('/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261024_11')
r={}
for k,c in {'sacct':['sacct','-j','34861753','--noheader','--parsable2','--format=JobIDRaw,JobID,State,ExitCode,ElapsedRaw,CPUTimeRAW,TotalCPU,AllocCPUS,ReqMem,MaxRSS,AllocTRES,Account,Partition,QOS'],'squeue':['squeue','-h','-j','34861753','-o','%i|%j|%T|%R']}.items():
 q=subprocess.run(c,capture_output=True,text=True,timeout=20);r[k]={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
d={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':r,'files':{x.name:x.stat().st_size for x in p.iterdir() if x.is_file()}}
for n in ['replay_execution.json','launcher_failure.json']:
 if (p/n).is_file():
  q=json.loads((p/n).read_bytes());d[n]={k:q.get(k) for k in ['status','stage','exit_code','failure_type','failure_reason','finished_at','new_fits','new_responses']}
print(json.dumps(d,indent=2))

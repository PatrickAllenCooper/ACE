import subprocess,json,datetime,pathlib
p=pathlib.Path('/scratch/alpine/paco0228/ACE/results/delivery_qualified_replay_20261024_11')
commands={'scontrol':['scontrol','show','job','34861753','-o'],'squeue':['squeue','-h','-j','34861753','-o','%i|%j|%T|%R'],'sacct':['sacct','-j','34861753','--noheader','--parsable2','--format=JobIDRaw,JobID,State,ExitCode,AllocCPUS,ReqMem,AllocTRES,Account,Partition,QOS,Timelimit,ElapsedRaw,CPUTimeRAW']}
r={}
for k,c in commands.items():
 q=subprocess.run(c,text=True,capture_output=True,timeout=25);r[k]={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
print(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':34861753,'commands':r,'files':{x.name:x.stat().st_size for x in p.iterdir() if x.is_file()}},indent=2))

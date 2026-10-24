import subprocess,json,datetime
commands={'scontrol':['scontrol','show','job','34867191','-o'],'sacct':['sacct','-X','-j','34867191','--noheader','--parsable2','--format=JobIDRaw,JobID,State,ExitCode,ElapsedRaw,CPUTimeRAW,AllocTRES,Account,Partition,QOS']}
r={}
for k,c in commands.items():
 p=subprocess.run(c,capture_output=True,text=True,timeout=20);r[k]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':r},indent=2))

import subprocess,json,datetime
commands={}
for name,cmd in [('scontrol',['scontrol','show','job','34864950','-o']),('sacct',['sacct','-X','-j','34864950','--noheader','--parsable2','--format=JobIDRaw,JobID,State,ExitCode,AllocCPUS,ElapsedRaw,CPUTimeRAW,ReqMem,AllocTRES,Account,Partition,QOS'])]:
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=15);commands[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':commands},indent=2))

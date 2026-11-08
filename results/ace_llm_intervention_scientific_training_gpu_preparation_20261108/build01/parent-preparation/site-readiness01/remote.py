import subprocess,json,datetime
commands={"hostname":["hostname"],"queue":["squeue","-u","paco0228","-h","-o","%i|%j|%T|%a|%R"],"gpu_types":["sinfo","-p","aa100","-h","-o","%P|%a|%l|%G"],"partition":["scontrol","show","partition","aa100","-o"],"association":["sacctmgr","show","assoc","where","user=paco0228","account=ucb736_asc2","format=Account,User,Partition,QOS,MaxJobs,MaxSubmitJobs,GrpTRES,MaxTRES","-n","-P"]}
results={}
for name,args in commands.items():
 p=subprocess.run(args,capture_output=True,text=True,timeout=20);results[name]=dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope="read_only_site_observation_no_submission",results=results)))

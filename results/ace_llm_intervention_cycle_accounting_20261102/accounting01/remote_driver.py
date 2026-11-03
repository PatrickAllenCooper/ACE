import subprocess,json,datetime
ids=['35200690','35202401','35202931','35203022','35204167','35204547','35207064','35208932','35209536','35209701','35209904','35210147']
a=subprocess.check_output(['sacct','-X','-j',','.join(ids),'--format','JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Account,Partition,QOS,Start,End','-n','-P'],text=True)
q=subprocess.check_output(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%C|%b'],text=True)
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_ids':ids,'sacct':a,'squeue':q},indent=2))

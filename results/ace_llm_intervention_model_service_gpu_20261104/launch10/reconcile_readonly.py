import subprocess,json,pathlib,datetime
base=pathlib.Path('/scratch/alpine/paco0228/ACE/results')
queue=subprocess.run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%C|%b'],capture_output=True,text=True,check=True).stdout
p=subprocess.run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'],capture_output=True,text=True,check=True)
rows=[line for line in p.stdout.splitlines() if 'ace_llm' in line]
journals={str(f):json.loads(f.read_bytes()) for d in sorted(base.glob('ace_llm_intervention*')) for f in sorted(d.glob('submission.*.json'))}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queue':queue,'accounting':rows,'journals':journals,'controller_pid_local':7883}))

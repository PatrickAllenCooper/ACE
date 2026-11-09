from pathlib import Path
import os,stat,json,subprocess,tempfile,datetime,signal
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("bounded readiness expired")))
signal.alarm(150)
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':{}}
for n,c in {'gpu_partition':['scontrol','show','partition','aa100','-o'],'gpu_nodes':['sinfo','-p','aa100','-h','-o','%P|%a|%l|%G'],'hostname':['hostname'],'quota':['curc-quota'],'df':['df','-Pk','/projects/paco0228','/scratch/alpine/paco0228'],'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],'accounting':['sacct','-X','-u','paco0228','-S','2026-11-02','--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'],'association':['sacctmgr','-n','-P','show','assoc','where','user=paco0228','account=ucb736_asc2','format=User,Account,Partition,QOS,MaxJobs,MaxSubmitJobs']}.items():
 r=subprocess.run(c,capture_output=True,text=True,timeout=25);out['commands'][n]={'command':c,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
assert all(v['returncode']==0 for v in out['commands'].values())
assert not any('ace_llm' in l for l in out['commands']['queue']['stdout'].splitlines())
parent=Path('/scratch/alpine/paco0228/ACE/results');assert parent.resolve()==parent
probe=Path(tempfile.mkdtemp(prefix='ace-remote-custody-readiness-',dir=parent))
f=probe/'probe'
with f.open('xb') as s:s.write(b'ACE remote custody readiness\n');s.flush();os.fsync(s.fileno())
assert f.read_bytes()==b'ACE remote custody readiness\n'
f.unlink();probe.rmdir()
out['scratch_write_read_cleanup']=True
for n in ['/projects/paco0228','/scratch/alpine/paco0228/ACE/results']:
 s=Path(n).stat();v=os.statvfs(n);out[n]={'uid':s.st_uid,'mode':oct(stat.S_IMODE(s.st_mode)),'canonical':str(Path(n).resolve()),'available_bytes':v.f_bavail*v.f_frsize}
print(json.dumps(out,sort_keys=True))

import pathlib,json,hashlib,subprocess,datetime,signal,os,stat
signal.signal(signal.SIGALRM,lambda a,b:(_ for _ in ()).throw(TimeoutError('bounded failure observation')));signal.alarm(50)
root=pathlib.Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01')
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'35498218','files':{},'commands':{},'gpu_output_exists':os.path.lexists(root/'gpu')}
for rel in ('logs/35498218.err','logs/35498218.out','submission.intent.json','submission.result.json'):
 p=root/rel;st=p.lstat();assert stat.S_ISREG(st.st_mode) and not p.is_symlink() and st.st_size<=2*1024**2
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:data=f.read(2*1024**2+1);after=os.fstat(f.fileno())
 assert len(data)<=2*1024**2 and (st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 v['files'][rel]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode()}
commands={'accounting':['sacct','-j','35498218','-n','-P','--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180,MaxRSS,TotalCPU'],'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],'details':['scontrol','show','job','35498218','-o']}
for k,a in commands.items():
 p=subprocess.run(a,capture_output=True,text=True,timeout=12);v['commands'][k]={'argv':a,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(v,sort_keys=True))

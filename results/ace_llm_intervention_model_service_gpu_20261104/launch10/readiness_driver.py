import pathlib,subprocess,json,os,uuid,datetime
base=pathlib.Path('/scratch/alpine/paco0228/ACE/results')
queue=subprocess.run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%C|%b'],capture_output=True,text=True,check=True).stdout
accounting=subprocess.run(['sacct','-X','-u','paco0228','-S','2026-11-02','--name','ace_llm_service10','--format','JobIDRaw,JobName,State,Submit','-n','-P'],capture_output=True,text=True,check=True).stdout
if 'ace_llm' in queue or accounting.strip():raise RuntimeError('related active or same-name submission requires reconciliation')
st=base.stat();fs=os.statvfs(base)
if st.st_uid!=os.getuid() or fs.f_bavail*fs.f_frsize<1024**3:raise RuntimeError('scratch ownership/free-space gate failed')
root=base/'ace_llm_intervention_model_service_gpu_20261104_10'
root.mkdir();(root/'input').mkdir();(root/'logs').mkdir()
probe=root/'readiness-probe';value=('ace-readiness-'+str(uuid.uuid4())).encode()
with probe.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
if probe.read_bytes()!=value:raise RuntimeError('readback mismatch')
probe.unlink()
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'base_uid':st.st_uid,'base_mode':oct(st.st_mode & 0o7777),'free_bytes':fs.f_bavail*fs.f_frsize,'df':subprocess.run(['df','-h',str(base)],capture_output=True,text=True,check=True).stdout,'write_read_cleanup':True,'queue':queue,'same_name_accounting':accounting,'controller_lock_pid_local':7883,'submitted':False}
with (root/'readiness.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))

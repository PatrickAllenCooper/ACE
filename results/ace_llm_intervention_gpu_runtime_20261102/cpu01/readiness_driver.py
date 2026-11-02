import pathlib,subprocess,json,os,uuid,datetime
base=pathlib.Path('/scratch/alpine/paco0228/ACE/results')
queue=subprocess.run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%C|%b'],capture_output=True,text=True,check=True).stdout
accounting=subprocess.run(['sacct','-X','-u','paco0228','-S','2026-11-02','--name','ace_llm_gcpu01','--format','JobIDRaw,JobName,State,Submit','-n','-P'],capture_output=True,text=True,check=True).stdout
if 'ace_llm_gcpu01' in queue+accounting:raise RuntimeError('existing matching job requires reconciliation')
journals={}
for folder in sorted(base.glob('ace_llm_intervention*')):
 for p in sorted(folder.glob('submission.*.json')):
  journals[str(p)]=json.loads(p.read_text())
root=base/'ace_llm_intervention_gpu_cpu_20261102_01'
root.mkdir();(root/'input').mkdir();(root/'logs').mkdir()
probe=root/'readiness-probe'
value=('ace-readiness-'+str(uuid.uuid4())).encode()
with probe.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
if probe.read_bytes()!=value:raise RuntimeError('readiness readback failed')
probe.unlink()
st=base.stat()
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'uid':os.getuid(),'base_uid':st.st_uid,'base_mode':oct(st.st_mode & 0o7777),'df':subprocess.run(['df','-h',str(base)],capture_output=True,text=True,check=True).stdout,'write_read_cleanup':True,'squeue':queue,'matching_sacct':accounting,'prior_submission_journals':journals}
with (root/'readiness.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))

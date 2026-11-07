import subprocess,pathlib,os,json,datetime,uuid
base=pathlib.Path('/scratch/alpine/paco0228/ACE/results')
queue=subprocess.run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R'],capture_output=True,text=True,check=True).stdout
if 'ace_llm' in queue:raise RuntimeError('related active job')
a=subprocess.run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'],capture_output=True,text=True,check=True).stdout
rows=[l for l in a.splitlines() if 'ace_llm' in l]
parsed=[l.split('|') for l in rows]
if len(parsed)!=13 or any(v[2].split()[0] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for v in parsed):raise RuntimeError('unreconciled prior allocation')
cpu=sum(int(v[4])*int(v[5]) for v in parsed)
gpu=sum(int(v[5]) for v in parsed if 'gres/gpu:' in v[7])
if (cpu,gpu)!=(4450,836):raise RuntimeError('accounting changed '+str((cpu,gpu)))
journals={str(p):json.loads(p.read_bytes()) for d in base.glob('ace_llm_intervention*') for p in d.glob('submission.*.json')}
if len(journals)!=28:raise RuntimeError('journal count differs')
root=base/'ace_llm_intervention_import_diagnostic_20261107_01'
root.mkdir();(root/'logs').mkdir()
probe=root/'write-read-probe';token=str(uuid.uuid4()).encode()
with probe.open('xb') as f:f.write(token);f.flush();os.fsync(f.fileno())
if probe.read_bytes()!=token:raise RuntimeError('probe differs')
probe.unlink()
st=base.stat()
if st.st_uid!=os.getuid():raise RuntimeError('scratch ownership differs')
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queue':queue,'accounting':rows,'journals':journals,'allocated_cpu_seconds':cpu,'allocated_gpu_seconds':gpu,'scratch':{'owner':st.st_uid,'mode':oct(st.st_mode),'df':subprocess.run(['df','-h',str(base)],capture_output=True,text=True,check=True).stdout,'write_read_cleanup':True},'new_exclusive_output':str(root),'local_controller_pid':95527},sort_keys=True,indent=2))

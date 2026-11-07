import pathlib,json,subprocess,datetime,hashlib,os
intent={'utc': '2026-11-07T16:25:16.412524+00:00', 'source_commit': '49439d67c80aa6f30f9139eed65b8c3a58464b38', 'packet_sha256': '94b490df5adbcbecd311fca2ff20e810351d9d7a1ce42228669b6d48dd974c57', 'remote_root': '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_import_diagnostic_20261107_01', 'pins': {'job.sbatch': 'd40242b2cc9144fb0423f9df4b9d8480cc3348ae022782a866de038af11ba266', 'packet.json': '94b490df5adbcbecd311fca2ff20e810351d9d7a1ce42228669b6d48dd974c57', 'probe.py': 'a885504774e34df3245d759a57075456d46970807393b8e022d325b2a54d9fec'}, 'resources': {'account': 'ucb736_asc1', 'cpus': 1, 'gpus': 0, 'memory_gib': 3, 'no_requeue': True, 'partition': 'acpu', 'qos': 'cpu-normal', 'wall_seconds': 480}, 'admission': {'candidate_cpu_reservation': 480, 'ceiling_cpu_seconds': 57600, 'ceiling_gpu_seconds': 7200, 'closed_actual_cpu_seconds': 4450, 'closed_actual_gpu_seconds': 836, 'cumulative_cpu_preparation_allowance': 3600, 'total_cpu_seconds': 8530, 'total_gpu_seconds': 836}, 'controller_pid_local': 95527, 'scope': 'two predeclared fresh-process import orders; no models, no explicit CUDA API, no scientific responses', 'blind_retry_forbidden': True, 'review_sha256': 'ba2141c7b614c7a41023de7c4a197928e1c7c49fa72cbdaf7d9a6926c909f1e6'}
root=pathlib.Path(intent['remote_root']);base=root.parent
if (root/'submission.intent.json').exists() or (root/'submission.result.json').exists():raise RuntimeError('prior submission exists')
queue=subprocess.run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R'],capture_output=True,text=True,check=True).stdout
if 'ace_llm' in queue:raise RuntimeError('related active job')
a=subprocess.run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'],capture_output=True,text=True,check=True).stdout
rows=[l.split('|') for l in a.splitlines() if 'ace_llm' in l]
if len(rows)!=13 or any(v[2].split()[0] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for v in rows):raise RuntimeError('prior job state changed')
if sum(int(v[4])*int(v[5]) for v in rows)!=4450 or sum(int(v[5]) for v in rows if 'gres/gpu:' in v[7])!=836:raise RuntimeError('accounting changed')
journals=[str(p) for d in base.glob('ace_llm_intervention*') for p in d.glob('submission.*.json')]
if len(journals)!=28:raise RuntimeError('prior journals changed')
for name,h in intent['pins'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise RuntimeError('input drift '+name)
intent.update(remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=queue,prior_accounting=rows,prior_journals=journals)
with (root/'submission.intent.json').open('x') as f:json.dump(intent,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'intent_sha256':hashlib.sha256((root/'submission.intent.json').read_bytes()).hexdigest(),'job_id':None}
try:
 p=subprocess.run(['sbatch','--parsable',str(root/'job.sbatch')],capture_output=True,text=True,timeout=30)
 job=p.stdout.strip().split(';')[0]
 result.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,job_id=job if p.returncode==0 and job.isdigit() else None)
except BaseException as exc:result.update(ambiguous=True,exception_type=type(exc).__name__,message=str(exc))
with (root/'submission.result.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
if result['job_id'] is None:raise RuntimeError('failed or ambiguous, never blind retry')

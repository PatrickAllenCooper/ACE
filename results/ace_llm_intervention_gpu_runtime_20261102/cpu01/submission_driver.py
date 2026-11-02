import pathlib,json,hashlib,subprocess,datetime,os,uuid
manifest={'freeze_sha256': '62d0ef536f7036006c06a43e941b00bfb0c7796cbf3f7d0244c94fd61cd8e9a6', 'files': {'cpu01.sbatch': 'f300169692af9034e4956442d2dc18e5db947acf18c42f299f4cd9b15c2acf1f', 'freeze.json': '62d0ef536f7036006c06a43e941b00bfb0c7796cbf3f7d0244c94fd61cd8e9a6', 'llm_intervention_contract.py': 'ad89a8b72c8251c5a658bf45c27874353b161df5863b61929af40f5077254636', 'llm_intervention_learning.py': '065f9f68c55518190a82c6a69dc7be45a83e4ab03664e6f5df078ce6112f7aca', 'prepare_llm_intervention_gpu_cpu.py': '7a97a657e4de95dca9867cfbd13cca049f568c3228f780f4fba3282a25723e13', 'qualify_llm_intervention_gpu.py': '2f0dceb9b12cfdc82548a6cbd523c3034b69b3d07ea1b53acbea51029206d3e9', 'supervise_llm_intervention_gpu.py': '17d1c344f3ae533bd467cffa20df4a166eef13edee55b8100334ceb4ddaf62ef', 'test_llm_intervention_gpu_packet.py': '9ffc6745b246292d3364957ad3c5a1a72f326b7da08baceac3e7fc0eeb06a2a4', 'source-pins.sha256': 'b9b2462d6aac29fb920d683a0434c3e6f73e5fad7c4902e8b636c776a7039773'}, 'remote_root': '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_cpu_20261102_01'}
intent={'created_utc': '2026-11-02T20:46:07.200882+00:00', 'packet_commit': '0d63b00879cb56196c31aeb3119b5964cd6cca1c', 'freeze_sha256': '62d0ef536f7036006c06a43e941b00bfb0c7796cbf3f7d0244c94fd61cd8e9a6', 'remote_root': '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_cpu_20261102_01', 'resources': {'account': 'ucb736_asc1', 'cpus': 1, 'memory_mib': 4096, 'wall_seconds': 600, 'gpus': 0}, 'conservative_attempt_cpu_reservation_seconds': 1200, 'total_cycle_cpu_reservation_seconds': 6000, 'blind_retry_forbidden': True, 'review': 'independent actual packet review zero required issues'}
root=pathlib.Path(manifest['remote_root']);inputs=root/'input'
if {x.name for x in inputs.iterdir()}!=set(manifest['files']):raise RuntimeError('input membership mismatch')
for name,pin in manifest['files'].items():
 if hashlib.sha256((inputs/name).read_bytes()).hexdigest()!=pin:raise RuntimeError('input hash mismatch '+name)
queue=subprocess.run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%C|%b'],capture_output=True,text=True,check=True).stdout
acct=subprocess.run(['sacct','-X','-u','paco0228','-S','2026-11-02','--name','ace_llm_gcpu01','--format','JobIDRaw,JobName,State,Submit','-n','-P'],capture_output=True,text=True,check=True).stdout
if 'ace_llm_gcpu01' in queue+acct:raise RuntimeError('matching prior/current job requires reconciliation')
probe=root/'submission-readiness-probe';value=str(uuid.uuid4()).encode()
with probe.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
if probe.read_bytes()!=value:raise RuntimeError('write/read failed')
probe.unlink()
stat=root.stat()
if stat.st_uid!=os.getuid():raise RuntimeError('unexpected output owner')
intent.update({'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authenticated_inputs':len(manifest['files']),'squeue':queue,'matching_sacct':acct,'write_read_cleanup':True,'root_uid':stat.st_uid,'df':subprocess.run(['df','-h',str(root)],capture_output=True,text=True,check=True).stdout})
with (root/'submission.intent.json').open('x') as f:json.dump(intent,f,indent=2);f.flush();os.fsync(f.fileno())
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'intent':intent,'input_manifest':manifest}
try:
 result=subprocess.run(['sbatch','--parsable',str(inputs/'cpu01.sbatch')],capture_output=True,text=True,timeout=30)
 job=result.stdout.strip().split(';')[0]
 receipt.update({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'job_id':job if result.returncode==0 and job.isdigit() else None})
except BaseException as exc:
 receipt.update({'job_id':None,'ambiguous':True,'exception_type':type(exc).__name__,'message':str(exc)})
with (root/'submission.result.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt))
if receipt['job_id'] is None:raise RuntimeError('submission failed or ambiguous, never blind retry')

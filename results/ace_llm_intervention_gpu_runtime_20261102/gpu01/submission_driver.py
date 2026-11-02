import pathlib,json,hashlib,subprocess,datetime,os,uuid
manifest={'freeze_sha256': '717f77e9f663f356e68de6de36fd873380889a8398d4107912fd6c87f602e0d4', 'files': {'allocation_delta_receipt.json': '82848706488087da2ae6523c81ada96a4c34ca4011fa6446801ae6e9f4b62dcb', 'cpu_acceptance.json': 'e679f8ce2baa83bd716dae73c2cd3693c9cf4a9602c475ec441fbaa65d849b63', 'cpu_terminal.json': '06d4c4096137a8111d81af61db411aaf8397fe23f516a0a7c1801ca9fdfba127', 'fixture.json': '0c0d24ec20af68597aa79f439538c55a76395702246475d1b3466f496f4c1e52', 'freeze.json': '717f77e9f663f356e68de6de36fd873380889a8398d4107912fd6c87f602e0d4', 'gpu01.sbatch': 'd7e00b5fa054ed55d0e4d43afad3b1399534df014f90ce14333c00337fa3e6a7', 'llm_intervention_contract.py': 'ad89a8b72c8251c5a658bf45c27874353b161df5863b61929af40f5077254636', 'llm_intervention_learning.py': '065f9f68c55518190a82c6a69dc7be45a83e4ab03664e6f5df078ce6112f7aca', 'qualify_llm_intervention_gpu.py': '2f0dceb9b12cfdc82548a6cbd523c3034b69b3d07ea1b53acbea51029206d3e9', 'supervise_llm_intervention_gpu.py': '5ac036598af3d31e76357a238f449ecee589edfa93cc2d3eec57cca42d2bb332', 'source-pins.sha256': 'bbd35ec012471bd033fe89772859074a6cff7a5358d5b91442e83ae9205c721c'}, 'remote_root': '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_01'}
intent={'created_utc': '2026-11-02T20:54:01.505558+00:00', 'packet_commit': '254dc7fdf043fc1f0699ef06e3d68f415201d744', 'freeze_sha256': '717f77e9f663f356e68de6de36fd873380889a8398d4107912fd6c87f602e0d4', 'remote_root': '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_01', 'resources': {'account': 'ucb736_asc1', 'gpu_type': 'a100_3g.20gb', 'gpus': 1, 'max_allocated_cpus': 8, 'memory_mib': 12288, 'partition': 'aa100', 'qos': 'gpu-normal', 'requested_cpus': 4, 'wall_seconds': 900}, 'conservative_attempt_cpu_reservation_seconds': 7200, 'total_cycle_cpu_reservation_seconds': 13200, 'attempt_gpu_reservation_seconds': 900, 'total_cycle_gpu_reservation_seconds': 900, 'blind_retry_forbidden': True, 'review': 'independent actual GPU packet review zero required issues', 'cpu_qualification': '35202401 complete and accepted, full43object custody'}
root=pathlib.Path(manifest['remote_root']);inputs=root/'input'
if {x.name for x in inputs.iterdir()}!=set(manifest['files']):raise RuntimeError('input membership mismatch')
for name,pin in manifest['files'].items():
 if hashlib.sha256((inputs/name).read_bytes()).hexdigest()!=pin:raise RuntimeError('input hash mismatch '+name)
queue=subprocess.run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%C|%b'],capture_output=True,text=True,check=True).stdout
acct=subprocess.run(['sacct','-X','-u','paco0228','-S','2026-11-02','--name','ace_llm_gpu01','--format','JobIDRaw,JobName,State,Submit','-n','-P'],capture_output=True,text=True,check=True).stdout
if 'ace_llm_gpu01' in queue+acct:raise RuntimeError('matching prior/current job requires reconciliation')
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
 result=subprocess.run(['sbatch','--parsable',str(inputs/'gpu01.sbatch')],capture_output=True,text=True,timeout=30)
 job=result.stdout.strip().split(';')[0]
 receipt.update({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'job_id':job if result.returncode==0 and job.isdigit() else None})
except BaseException as exc:
 receipt.update({'job_id':None,'ambiguous':True,'exception_type':type(exc).__name__,'message':str(exc)})
with (root/'submission.result.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt))
if receipt['job_id'] is None:raise RuntimeError('submission failed or ambiguous, never blind retry')

import pathlib,json,hashlib,subprocess,datetime
manifest={'freeze_sha256': '9c114bc8d557f677dbd71bd36c6123daae9b3fff3d3bce96322c73b94ced6364', 'files': {'llm_intervention_contract.py': '7b1459c004fd7f1bd869cc303da127bc33015c0a76b82e68f966ed5bb8b2be5f', 'llm_intervention_learning.py': '065f9f68c55518190a82c6a69dc7be45a83e4ab03664e6f5df078ce6112f7aca', 'test_llm_intervention_learning.py': '76a974f4a2ebe29e14cb0a1089634f9e0d1335e7026426ca9919f7188aa87d96', 'test_llm_intervention_contract.py': 'dd6eeeb6b20f0f62c41929de6cde8db5a4dbcc9fd268d2a2c67725c242795675', 'qualify_llm_intervention_cpu.py': 'be2d2c5c36b668bce51c5b93baf249842ba0ad9973aa7224e541d536c02181dd', 'cpu01.sbatch': '824f49dd59f5d77f5a3d7dcbbebf43acfcb639378b3a715ad92227d86c9f877d', 'freeze.json': '9c114bc8d557f677dbd71bd36c6123daae9b3fff3d3bce96322c73b94ced6364', 'source-pins.sha256': '8c7513d390a83b223801ec55f61d08f523d73b726cf28ce556d39b54d669a161'}, 'remote_root': '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_20261102_01'}
intent={'created_utc': '2026-11-02T20:03:24.910280+00:00', 'packet_commit': '96d6f741f9f82ba6859dfb73fc476d253fcf60ea', 'freeze_sha256': '9c114bc8d557f677dbd71bd36c6123daae9b3fff3d3bce96322c73b94ced6364', 'remote_root': '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_20261102_01', 'resources': {'account': 'ucb736_asc1', 'cpus': 1, 'memory_mib': 4096, 'wall_seconds': 600, 'gpus': 0}, 'blind_retry_forbidden': True}
root=pathlib.Path(manifest['remote_root'])
inputs=root/'input'
if {x.name for x in inputs.iterdir()}!=set(manifest['files']): raise RuntimeError('input membership mismatch')
for name,pin in manifest['files'].items():
 if hashlib.sha256((inputs/name).read_bytes()).hexdigest()!=pin: raise RuntimeError('input hash mismatch '+name)
queue=subprocess.run(['squeue','-u','paco0228','-h','-o','%i|%j|%T'],capture_output=True,text=True,check=True).stdout
accounting=subprocess.run(['sacct','-X','-u','paco0228','-S','2026-11-02','--name','ace_llm_cpu01','--format','JobIDRaw,JobName,State,Submit','-n','-P'],capture_output=True,text=True,check=True).stdout
if any('ace_llm_cpu01' in line for line in (queue+accounting).splitlines()): raise RuntimeError('matching prior/current job requires reconciliation')
intent.update({'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authenticated_inputs':len(manifest['files']),'squeue':queue,'matching_sacct':accounting})
with (root/'submission.intent.json').open('x') as f: json.dump(intent,f,indent=2);f.flush()
result=subprocess.run(['sbatch','--parsable',str(inputs/'cpu01.sbatch')],capture_output=True,text=True,timeout=30)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'input_manifest':manifest,'intent':intent}
job=result.stdout.strip().split(';')[0]
receipt['job_id']=job if result.returncode==0 and job.isdigit() else None
with (root/'submission.result.json').open('x') as f: json.dump(receipt,f,indent=2);f.flush()
print(json.dumps(receipt))
if receipt['job_id'] is None: raise RuntimeError('submission failed or ambiguous; do not retry')

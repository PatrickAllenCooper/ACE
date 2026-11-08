import pathlib,json,hashlib,subprocess,datetime
r=pathlib.Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01')
assert hashlib.sha256((r/'submission.intent.json').read_bytes()).hexdigest()=='61b1d2a2043764a8d82e04dcfafc95baf6dd66598597ed2e9bfa0e94b42ce9b8'
s=subprocess.run(['sacct','-j','35407428','--format','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS','-n','-P'],capture_output=True,text=True,timeout=15)
paths=['collection/session/worker/work-start.json','collection/session/worker/runtime-before_collection.json','collection/terminal.json','collection/failure.json','collection/session/worker/task/terminal.json','collection/session/worker/task/failure.json']
e={n:dict(present=(r/n).is_file(),sha256=hashlib.sha256((r/n).read_bytes()).hexdigest() if (r/n).is_file() else None) for n in paths}
c=r/'collection/session/worker/task/collection';worlds=[c/f'world-{i:03d}' for i in range(32)]
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scheduler=dict(returncode=s.returncode,stdout=s.stdout,stderr=s.stderr),markers=e,world_directories_present=sum(p.is_dir() for p in worlds),record_files_present=sum(sum(1 for _ in p.glob('*/record.json')) for p in worlds),outcome_decoding=False,acceptance=False)))

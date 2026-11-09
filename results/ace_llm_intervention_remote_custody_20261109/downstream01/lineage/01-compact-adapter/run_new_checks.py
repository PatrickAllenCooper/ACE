from pathlib import Path
import datetime,hashlib,json,os,resource,shutil,subprocess,time
r=Path('/tmp/ace-downstream-compact-adapter-6r23szeq');p=r/'attempt03-wiring';p.mkdir()
shutil.copytree(r/'implementation',p/'source')
before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
command=['python3','-B','-m','unittest','-v','test_compact_wiring']
proc=subprocess.run(command,cwd=p/'source',capture_output=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
after=resource.getrusage(resource.RUSAGE_CHILDREN)
(p/'stdout.txt').write_bytes(proc.stdout);(p/'stderr.txt').write_bytes(proc.stderr)
receipt=dict(command=command,returncode=proc.returncode,wall_seconds=time.monotonic()-start,
child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,child_maxrss_bytes=after.ru_maxrss,
source_pins={str(f.relative_to(p/'source')):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((p/'source').rglob('*.py'))},
remote_actions=0,project_or_model_execution=False,old_tests_repeated=False)
(p/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(proc.stderr.decode());print(json.dumps(receipt,sort_keys=True))

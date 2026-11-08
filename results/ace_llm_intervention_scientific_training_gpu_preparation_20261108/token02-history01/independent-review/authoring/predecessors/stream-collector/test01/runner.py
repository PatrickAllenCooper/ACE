from pathlib import Path
import os,sys,json,hashlib,resource,subprocess,time,shutil,datetime
root=Path('/tmp/ace-scientific-training-gpu-stream-current.txt').read_text().strip();root=Path(root)
attempt=root/'test01';attempt.mkdir();(attempt/'fixtures').mkdir();(attempt/'source').mkdir()
files=('pull_terminal.py','test_gpu_stream_integration.py','stream_custody.py','stream_custody_review.md')
pins={}
for name in files:
    data=(root/name).read_bytes();pins[name]=hashlib.sha256(data).hexdigest();shutil.copyfile(root/name,attempt/'source'/name)
command=[sys.executable,'-B',str(root/'test_gpu_stream_integration.py')]
env=dict(os.environ,ACE_TRANSPORT_TEST_EVIDENCE=str(attempt/'fixtures'),PYTHONDONTWRITEBYTECODE='1')
start=time.monotonic();cpu=resource.getrusage(resource.RUSAGE_CHILDREN)
with (attempt/'stdout.txt').open('xb') as out,(attempt/'stderr.txt').open('xb') as err:
    result=subprocess.run(command,cwd='/tmp',env=env,stdout=out,stderr=err,timeout=120)
end=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt=dict(schema='ace-fabricated-gpu-stream-integration-tests-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,cwd='/tmp',source_pins=pins,returncode=result.returncode,wall_seconds=time.monotonic()-start,child_cpu_seconds=end.ru_utime+end.ru_stime-cpu.ru_utime-cpu.ru_stime,tests_scope='Six NEW fabricated integration methods only; mock capture/Git; actual helper emit/materialize solely local artificial bytes.',old_tests_repeated=0,remote_actions=0,scientific_responses=0,models_loaded=0,acceptance=False,stdout_sha256=hashlib.sha256((attempt/'stdout.txt').read_bytes()).hexdigest(),stderr_sha256=hashlib.sha256((attempt/'stderr.txt').read_bytes()).hexdigest())
(attempt/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
shutil.copyfile(__file__,attempt/'runner.py')
print(json.dumps(receipt,indent=2));print((attempt/'stderr.txt').read_text())

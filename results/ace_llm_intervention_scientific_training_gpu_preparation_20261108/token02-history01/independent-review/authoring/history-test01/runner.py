from pathlib import Path
import os,sys,json,hashlib,resource,subprocess,time,shutil
r=Path(Path('/tmp/ace-scientific-training-gpu-token02-current.txt').read_text().strip());a=r/'history-test01';a.mkdir();(a/'fixtures').mkdir();(a/'source').mkdir()
pins={}
for n in ['build_packet.py','launch_once.py','monitor_once.py','test_token02_history.py']:
 shutil.copyfile(r/n,a/'source'/n);pins[n]=hashlib.sha256((r/n).read_bytes()).hexdigest()
cmd=[sys.executable,'-B',str(r/'test_token02_history.py')];cpu=resource.getrusage(resource.RUSAGE_CHILDREN);t=time.monotonic()
with (a/'stdout.txt').open('xb') as out,(a/'stderr.txt').open('xb') as err:
 p=subprocess.run(cmd,cwd='/tmp',env=dict(os.environ,ACE_HISTORY_TEST_EVIDENCE=str(a/'fixtures'),PYTHONDONTWRITEBYTECODE='1'),stdout=out,stderr=err,timeout=60)
c=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt=dict(command=cmd,source_pins=pins,returncode=p.returncode,wall_seconds=time.monotonic()-t,child_cpu_seconds=c.ru_utime+c.ru_stime-cpu.ru_utime-cpu.ru_stime,scope='four NEW artificial metadata history-join methods',old_tests_repeated=0,remote_actions=0,scientific_execution=False,acceptance=False)
(a/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n');shutil.copyfile(__file__,a/'runner.py')
print(json.dumps(receipt,indent=2));print((a/'stderr.txt').read_text())

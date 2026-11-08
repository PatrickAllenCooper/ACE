from pathlib import Path
import hashlib,json,os,resource,shutil,subprocess,time
root=Path('/tmp/ace-scientific-finalizer-token02-8wf1foat');out=root/'delta02';out.mkdir();(out/'fixtures').mkdir();(out/'sources').mkdir()
files=['build_packet.py','launch_once.py','monitor_once.py','collector/pull_terminal.py','collector/stream_custody.py','test_history_transport_delta.py']
pins={}
for n in files:
 p=out/'sources'/n;p.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(root/n,p);pins[n]=hashlib.sha256(p.read_bytes()).hexdigest()
command=['/Library/Developer/CommandLineTools/usr/bin/python3','-B',str(root/'test_history_transport_delta.py'),'NewFinalizerHistory.test_stream_eight_roots_and_two_independent_private_copies','NewFinalizerHistory.test_whole_nested_failure_archive_required']
start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
run=subprocess.run(command,cwd=root,env={**os.environ,'ACE_FINALIZER_TEST_EVIDENCE':str(out/'fixtures'),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True)
after=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.monotonic()-start
(out/'stdout.txt').write_bytes(run.stdout);(out/'stderr.txt').write_bytes(run.stderr)
receipt=dict(command=command,returncode=run.returncode,source_pins=pins,wall_seconds=wall,child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,maxrss=after.ru_maxrss,scope='only two affected NEW methods after resolved Darwin temporary path and sorted fixture directory correction; seven passing methods not repeated',old_tests_repeated=0,remote_actions=0,science=False,actual_packet_build=False)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt));print(run.stderr.decode())

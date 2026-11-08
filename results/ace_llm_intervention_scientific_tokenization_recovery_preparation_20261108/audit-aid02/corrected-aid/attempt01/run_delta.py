from pathlib import Path
import hashlib,json,os,resource,shutil,subprocess,sys,time
root=Path(__file__).resolve().parent
out=root/'attempt01';out.mkdir(mode=0o700,exist_ok=False)
for n in ('audit_saved.py','test_review_delta.py','run_delta.py','reference-pins.json','HANDOFF.md','initial-review.md'):
 shutil.copyfile(root/n,out/n)
shutil.copytree(root/'source',out/'source')
pins={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()}
command=[sys.executable,'-I','-S','-B',str(out/'test_review_delta.py')]
(out/'intent.json').write_text(json.dumps(dict(command=command,cwd='/tmp',source_pins=pins,scope='only NEW R1/R2 artificial metadata fixtures; no actual outputs',automatic_retry=False),indent=2,sort_keys=True)+'\n')
start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
 p=subprocess.run(command,cwd='/tmp',stdout=stdout,stderr=stderr,timeout=60)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
r=dict(returncode=p.returncode,wall_seconds=time.monotonic()-start,child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
 new_methods=2,old_tests_repeated=0,actual_outputs_read=False,models_tokenizers_tensors_loaded=False,remote_actions=0,
 source_pins=pins,stdout_sha256=hashlib.sha256((out/'stdout.log').read_bytes()).hexdigest(),stderr_sha256=hashlib.sha256((out/'stderr.log').read_bytes()).hexdigest(),
 cost_scope='new focused test child only; authoring overhead not measured',all_failures_retained=True)
(out/'receipt.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='source_pins'},indent=2))
print((out/'stderr.log').read_text())
raise SystemExit(p.returncode)

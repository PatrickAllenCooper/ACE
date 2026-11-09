from pathlib import Path
import json,hashlib,resource,shutil,subprocess,time,sys
r=Path(__file__).resolve().parent;out=r/sys.argv[1];out.mkdir()
shutil.copytree(r/'implementation',out/'source')
# Test AST baseline stays source-adjacent; no old method is invoked.
shutil.copytree(r/'predecessor',out/'predecessor')
cmd=['python3','-B','-m','unittest','-v',*sys.argv[2:]]
a=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();p=subprocess.run(cmd,cwd=out/'source',capture_output=True);z=resource.getrusage(resource.RUSAGE_CHILDREN)
for name,data in [('stdout.txt',p.stdout),('stderr.txt',p.stderr)]: (out/name).write_bytes(data)
v=dict(command=cmd,returncode=p.returncode,elapsed_seconds=time.monotonic()-start,child_cpu_seconds=z.ru_utime+z.ru_stime-a.ru_utime-a.ru_stime,maxrss_bytes=z.ru_maxrss,remote_actions=0,project_model_science_execution=False,old_methods_run=False,source_pins={str(f.relative_to(out/'source')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (out/'source').rglob('*.py')})
(out/'receipt.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');print(p.stderr.decode());print(json.dumps({k:v for k,v in v.items() if k!='source_pins'}))

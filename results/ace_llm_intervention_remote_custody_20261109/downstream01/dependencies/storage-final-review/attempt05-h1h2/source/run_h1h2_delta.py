from pathlib import Path
import hashlib,json,os,resource,shutil,subprocess,sys,time
O=Path(__file__).resolve().parent;E=O/'attempt05-h1h2';E.mkdir();(E/'source').mkdir();(E/'fixtures').mkdir()
for n in ['remote_custody.py','test_cleanup_bootstrap_delta.py','run_h1h2_delta.py']:shutil.copyfile(O/n,E/'source'/n)
cmd=[sys.executable,'-B',str(E/'source/test_cleanup_bootstrap_delta.py')];env=dict(os.environ,ACE_H1H2_FIXTURES=str(E/'fixtures'),PYTHONDONTWRITEBYTECODE='1')
b=resource.getrusage(resource.RUSAGE_CHILDREN);t=time.monotonic()
with (E/'stdout.txt').open('xb') as out,(E/'stderr.txt').open('xb') as err:r=subprocess.run(cmd,cwd='/tmp',env=env,stdout=out,stderr=err)
a=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.monotonic()-t
files={str(p.relative_to(E)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(E.rglob('*')) if p.is_file()}
receipt={'status':'passed' if r.returncode==0 else 'failed','returncode':r.returncode,'command':cmd,'cwd':'/tmp','wall_seconds':wall,'child_cpu_seconds':a.ru_utime+a.ru_stime-b.ru_utime-b.ru_stime,'peak_rss_bytes_macos':a.ru_maxrss,'files':files,'new_methods':3,'remote_actions':False,'actual_scientific_data_read':False,'local_subprocesses':'only isolated Python metadata bootstrap fixture'}
(E/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2));sys.exit(r.returncode)

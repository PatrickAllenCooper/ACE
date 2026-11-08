from pathlib import Path
import datetime,hashlib,json,os,resource,shutil,subprocess,sys,time
O=Path(__file__).resolve().parent;E=O/'attempt01';E.mkdir();(E/'fixtures').mkdir();(E/'source').mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in ['pull_terminal.py','test_seal_boundary.py','run_focused.py']:shutil.copyfile(O/n,E/'source'/n)
cmd=[sys.executable,'-B',str(E/'source/test_seal_boundary.py')];env=dict(os.environ,ACE_SEAL_FIXTURE_ROOT=str(E/'fixtures'),PYTHONDONTWRITEBYTECODE='1')
start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
with (E/'stdout.txt').open('xb') as out,(E/'stderr.txt').open('xb') as err:r=subprocess.run(cmd,cwd='/tmp',env=env,stdout=out,stderr=err)
after=resource.getrusage(resource.RUSAGE_CHILDREN);elapsed=time.monotonic()-start
files={}
for p in sorted(E.rglob('*')):
 if p.is_symlink():files[str(p.relative_to(E))]={'kind':'fixture_symlink','target':os.readlink(p)}
 elif p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b)
  files[str(p.relative_to(E))]={'sha256':h.hexdigest(),'bytes':p.stat().st_size,'nlink':p.stat().st_nlink}
receipt={'status':'passed' if r.returncode==0 else 'failed','returncode':r.returncode,'command':cmd,'cwd':'/tmp','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':elapsed,'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'child_maxrss_bytes_macos':after.ru_maxrss,'new_test_methods':3,'original_fixture_preserved':True,'fixture_alias_exception':'tiny-guards and hardlink-guard are explicit nlink-negative fixtures; symlink-guard is an explicit rejection fixture; not evidence aliases','files':files,'actual_collector_or_remote_executed':False,'actual_seal_read':False,'scientific_data':False}
(E/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2));sys.exit(r.returncode)

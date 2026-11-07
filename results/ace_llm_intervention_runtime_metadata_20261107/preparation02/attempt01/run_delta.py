"""One exclusive v2 delta attempt; preserve all attempted bytes and failures."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time

REPO = Path('/Users/pat/code/ACE')
LOCAL = REPO/'results/ace_llm_intervention_runtime_metadata_20261107/preparation02'
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-runtime-metadata-preparation-20261107-02')

def raw(value): return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def save(path,value):
    with path.open('xb') as f:f.write(raw(value));f.flush();os.fsync(f.fileno())
def pin(path):
    b=path.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def members(root):
    rows=[]
    for p in sorted(root.rglob('*')):
        name=str(p.relative_to(root))
        if p.is_symlink():rows.append({'path':name,'kind':'symlink','target':os.readlink(p)})
        elif p.is_file():rows.append({'path':name,'kind':'file',**pin(p)})
    return rows

attempt=sys.argv[1]
if not re.fullmatch(r'attempt\d\d',attempt):raise ValueError('exclusive attempt name required')
out=EXTERNAL/attempt;target=LOCAL/attempt
if target.exists() or out.exists():raise FileExistsError('attempt already exists')
out.mkdir();source=out/'source';source.mkdir()
paths=['scripts/research/qualify_llm_intervention_runtime_metadata.py',
       'scripts/research/test_llm_intervention_runtime_metadata.py',
       'docs/development/guidance/ace_llm_intervention_runtime_metadata_2026-11-07.md']
sources={}
for name in paths:
    path=REPO/name;copied=source/path.name
    with copied.open('xb') as f:f.write(path.read_bytes())
    sources[name]=pin(copied)
save(out/'source_manifest.json',sources)
with (out/'run_delta.py').open('xb') as f:f.write(Path(__file__).read_bytes())
bootstrap='''import hashlib,importlib.abc,json,sys,types,unittest
from pathlib import Path
class NoRealRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy','scipy','torch','transformers'):
            raise AssertionError('actual runtime import prohibited: '+fullname)
sys.meta_path.insert(0,NoRealRuntime())
root=Path(sys.argv[1])
for name,pin in json.loads(sys.argv[2]).items():
    path=root/(name+'.py');b=path.read_bytes()
    assert hashlib.sha256(b).hexdigest()==pin
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(b,str(path),'exec'),module.__dict__)
test=sys.modules['test_llm_intervention_runtime_metadata']
suite=unittest.defaultTestLoader.loadTestsFromTestCase(test.MetadataDeltaTests)
assert suite.countTestCases()==6
assert all(x._testMethodName.startswith('test_delta_') for x in suite)
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
'''
pins={Path(name).stem:r['sha256'] for name,r in sources.items() if name.endswith('.py')}
command=[sys.executable,'-I','-B','-c',bootstrap,str(source),json.dumps(pins)]
env={'ACE_RUNTIME_METADATA_DELTA_OUTPUT':str(out/'fixtures'),'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
     'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
metadata={'command':command,'cwd':'/tmp','environment_overrides':env,'python':sys.version,
          'scope':'Six NEW MetadataDeltaTests methods only; stdlib and fabricated source/modules. Real runtime imports blocked.',
          'old_methods_executed':0,'runtime_adopted':False,'remote_operations':0,'gpu_allocations':0,
          'environment_responses':0,'models':0,'training_updates':0,'timeout_seconds':30,
          'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
save(out/'command.json',metadata)
before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();cpu=time.process_time()
with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
    try:
        result=subprocess.run(command,cwd='/tmp',env=dict(os.environ,**env),stdout=stdout,stderr=stderr,timeout=30)
        code,error=result.returncode,None
    except BaseException as exc:code,error=None,type(exc).__name__+': '+str(exc)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt={**metadata,'returncode':code,'error':error,'elapsed_seconds':time.monotonic()-start,
         'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
         'parent_cpu_seconds':time.process_time()-cpu,'peak_child_rss_bytes_macos':after.ru_maxrss,
         'cost_scope':'This test attempt only; excludes preparation, review and copying, not full-cycle CPU.',
         'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
save(out/'receipt.json',receipt)
save(out/'closeout.json',{'status':'delta_passed_review_pending' if code==0 else 'failed_attempt_retained',
                        'source_pins':sources,'receipt_sha256':pin(out/'receipt.json')['sha256'],
                        'runtime_qualified':False,'service_qualified':False,'launched':False})
save(out/'manifest.json',{'members':members(out)})
shutil.copytree(out,target,symlinks=True)
assert members(out)==members(target)
print(json.dumps({'attempt':attempt,'returncode':code,'elapsed_seconds':receipt['elapsed_seconds'],
                 'child_cpu_seconds':receipt['child_cpu_seconds'],'peak_child_rss_bytes_macos':after.ru_maxrss,
                 'receipt_sha256':pin(out/'receipt.json')['sha256'],'manifest_sha256':pin(out/'manifest.json')['sha256'],
                 'files_and_links':len(members(out)),'source_pins':sources},indent=2))
print((out/'stderr.log').read_text())
raise SystemExit(0 if code==0 else 1)

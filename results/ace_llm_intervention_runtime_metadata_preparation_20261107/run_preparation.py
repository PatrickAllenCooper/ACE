"""One exclusive NEW stdlib/fabricated runtime-metadata integration attempt."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_runtime_metadata_preparation_20261107'
EXTERNAL=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-runtime-metadata-preparation-20261107-01')

def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def pin(path):
    b=path.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

EXTERNAL.mkdir(exist_ok=False)
source=EXTERNAL/'source';source.mkdir()
inputs=[REPO/'scripts/research'/name for name in ('qualify_llm_intervention_runtime_metadata.py','test_llm_intervention_runtime_metadata.py','llm_intervention_runtime_archive.py')]
inputs += [REPO/'results/ace_llm_intervention_staged_runtime_20261107/launch01/probe.py',REPO/'docs/development/guidance/ace_llm_intervention_runtime_metadata_2026-11-07.md']
sources={}
for path in inputs:
    shutil.copyfile(path,source/path.name);sources[str(path.relative_to(REPO))]=pin(source/path.name)
save(EXTERNAL/'source_manifest.json',sources)
context=EXTERNAL/'read-context';context.mkdir()
b=REPO/'results/ace_llm_intervention_staged_runtime_20261107'
context_paths=[b/'terminal01/remote/qualification/original_sequence/failure.json',b/'terminal01/remote/qualification/stage.json',
               b/'terminal01/remote/qualification/worker.cleanup.json',b/'numpy-forensics01/forensics.json',b/'conda-metadata01/comparison.json',
               b/'cache-headers01/headers.json',b/'runtime-code-versions01/source_versions.json',
               REPO/'docs/development/guidance/reviews/ace_llm_intervention_staged_runtime_results_review_2026-11-07.md',
               REPO/'scripts/research/qualify_llm_intervention_staged_import.py']
context_pins={}
for path in context_paths:
    target=context/path.name;shutil.copyfile(path,target);context_pins[str(path.relative_to(REPO))]=pin(target)
save(EXTERNAL/'read-context-pins.json',context_pins)
shutil.copyfile(__file__,EXTERNAL/'run_preparation.py')
env={'ACE_RUNTIME_METADATA_TEST_OUTPUT':str(EXTERNAL/'fixtures'),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'',
     'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
command=[sys.executable,'-B',str(source/'test_llm_intervention_runtime_metadata.py')]
metadata={'command':command,'cwd':'/tmp','environment_overrides':env,'python':sys.version,
          'started_utc':datetime.now(timezone.utc).isoformat(),'scope':'Eight NEW stdlib/fabricated methods; tiny hand-written tar, cached code without execution, fake numerical modules, two tiny stdlib bootstrap children.',
          'execution_invocations':1,'old_test_suites_repeated':0,'real_runtime_or_numerical_packages_executed':0,
          'bulk_archive_staged_or_rehashed':False,'worlds':0,'environment_responses':0,'models':0,'training_updates':0,'gpu_allocations':0,'remote_operations':0}
save(EXTERNAL/'command.json',metadata)
before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();cpu=time.process_time()
with (EXTERNAL/'stdout.log').open('xb') as out,(EXTERNAL/'stderr.log').open('xb') as err:
    try:
        result=subprocess.run(command,cwd='/tmp',env=dict(os.environ,**env),stdout=out,stderr=err,timeout=30,check=False)
        code,error=result.returncode,None
    except BaseException as exc:code,error=None,type(exc).__name__+': '+str(exc)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt={**metadata,'finished_utc':datetime.now(timezone.utc).isoformat(),'returncode':code,'exception':error,
         'elapsed_seconds':time.monotonic()-start,'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
         'supervisor_cpu_seconds':time.process_time()-cpu,'peak_child_rss_bytes_macos':after.ru_maxrss,
         'scope_of_cost':'Test child and waited descendants only; excludes source preparation and review; not aggregate memory or full-cycle CPU.'}
save(EXTERNAL/'receipt.json',receipt)
save(EXTERNAL/'closeout.json',{'status':'new_fabricated_tests_passed_independent_review_pending' if code==0 else 'failed_attempt_retained',
     'source_pins':sources,'test_receipt_sha256':pin(EXTERNAL/'receipt.json')['sha256'],'launched':False,
     'runtime_qualified':False,'service_qualified':False,'historical_runtime_certified':False})
members=[{'path':str(p.relative_to(EXTERNAL)),**pin(p)} for p in sorted(EXTERNAL.rglob('*')) if p.is_file()]
save(EXTERNAL/'manifest.json',{'member_count_excluding_manifest':len(members),'members':members})
for path in EXTERNAL.iterdir():
    target=LOCAL/path.name
    if path.is_dir():shutil.copytree(path,target)
    elif path.name=='run_preparation.py':assert path.read_bytes()==target.read_bytes()
    else:
        with target.open('xb') as f:f.write(path.read_bytes())
print(json.dumps(receipt,indent=2))
print((EXTERNAL/'stderr.log').read_text())
print(json.dumps({'custody_files':len(members)+1,'custody_bytes':sum(p.stat().st_size for p in EXTERNAL.rglob('*') if p.is_file())}))
raise SystemExit(0 if code==0 else 1)

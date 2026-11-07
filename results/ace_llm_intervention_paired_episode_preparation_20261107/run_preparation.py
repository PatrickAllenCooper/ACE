"""One exclusive new artificial integration attempt, preserving all outcomes."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_paired_episode_preparation_20261107'
EXTERNAL=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-paired-episode-preparation-20261107-01')


def save(p,v):
    with p.open('x') as f:
        json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def pin(p):
    b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


EXTERNAL.mkdir(exist_ok=False)
shutil.copytree(LOCAL/'pre_review_sources',EXTERNAL/'pre_review_sources')
source=EXTERNAL/'source';source.mkdir()
pending=['llm_intervention_paired_episode','test_llm_intervention_paired_episode']
sources={}
while pending:
    module=pending.pop()
    if module in sources:continue
    p=REPO/'scripts/research'/(module+'.py')
    shutil.copyfile(p,source/p.name)
    sources[module]={'original':str(p),**pin(source/p.name)}
    for n in ast.walk(ast.parse(p.read_text())):
        names=([n.module] if isinstance(n,ast.ImportFrom) and n.module else [v.name for v in n.names] if isinstance(n,ast.Import) else [])
        pending.extend(name for name in names if name.startswith('llm_intervention_'))
guide=REPO/'docs/development/guidance/ace_llm_intervention_paired_episode_2026-11-07.md'
shutil.copyfile(guide,source/guide.name)
sources[guide.name]={'original':str(guide),**pin(source/guide.name)}
save(EXTERNAL/'source_manifest.json',sources)
shutil.copyfile(__file__,EXTERNAL/'run_preparation.py')
env={'ACE_PAIRED_TEST_OUTPUT':str(EXTERNAL/'fixtures'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
    'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
command=[sys.executable,'-B',str(source/'test_llm_intervention_paired_episode.py')]
metadata={'command':command,'cwd':'/tmp','environment_overrides':env,'python':sys.version,
    'numpy':importlib.metadata.version('numpy'),'started_utc':datetime.now(timezone.utc).isoformat(),
    'scope':'Seven NEW end-to-end artificial supplied-world methods. New artificial responses and SCM updates, no pretrained model/neural training/scientific draws.',
    'old_test_suites_repeated':0,'scientific_worlds':0,'pretrained_models':0,'new_gpu_allocations':0}
save(EXTERNAL/'command.json',metadata)
before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
with (EXTERNAL/'stdout.log').open('xb') as out,(EXTERNAL/'stderr.log').open('xb') as err:
    try:
        p=subprocess.run(command,cwd='/tmp',env=dict(os.environ,**env),stdout=out,stderr=err,timeout=90,check=False)
        code,error=p.returncode,None
    except BaseException as exc:
        code,error=None,type(exc).__name__+': '+str(exc)
elapsed=time.monotonic()-start;after=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt={**metadata,'finished_utc':datetime.now(timezone.utc).isoformat(),'returncode':code,'exception':error,
    'elapsed_seconds':elapsed,'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
    'peak_child_rss_bytes_macos':after.ru_maxrss,'scope_of_cost':'Command child and its waited children, not aggregate process-tree peak or full development cost. No allowance reset.'}
save(EXTERNAL/'receipt.json',receipt)
save(EXTERNAL/'closeout.json',{'status':'new_integration_passed_review_pending' if code==0 else 'failed_attempt_retained',
    'source_pins':sources,'test_receipt_sha256':pin(EXTERNAL/'receipt.json')['sha256'],'execution_invocations':1,
    'scientific_provenance_qualified':False,'scientific_screen_executed':False})
members=[{'path':str(q.relative_to(EXTERNAL)),**pin(q)} for q in sorted(EXTERNAL.rglob('*')) if q.is_file()]
save(EXTERNAL/'manifest.json',{'member_count_excluding_manifest':len(members),'members':members})
for q in EXTERNAL.iterdir():
    target=LOCAL/q.name
    if q.is_dir():
        if q.name=='pre_review_sources':
            assert {str(f.relative_to(q)):f.read_bytes() for f in q.rglob('*') if f.is_file()} == {str(f.relative_to(target)):f.read_bytes() for f in target.rglob('*') if f.is_file()}
        else:shutil.copytree(q,target)
    elif q.name=='run_preparation.py':assert q.read_bytes()==target.read_bytes()
    else:
        with target.open('xb') as f:f.write(q.read_bytes())
print(json.dumps(receipt,indent=2))
print((EXTERNAL/'stderr.log').read_text())
print(json.dumps({'custody_files':len(members)+1,'custody_bytes':sum(q.stat().st_size for q in EXTERNAL.rglob('*') if q.is_file())}))
raise SystemExit(0 if code==0 else 1)

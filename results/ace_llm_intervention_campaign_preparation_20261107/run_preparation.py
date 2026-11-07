"""One new exclusive campaign fixture attempt; preserve all failures and costs."""
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
LOCAL=REPO/'results/ace_llm_intervention_campaign_preparation_20261107'
EXTERNAL=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-preparation-20261107-01')

def save(p,value):
    with p.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def pin(p):
    raw=p.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

EXTERNAL.mkdir(exist_ok=False)
source=EXTERNAL/'source';source.mkdir()
pending=['llm_intervention_campaign','test_llm_intervention_campaign']
sources={}
while pending:
    module=pending.pop()
    if module in sources:continue
    p=REPO/'scripts/research'/(module+'.py')
    shutil.copyfile(p,source/p.name)
    sources[module]={'original':str(p),**pin(source/p.name)}
    for node in ast.walk(ast.parse(p.read_text())):
        names=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
               [v.name for v in node.names] if isinstance(node,ast.Import) else [])
        pending.extend(name for name in names if name.startswith('llm_intervention_'))
guide=REPO/'docs/development/guidance/ace_llm_intervention_campaign_2026-11-07.md'
shutil.copyfile(guide,source/guide.name)
sources[guide.name]={'original':str(guide),**pin(source/guide.name)}
save(EXTERNAL/'source_manifest.json',sources)
context=EXTERNAL/'read-context';context.mkdir()
for name in ('ace_llm_intervention_execution_2026-11-02.md','ace_llm_intervention_screen_protocol_2026-11-07.md',
             'ace_llm_intervention_paired_episode_2026-11-07.md','ace_llm_intervention_screen_reporting_2026-11-07.md'):
    shutil.copyfile(REPO/'docs/development/guidance'/name,context/name)
shutil.copyfile(__file__,EXTERNAL/'run_preparation.py')
env={'ACE_CAMPAIGN_TEST_OUTPUT':str(EXTERNAL/'fixtures'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
     'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
command=[sys.executable,'-B',str(source/'test_llm_intervention_campaign.py')]
meta={'command':command,'cwd':'/tmp','environment_overrides':env,'python':sys.version,
      'numpy':importlib.metadata.version('numpy'),'started_utc':datetime.now(timezone.utc).isoformat(),
      'scope':'Eight NEW bounded multiworld integration methods; hand-written artificial IDs51/52 only; model-free services.',
      'old_test_suites_repeated':0,'scientific_worlds':0,'pretrained_models':0,'training_updates':0,'gpu_allocations':0,
      'remote_operations':0,'execution_invocations':1}
save(EXTERNAL/'command.json',meta)
before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();self_start=time.process_time()
with (EXTERNAL/'stdout.log').open('xb') as out,(EXTERNAL/'stderr.log').open('xb') as err:
    try:
        p=subprocess.Popen(command,cwd='/tmp',env=dict(os.environ,**env),stdout=out,stderr=err)
        try:
            code=p.wait(timeout=90);error=None
        except subprocess.TimeoutExpired:
            import signal
            p.send_signal(signal.SIGINT)
            try:code=p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();code=p.wait(timeout=5)
            error='outer timeout; SIGINT cleanup requested before forced termination if necessary'
    except BaseException as exc:
        code,error=None,type(exc).__name__+': '+str(exc)
elapsed=time.monotonic()-start;after=resource.getrusage(resource.RUSAGE_CHILDREN)
response_records=[json.loads(p.read_bytes()) for p in sorted((EXTERNAL/'fixtures').glob('*/artificial-response-record.json'))]
receipt={**meta,'finished_utc':datetime.now(timezone.utc).isoformat(),'returncode':code,'exception':error,
         'elapsed_seconds':elapsed,'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
         'supervisor_cpu_seconds':time.process_time()-self_start,'peak_child_rss_bytes_macos':after.ru_maxrss,
         'artificial_response_returns':sum(r['count'] for r in response_records),'response_records':len(response_records),
         'scope_of_cost':'Child and its waited children; not whole-development CPU or aggregate process-tree RSS. Preparation allowance not reset.'}
save(EXTERNAL/'receipt.json',receipt)
save(EXTERNAL/'closeout.json',{'status':'new_campaign_tests_passed_review_pending' if code==0 else 'failed_attempt_retained',
     'source_pins':sources,'test_receipt_sha256':pin(EXTERNAL/'receipt.json')['sha256'],
     'scientific_provenance_qualified':False,'scientific_screen_executed':False})
members=[{'path':str(q.relative_to(EXTERNAL)),**pin(q)} for q in sorted(EXTERNAL.rglob('*')) if q.is_file()]
save(EXTERNAL/'manifest.json',{'member_count_excluding_manifest':len(members),'members':members})
for q in EXTERNAL.iterdir():
    target=LOCAL/q.name
    if q.is_dir():shutil.copytree(q,target)
    elif q.name=='run_preparation.py':assert q.read_bytes()==target.read_bytes()
    else:
        with target.open('xb') as f:f.write(q.read_bytes())
print(json.dumps(receipt,indent=2))
print((EXTERNAL/'stderr.log').read_text())
print(json.dumps({'custody_files':len(members)+1,'custody_bytes':sum(q.stat().st_size for q in EXTERNAL.rglob('*') if q.is_file())}))
raise SystemExit(0 if code==0 else 1)

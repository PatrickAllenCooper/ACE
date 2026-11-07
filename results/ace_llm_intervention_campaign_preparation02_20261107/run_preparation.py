"""One exclusive R1 publication-only regression run; never replay prep01."""
import ast
from datetime import datetime, timezone
import difflib
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
LOCAL=REPO/'results/ace_llm_intervention_campaign_preparation02_20261107'
EXTERNAL=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-preparation-20261107-02')
INITIAL=REPO/'results/ace_llm_intervention_campaign_preparation_20261107'
INITIAL_EXTERNAL=EXTERNAL.with_name('ACE-LLM-intervention-campaign-preparation-20261107-01')

def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def pin(path):
    raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def verify_initial():
    manifest=json.loads((INITIAL/'manifest.json').read_bytes())
    assert pin(INITIAL/'manifest.json')['sha256']=='9856342ff9927cdcae7fbbe1cc7f7b3916d49ae6bc5536a4ec08df58f42acbf7'
    expected={r['path'] for r in manifest['members']}|{'manifest.json'}
    assert len(expected)==1405
    for root in (INITIAL,INITIAL_EXTERNAL):
        assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==expected
        assert (root/'manifest.json').read_bytes()==(INITIAL/'manifest.json').read_bytes()
        for row in manifest['members']:
            assert pin(root/row['path'])=={k:row[k] for k in ('bytes','sha256')},row['path']
    return {'files_per_copy':1405,'bytes_per_copy':2563875,'manifest_sha256':pin(INITIAL/'manifest.json')['sha256'],
            'verified_utc':datetime.now(timezone.utc).isoformat(),'both_original_capsules_unchanged':True}

EXTERNAL.mkdir(exist_ok=False)
save(EXTERNAL/'initial-custody-before.json',verify_initial())
source=EXTERNAL/'source';source.mkdir()
pending=['llm_intervention_campaign','test_llm_intervention_campaign'];sources={}
while pending:
    module=pending.pop()
    if module in sources:continue
    path=REPO/'scripts/research'/(module+'.py')
    shutil.copyfile(path,source/path.name)
    sources[module]={'original':str(path),**pin(source/path.name)}
    for node in ast.walk(ast.parse(path.read_text())):
        names=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
               [v.name for v in node.names] if isinstance(node,ast.Import) else [])
        pending.extend(name for name in names if name.startswith('llm_intervention_'))
guide=REPO/'docs/development/guidance/ace_llm_intervention_campaign_2026-11-07.md'
shutil.copyfile(guide,source/guide.name)
sources[guide.name]={'original':str(guide),**pin(source/guide.name)}
save(EXTERNAL/'source_manifest.json',sources)
review=REPO/'docs/development/guidance/reviews/ace_llm_intervention_campaign_review_2026-11-07.md'
shutil.copyfile(review,EXTERNAL/review.name)
old=EXTERNAL/'reviewed-source';old.mkdir()
delta=[]
for name in ('llm_intervention_campaign.py','test_llm_intervention_campaign.py',guide.name):
    shutil.copyfile(INITIAL/'source'/name,old/name)
    delta.extend(difflib.unified_diff((old/name).read_text().splitlines(True),(source/name).read_text().splitlines(True),
                                    fromfile='prep01/'+name,tofile='prep02/'+name))
with (EXTERNAL/'delta.patch').open('x') as f:f.write(''.join(delta))
shutil.copyfile(__file__,EXTERNAL/'run_preparation.py')
env={'ACE_CAMPAIGN_PUBLICATION_TEST_OUTPUT':str(EXTERNAL/'fixtures'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
     'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
command=[sys.executable,'-B',str(source/'test_llm_intervention_campaign.py'),'CampaignPublicationTests']
meta={'command':command,'cwd':'/tmp','environment_overrides':env,'python':sys.version,
      'numpy':importlib.metadata.version('numpy'),'started_utc':datetime.now(timezone.utc).isoformat(),
      'scope':'Five NEW pure publication-only R1 regressions; mocked campaign admission and reporting; real exclusive file/link I/O.',
      'old_test_suites_repeated':0,'worlds':0,'environment_responses':0,'policy_processes':0,'pretrained_models':0,
      'training_updates':0,'gpu_allocations':0,'remote_operations':0,'execution_invocations':1}
save(EXTERNAL/'command.json',meta)
before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();self_start=time.process_time()
with (EXTERNAL/'stdout.log').open('xb') as out,(EXTERNAL/'stderr.log').open('xb') as err:
    try:
        result=subprocess.run(command,cwd='/tmp',env=dict(os.environ,**env),stdout=out,stderr=err,timeout=30,check=False)
        code,error=result.returncode,None
    except BaseException as exc:code,error=None,type(exc).__name__+': '+str(exc)
elapsed=time.monotonic()-start;after=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt={**meta,'finished_utc':datetime.now(timezone.utc).isoformat(),'returncode':code,'exception':error,
         'elapsed_seconds':elapsed,'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
         'supervisor_cpu_seconds':time.process_time()-self_start,'peak_child_rss_bytes_macos':after.ru_maxrss,
         'scope_of_cost':'Only test child and waited descendants, not whole-development CPU or aggregate process-tree RSS; preparation allowance not reset.'}
save(EXTERNAL/'receipt.json',receipt)
save(EXTERNAL/'initial-custody-after.json',verify_initial())
save(EXTERNAL/'delta-review-ready.json',{'status':'R1_corrected_new_regressions_passed_delta_review_pending' if code==0 else 'failed_attempt_retained',
     'required_finding':'R1 P2 finalization publication reconciliation','review_sha256':pin(review)['sha256'],
     'source_pins':sources,'test_receipt_sha256':pin(EXTERNAL/'receipt.json')['sha256'],
     'original_1405_files_untouched':True,'prior_test_replays':0,'scientific_provenance_qualified':False})
members=[{'path':str(q.relative_to(EXTERNAL)),**pin(q)} for q in sorted(EXTERNAL.rglob('*')) if q.is_file()]
save(EXTERNAL/'manifest.json',{'member_count_excluding_manifest':len(members),'members':members})
for path in EXTERNAL.iterdir():
    target=LOCAL/path.name
    if path.is_dir():shutil.copytree(path,target)
    elif path.name=='run_preparation.py':assert path.read_bytes()==target.read_bytes()
    else:
        with target.open('xb') as f:f.write(path.read_bytes())
print(json.dumps(receipt,indent=2))
print((EXTERNAL/'stderr.log').read_text())
print(json.dumps({'custody_files':len(members)+1,'custody_bytes':sum(q.stat().st_size for q in EXTERNAL.rglob('*') if q.is_file())}))
raise SystemExit(0 if code==0 else 1)

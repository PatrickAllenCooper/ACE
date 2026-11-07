"""Seal saved artificial evidence only. No engine imports or test rerun."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

repo=Path('/Users/pat/code/ACE')
root=repo/'results/ace_llm_intervention_training_engine_preparation_20261107'
external=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-engine-preparation-20261107-01')
assert not external.exists(), 'external custody must be exclusive'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True)
        f.write('\n')
attempts=[root/'test-attempt01',root/'test-attempt02']
receipts=[json.loads((p/'execution-receipt.json').read_text()) for p in attempts]
assert [r['exit_code'] for r in receipts] == [1,0]
assert 'Ran 6 tests in 0.728s' in (attempts[0]/'combined-output.txt').read_text()
assert 'FAILED (failures=1)' in (attempts[0]/'combined-output.txt').read_text()
assert 'Ran 1 test in 0.446s' in (attempts[1]/'combined-output.txt').read_text()
test_path='scripts/research/test_llm_intervention_training_engine.py'
changed=[name for name,pin in receipts[0]['source_sha256'].items() if receipts[1]['source_sha256'][name]!=pin]
assert changed == [test_path], changed
for p,r in zip(attempts,receipts):
    for name,pin in r['source_sha256'].items():
        assert sha(p/'source'/name) == pin, (p,name)
for name,pin in receipts[-1]['source_sha256'].items():
    assert sha(repo/name) == pin, ('post-test source changed',name)
def tests(path):
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EngineTests')
    return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
old=tests(attempts[0]/'source'/test_path)
new=tests(attempts[1]/'source'/test_path)
old_failed='test_failed_step_retains_prior_steps_nonfinite_loss_and_partial_checkpoint'
new_passed='test_failed_step_retains_prior_steps_nonfinite_scores_and_partial_checkpoint'
assert set(old)-{old_failed} == set(new)-{new_passed}
assert all(old[n]==new[n] for n in old if n!=old_failed)
engine_path='scripts/research/llm_intervention_training_engine.py'
save(root/'closeout.json',{
    'status':'core_implemented_six_unique_controls_passed_across_two_retained_attempts_awaiting_review',
    'finalized_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'finalization_command':[sys.executable,'-B',str(Path(__file__).resolve())],
    'finalization_cwd':'/tmp',
    'test_invocations':2,'failed_test_invocations':1,
    'attempt01':'five methods passed; one assertion expected a later guard',
    'attempt02':'only corrected nonfinite-score failure-custody method passed',
    'five_passing_test_methods_unchanged_and_not_rerun':True,
    'engine_source_unchanged_between_attempts':True,
    'engine_sha256':receipts[-1]['source_sha256'][engine_path],
    'final_source_sha256':receipts[-1]['source_sha256'],
    'total_test_child_cpu_seconds':sum(r['child_user_cpu_seconds']+r['child_system_cpu_seconds'] for r in receipts),
    'total_test_child_wall_seconds':sum(r['elapsed_seconds'] for r in receipts),
    'measurement_scope':'Both new test children including imports; not whole-cycle CPU. No scheduler allocations.',
    'real_qwen_qualification':False,'pretrained_models_loaded':0,'gpu_actions':0,
    'scientific_data_used':False,'remote_actions':0,'installs':0,'git_actions':0,'old_suites_run':0,
    'scientific_provenance_qualified':False,'external_custody':str(external),
    'integration_gaps':['independent Sagan review','reviewed real matched token packet and exposure/checkpoint cadence',
        'authenticated initialized Qwen/GPU09 runtime and recipe','outer data/source/model/provenance/resource freeze',
        'integrated actual-model qualification and telemetry/deadlines','separate numerical training and inference/reload custody'],
})
members=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file()]
save(root/'manifest.json',{'version':'ace-training-engine-preparation-custody-v1',
    'member_count_excluding_manifest':len(members),'member_bytes_excluding_manifest':sum(m['bytes'] for m in members),
    'object_count_including_manifest':len(members)+1,'members':members})
shutil.copytree(root,external)
a={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
b={str(p.relative_to(external)):sha(p) for p in external.rglob('*') if p.is_file()}
assert a==b
print(json.dumps({'status':'sealed_identical_local_copies','files_including_manifest':len(a),
    'bytes_including_manifest':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
    'manifest_sha256':sha(root/'manifest.json'),'engine_sha256':receipts[-1]['source_sha256'][engine_path],
    'total_test_child_cpu_seconds':sum(r['child_user_cpu_seconds']+r['child_system_cpu_seconds'] for r in receipts),
    'total_test_child_wall_seconds':sum(r['elapsed_seconds'] for r in receipts),
    'repository':str(root),'external':str(external)},indent=2))

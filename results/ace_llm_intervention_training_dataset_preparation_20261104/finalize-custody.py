"""One-shot custody finalization only; never imports the implementation."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

repo = Path('/Users/pat/code/ACE')
root = repo/'results/ace_llm_intervention_training_dataset_preparation_20261104'
external = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-dataset-preparation-20261104-01')
assert not external.exists(), 'external custody must be exclusive'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n')

attempt = root/'test-attempt01'
receipt = json.loads((attempt/'execution-receipt.json').read_text())
assert receipt['exit_code'] == 0
log = (attempt/'combined-output.txt').read_text()
assert 'Ran 6 tests in 0.345s' in log and '\nOK\n' in log
observed = json.loads((attempt/'fixtures/test_analytic_counts_cost_crossings_precommitted_teachers_and_matched_views/assembly/plan.json').read_text())['source_file_sha256']
sources = root/'source'
sources.mkdir(exist_ok=False)
copied = {}
for name, expected in observed.items():
    relative = Path('scripts/research')/name
    original, target = repo/relative, sources/relative
    assert sha(original) == expected, ('source changed since tests', name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, target)
    copied[str(relative)] = sha(target)
for relative in (Path('scripts/research/test_llm_intervention_training_dataset.py'),
                 Path('docs/development/guidance/ace_llm_intervention_training_dataset_2026-11-04.md')):
    original, target = repo/relative, sources/relative
    assert sha(original) == receipt['source_sha256'][relative.name]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, target)
    copied[str(relative)] = sha(target)
assert copied['scripts/research/llm_intervention_training_dataset.py'] == receipt['source_sha256']['llm_intervention_training_dataset.py']
context = root/'read-context'
context.mkdir()
for relative in ('docs/development/guidance/ace_llm_intervention_agenda_2026-11-02.md',
                 'docs/development/guidance/reviews/ace_llm_intervention_first_screen_design_2026-11-02.md',
                 'docs/development/guidance/ace_llm_intervention_world_2026-11-04.md'):
    shutil.copyfile(repo/relative, context/Path(relative).name)
attempts = []
for path in sorted((attempt/'fixtures').rglob('plan.json')):
    plan = json.loads(path.read_text())
    if plan.get('version') != 'ace-single-world-training-dataset-v1':
        continue
    assert plan['scope'] == 'artificial'
    assert plan['source_file_sha256'] == observed
    terminal = path.parent/'terminal.json'
    failure = path.parent/'failure.json'
    assert terminal.exists() != failure.exists()
    record = json.loads((terminal if terminal.exists() else failure).read_text())
    assert record['accounting']['accounting_complete']
    attempts.append({'path':str(path.parent.relative_to(root)), 'status':record['status'],
        'reserved_cost_units':record['accounting']['total_reserved_cost_units'],
        'reserved_response_attempts':record['accounting']['reserved_response_attempts'],
        'known_returns':record['accounting']['known_returns'],
        'executor_accepted_responses':record['accounting']['executor_accepted_responses'],
        'unknown_return_attempts':record['accounting']['unknown_return_attempts']})
save(root/'closeout.json', {
    'status':'implemented_artificial_tests_passed_awaiting_independent_review',
    'finalized_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'finalization_command':[sys.executable, '-B', str(Path(__file__).resolve())],
    'finalization_cwd':'/tmp', 'new_test_methods':6, 'test_invocations':1,
    'failed_test_invocations':0, 'nonzero_readonly_discovery_searches':1,
    'test_receipt':'test-attempt01/execution-receipt.json',
    'post_test_source_edits':False, 'source_sha256':copied,
    'artificial_assembly_attempts':attempts,
    'test_child_elapsed_seconds':receipt['elapsed_seconds'],
    'test_child_cpu_seconds':receipt['child_user_cpu_seconds']+receipt['child_system_cpu_seconds'],
    'cpu_measurement_scope':'Only new test child; read-only inspection and editing not fully metered. No scheduler allocation.',
    'scientific_worlds_generated':0, 'models_loaded':0, 'policy_optimizer_updates':0,
    'old_test_suites_run':0, 'remote_actions':0, 'git_actions':0,
    'scientific_provenance_qualified':False,
    'future_scientific_schedule_changed':False,
    'external_custody':str(external),
    'limitations':'See guide and review_request.md. Parent owns review, admission and integration.',
})
members = [{'path':str(p.relative_to(root)), 'bytes':p.stat().st_size, 'sha256':sha(p)}
           for p in sorted(root.rglob('*')) if p.is_file()]
manifest = {'version':'ace-training-dataset-preparation-custody-v1',
    'member_count_excluding_manifest':len(members),
    'member_bytes_excluding_manifest':sum(m['bytes'] for m in members),
    'object_count_including_manifest':len(members)+1, 'members':members}
save(root/'manifest.json', manifest)
shutil.copytree(root, external)
local_files = {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
external_files = {str(p.relative_to(external)):sha(p) for p in external.rglob('*') if p.is_file()}
assert local_files == external_files, 'copy member/hash mismatch'
print(json.dumps({'status':'sealed_identical_local_and_external', 'repository_custody':str(root),
    'external_custody':str(external), 'files_including_manifest':len(local_files),
    'bytes_including_manifest':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
    'manifest_sha256':sha(root/'manifest.json'), 'source_sha256':copied,
    'artificial_attempts':attempts}, indent=2))

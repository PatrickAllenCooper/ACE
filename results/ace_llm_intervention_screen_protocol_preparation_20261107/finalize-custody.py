"""Seal saved pure-test evidence only; no protocol imports or test execution."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

repo = Path('/Users/pat/code/ACE')
root = repo/'results/ace_llm_intervention_screen_protocol_preparation_20261107'
external = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-screen-protocol-preparation-20261107-01')
assert not external.exists(), 'exclusive external custody already exists'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
attempt = root/'test-attempt01'
receipt = json.loads((attempt/'execution-receipt.json').read_text())
assert receipt['exit_code'] == 0
log = (attempt/'combined-output.txt').read_text()
assert 'Ran 6 tests in 2.458s' in log and '\nOK\n' in log
for relative, expected in receipt['source_sha256'].items():
    assert sha(repo/relative) == expected, ('changed since tests', relative)
    assert sha(attempt/'source'/relative) == expected, ('custody source mismatch', relative)
proposal_path = attempt/'fixtures/prospective-proposal.json'
proposal = json.loads(proposal_path.read_text())
declared = json.loads((attempt/'fixtures/declared-source-pins.json').read_text())
assert proposal['source_pins'] == declared
assert len(declared) == 12
assert all(receipt['source_sha256'][path] == expected for path, expected in declared.items())
assert len(proposal['training_worlds']) == 32 and len(proposal['development_worlds']) == 24
assert proposal['worlds_generated'] == proposal['responses_generated'] == 0
assert proposal['scientific_protocol_frozen'] is False
assert proposal['scientific_execution_authorized'] is False
context = root/'read-context'
context.mkdir(exist_ok=False)
for relative in (
    'docs/development/guidance/ace_llm_intervention_agenda_2026-11-02.md',
    'docs/development/guidance/reviews/ace_llm_intervention_first_screen_design_2026-11-02.md',
    'docs/development/guidance/ace_llm_intervention_training_dataset_2026-11-04.md',
    'docs/development/guidance/ace_llm_intervention_world_2026-11-04.md'):
    shutil.copyfile(repo/relative, context/Path(relative).name)
save(root/'closeout.json', {
    'status':'new_implementation_pure_tests_passed_awaiting_independent_review',
    'finalized_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'finalization_command':[sys.executable, '-B', str(Path(__file__).resolve())],
    'finalization_cwd':'/tmp',
    'test_invocations':1, 'new_test_methods':6, 'failed_test_invocations':0,
    'failed_shell_commands':0, 'source_changes_after_test':False,
    'source_sha256':receipt['source_sha256'], 'proposal_file_sha256':sha(proposal_path),
    'source_map_entries':len(declared), 'training_identities':32, 'development_identities':24,
    'confirmation_identities_enumerated':0, 'scientific_protocol_frozen':False,
    'scientific_execution_authorized':False, 'scientific_provenance_qualified':False,
    'worlds_generated':0, 'responses_generated':0, 'models_loaded':0,
    'old_test_suites_run':0, 'remote_actions':0, 'git_actions':0, 'install_actions':0,
    'test_child_wall_seconds':receipt['elapsed_seconds'],
    'test_child_user_cpu_seconds':receipt['child_user_cpu_seconds'],
    'test_child_system_cpu_seconds':receipt['child_system_cpu_seconds'],
    'test_child_total_cpu_seconds':receipt['child_user_cpu_seconds']+receipt['child_system_cpu_seconds'],
    'measurement_scope':'One new test child only. Read/edit/custody work not fully metered. No scheduler allocation accounting changed.',
    'remaining_required_decisions':proposal['remaining_required_decisions'],
    'external_custody':str(external),
})
members = [{'path':str(p.relative_to(root)), 'bytes':p.stat().st_size, 'sha256':sha(p)}
           for p in sorted(root.rglob('*')) if p.is_file()]
save(root/'manifest.json', {'version':'ace-screen-protocol-preparation-custody-v1',
    'members_excluding_manifest':len(members), 'bytes_excluding_manifest':sum(m['bytes'] for m in members),
    'object_count_including_manifest':len(members)+1, 'members':members})
shutil.copytree(root, external)
local = {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
remote_copy = {str(p.relative_to(external)):sha(p) for p in external.rglob('*') if p.is_file()}
assert local == remote_copy
print(json.dumps({'status':'identical_exclusive_local_custody_copies',
    'repository':str(root), 'external':str(external), 'files_including_manifest':len(local),
    'bytes_including_manifest':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
    'manifest_sha256':sha(root/'manifest.json'), 'proposal_sha256':sha(proposal_path),
    'new_source_sha256':{path:pin for path,pin in receipt['source_sha256'].items() if 'screen_protocol' in path},
    'test_child_cpu_seconds':receipt['child_user_cpu_seconds']+receipt['child_system_cpu_seconds']}, indent=2))

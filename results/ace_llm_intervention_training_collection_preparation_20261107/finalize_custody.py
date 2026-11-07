"""Read saved evidence only; close exact external/repository custody without reruns."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

LOCAL = Path('/Users/pat/code/ACE/results/ace_llm_intervention_training_collection_preparation_20261107')
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-collection-preparation-20261107-01')


def pin(path):
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def save_both(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    for root in (EXTERNAL, LOCAL):
        with (root/name).open('xb') as handle:
            handle.write(raw)


with (EXTERNAL/Path(__file__).name).open('xb') as handle:
    handle.write(Path(__file__).read_bytes())
manifest = json.loads((EXTERNAL/'attempt02/source_manifest.json').read_text())
for key, row in manifest.items():
    if key.startswith(('llm_intervention_', 'test_llm_intervention_')) or key == 'ace_llm_intervention_training_collection_2026-11-07.md':
        assert pin(Path(row['original'])) == {k: row[k] for k in ('sha256', 'bytes')}, key
receipt = json.loads((EXTERNAL/'attempt02/receipt.json').read_text())
assert receipt['returncode'] == 0
batches = []
for intent in sorted((EXTERNAL/'attempt02/fixtures').rglob('intent.json')):
    directory = intent.parent
    terminal = directory/('terminal.json' if (directory/'terminal.json').exists() else 'failure.json')
    data = json.loads(terminal.read_text())
    if data['status'] != 'complete':
        assert not (directory/'inventory.json').exists()
        assert data['exposure_ready'] is False
    batches.append({'path': str(directory.relative_to(EXTERNAL)), 'status': data['status'],
        'world_statuses': [w['status'] for w in data['worlds']], 'accounting': data['accounting']})
totals = {k: sum(b['accounting'][k] for b in batches) for k in batches[0]['accounting']}
new = {k: manifest[k] for k in ('llm_intervention_training_collection', 'test_llm_intervention_training_collection',
    'ace_llm_intervention_training_collection_2026-11-07.md')}
save_both('closeout.json', {
    'version': 'ace-training-collection-preparation-v1', 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'stable_source_new_tests_passed_independent_review_pending',
    'local': str(LOCAL), 'external': str(EXTERNAL), 'new_sources': new,
    'exact_all_source_input_pins': 'attempt02/source_manifest.json',
    'source_manifest_sha256': pin(EXTERNAL/'attempt02/source_manifest.json')['sha256'],
    'resource_receipt_sha256': pin(EXTERNAL/'attempt02/receipt.json')['sha256'],
    'preparation_attempts': 2, 'initial_preparation_failures': 1,
    'initial_failure': 'attempt01: SyntaxError in new test declaration during source capture, before test child. Original source and failure retained; wrapper cost not metered.',
    'test_child_invocations': 1, 'new_test_methods_passed': 7, 'old_tests_run': 0,
    'test_child_cpu_seconds': receipt['child_user_cpu_seconds']+receipt['child_system_cpu_seconds'],
    'test_child_wall_seconds': receipt['elapsed_seconds'], 'peak_child_rss_bytes_macos': receipt['child_max_rss_bytes_macos'],
    'cost_scope': 'Only the one NEW test child. Includes its deliberately failed artificial collections. Excludes unmetered implementation, failed preparation wrapper, inspection, review and custody. Not whole-cycle costs or a new allowance.',
    'batch_count': len(batches), 'complete_batches': sum(b['status']=='complete' for b in batches),
    'failed_batches': sum(b['status']=='failed' for b in batches),
    'artificial_reservation_totals_without_double_counting_nested_primitives': totals, 'batches': batches,
    'successful_two_world_fixture': {'world_roster': [17, 4], 'records': 4, 'retained_pairs': 6, 'ties': 2,
        'sft_proposal_steps': 4, 'dpo_proposal_steps': 2, 'response_attempts': 20, 'response_cost_units': 26},
    'failed_or_unstarted_worlds_block_exposure_inventory': True,
    'policy_train': 'Exact current 32-recipe adoption/rejection checked without any scientific descriptor or response. Scientific source/runtime/scheduler/admission/freeze remain caller qualification.',
    'scientific_worlds_created': 0, 'pretrained_models_loaded': 0, 'policy_training_updates': 0,
    'private_evaluations': 0, 'gpu_seconds': 0, 'remote_operations': 0, 'installs': 0, 'commits': 0,
    'independent_review': 'parent managed, pending',
    'immutable_inputs': {k: manifest[k] for k in ('llm_intervention_training_dataset', 'llm_intervention_screen_protocol', 'llm_intervention_training_exposure')},
})
members = [{'path': str(p.relative_to(EXTERNAL)), **pin(p)} for p in sorted(EXTERNAL.rglob('*')) if p.is_file()]
save_both('manifest.json', {'version': 'ace-training-collection-custody-v1', 'members': members,
    'objects_excluding_manifest': len(members), 'bytes_excluding_manifest': sum(m['bytes'] for m in members)})
external = {str(p.relative_to(EXTERNAL)): pin(p) for p in EXTERNAL.rglob('*') if p.is_file()}
local = {str(p.relative_to(LOCAL)): pin(p) for p in LOCAL.rglob('*') if p.is_file()}
assert external == local, 'external/repository custody differs'
print(json.dumps({'objects': len(external), 'bytes': sum(v['bytes'] for v in external.values()),
    'manifest': pin(EXTERNAL/'manifest.json'), 'closeout': pin(EXTERNAL/'closeout.json'),
    'artificial_costs': totals, 'exact_mirror': True}, indent=2))

"""Seal correction evidence only; no tests or responses are replayed by this script."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

REPO = Path('/Users/pat/code/ACE')
LOCAL = REPO/'results/ace_llm_intervention_training_collection_correction_20261107'
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-collection-correction-20261107-01')
INITIAL = REPO/'results/ace_llm_intervention_training_collection_preparation_20261107'
INITIAL_EXTERNAL = EXTERNAL.with_name('ACE-LLM-intervention-training-collection-preparation-20261107-01')


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
initial_manifest_pin = 'f1868edaf7a735f2be2a9eed12496ea7af7ad05619acbb54f324a024d9e6bcdb'
for root in (INITIAL, INITIAL_EXTERNAL):
    assert pin(root/'manifest.json')['sha256'] == initial_manifest_pin
    initial = json.loads((root/'manifest.json').read_text())
    for row in initial['members']:
        assert pin(root/row['path']) == {k: row[k] for k in ('sha256', 'bytes')}
    assert len([p for p in root.rglob('*') if p.is_file()]) == len(initial['members'])+1
manifest = json.loads((EXTERNAL/'attempt01/source_manifest.json').read_text())
new = {k: manifest[k] for k in ('llm_intervention_training_collection', 'test_llm_intervention_training_collection',
    'ace_llm_intervention_training_collection_2026-11-07.md')}
for row in new.values():
    assert pin(Path(row['original'])) == {k: row[k] for k in ('sha256', 'bytes')}
receipt = json.loads((EXTERNAL/'attempt01/receipt.json').read_text())
assert receipt['returncode'] == 0
review = REPO/'docs/development/guidance/reviews/ace_llm_intervention_training_collection_review_2026-11-07.md'
save_both('closeout.json', {
    'status': 'corrected_source_stable_four_new_methods_passed_independent_recheck_pending',
    'finished_utc': datetime.now(timezone.utc).isoformat(), 'local': str(LOCAL), 'external': str(EXTERNAL),
    'new_sources': new, 'all_source_and_input_pins': 'attempt01/source_manifest.json',
    'source_manifest_sha256': pin(EXTERNAL/'attempt01/source_manifest.json')['sha256'],
    'resource_receipt_sha256': pin(EXTERNAL/'attempt01/receipt.json')['sha256'],
    'initial_capsule': str(INITIAL), 'initial_manifest_sha256': initial_manifest_pin,
    'initial_capsule_both_copies_unchanged': True, 'initial_capsule_objects': 422,
    'review_basis': {'path': str(review), 'sha256_at_closeout': pin(review)['sha256'],
        'required_findings': ['P2 terminal publication can leave contradictory final dispositions',
                              'P2 unreadable primitive plan suppresses batch failure receipt'],
        'optional_findings_addressed': ['response-adapter menu/query name bounds in preflight',
                                       'explicit authenticated terminal/disposition gate in exposure example']},
    'corrections': ['fsynced candidate + exclusive hard-link terminal, postcommit reconciliation diagnostic',
        'predeclared journal roster; readable charges preserved, unresolved totals explicitly lower bounds',
        'original failure and full failed/unstarted roster survive optional reconstruction failures',
        'world adapter menu validation over all acquisition/query menus before responses',
        'inventory follows complete accounting guard; exposure example rejects failure/interruption',
        'docstring distinguishes world draws, policy training and private evaluation from paid SCM updates'],
    'test_command': receipt['command'], 'test_child_invocations': 1, 'new_test_methods_passed': 4,
    'previous_successful_methods_repeated': 0, 'older_tests_run': 0,
    'test_child_cpu_seconds': receipt['child_user_cpu_seconds']+receipt['child_system_cpu_seconds'],
    'test_child_wall_seconds': receipt['elapsed_seconds'], 'peak_child_rss_bytes_macos': receipt['child_max_rss_bytes_macos'],
    'cost_scope': 'Only the correction test child; excludes implementation, review, inspection and custody. Saved response journals are copied evidence and add no new responses/charges. Not whole-cycle costs.',
    'new_response_attempts': 0, 'new_response_cost_units': 0,
    'scientific_worlds_created': 0, 'pretrained_models_loaded': 0, 'policy_training_updates': 0,
    'private_evaluations': 0, 'gpu_seconds': 0, 'remote_operations': 0, 'installs': 0, 'commits': 0,
    'runtime_scheduler_source_scientific_freeze': 'caller qualification remains open',
})
members = [{'path': str(p.relative_to(EXTERNAL)), **pin(p)} for p in sorted(EXTERNAL.rglob('*')) if p.is_file()]
save_both('manifest.json', {'version': 'ace-training-collection-correction-custody-v1', 'members': members,
    'objects_excluding_manifest': len(members), 'bytes_excluding_manifest': sum(m['bytes'] for m in members)})
external = {str(p.relative_to(EXTERNAL)): pin(p) for p in EXTERNAL.rglob('*') if p.is_file()}
local = {str(p.relative_to(LOCAL)): pin(p) for p in LOCAL.rglob('*') if p.is_file()}
assert external == local
print(json.dumps({'objects': len(external), 'bytes': sum(p['bytes'] for p in external.values()),
    'manifest': pin(EXTERNAL/'manifest.json'), 'closeout': pin(EXTERNAL/'closeout.json'), 'exact_mirror': True}, indent=2))

"""Read saved outputs and seal exact copies; does not run the reporter/tests."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path('/Users/pat/code/ACE')
LOCAL = REPO/'results/ace_llm_intervention_cost_curves_preparation_20261107'
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-cost-curves-preparation-20261107-01')


def pin(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def save(path, raw):
    with path.open('xb') as handle:
        handle.write(raw)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


receipt = json.loads((LOCAL/'attempt01/receipt.json').read_bytes())
sources = json.loads((LOCAL/'attempt01/source_manifest.json').read_bytes())
assert receipt['returncode'] == 0 and receipt['runner_exception'] is None
assert 'Ran 7 tests' in (LOCAL/'attempt01/stderr.log').read_text()
for source in sources.values():
    assert pin(Path(source['original'])) == {k: source[k] for k in ('bytes', 'sha256')}
for name, expected in receipt['files'].items():
    assert pin(LOCAL/'attempt01'/name) == expected
    assert pin(EXTERNAL/'attempt01'/name) == expected
assert (LOCAL/'attempt01/receipt.json').read_bytes() == (EXTERNAL/'attempt01/receipt.json').read_bytes()
attempt_files = {str(p.relative_to(LOCAL/'attempt01')): pin(p)
                 for p in sorted((LOCAL/'attempt01').rglob('*')) if p.is_file()}
assert set(attempt_files) == set(receipt['files']) | {'receipt.json'}
for name in ('run_preparation.py', 'finalize_custody.py', 'review_request.md'):
    save(EXTERNAL/name, (LOCAL/name).read_bytes())
reports = [json.loads(p.read_bytes()) for p in (LOCAL/'attempt01/fixtures').rglob('report.json')]
closeout = {
    'status': 'new_artificial_methods_passed_independent_review_pending',
    'closed_utc': datetime.now(timezone.utc).isoformat(),
    'new_test_methods': 7, 'test_invocations': 1, 'failed_test_invocations': 0, 'old_tests_run': 0,
    'saved_reports': len(reports),
    'operational_failure_reports': sum(r['operational_status'] == 'failed' for r in reports),
    'evaluation_failure_reports': sum(r['report_status'] == 'evaluation_failed' for r in reports),
    'retained_rejected_bundles': len(list((LOCAL/'attempt01/fixtures').rglob('rejection.json'))),
    'attempt_file_count': len(attempt_files), 'attempt_bytes': sum(p['bytes'] for p in attempt_files.values()),
    'attempt_receipt_sha256': pin(LOCAL/'attempt01/receipt.json')['sha256'],
    'source_pins': sources,
    'test_child_cpu_seconds': receipt['child_user_cpu_seconds'] + receipt['child_system_cpu_seconds'],
    'test_child_wall_seconds': receipt['elapsed_seconds'],
    'cost_scope': 'Only the new test subprocess is measured; implementation/inspection/copying unmetered. No allowance reset or scheduler change.',
    'generator_calls': 0, 'new_environment_responses': 0, 'scientific_generation': 0,
    'model_loads': 0, 'policy_training': 0, 'remote_operations': 0, 'installs': 0,
    'commits': 0, 'gpu_seconds': 0,
    'source_split_execution_provenance_qualified': False,
    'caller_obligations': 'Frozen paired roster/cap/task/configuration, original journal reconciliation, independent pins, private evaluator separation, failure retention and shared-initial ledger deduplication.',
    'finalization_command': [sys.executable, '-B', str(Path(__file__))],
    'external_custody': str(EXTERNAL)}
closeout_raw = encoded(closeout)
save(LOCAL/'closeout.json', closeout_raw)
save(EXTERNAL/'closeout.json', closeout_raw)
members = [{'path': str(p.relative_to(LOCAL)), **pin(p)} for p in sorted(LOCAL.rglob('*')) if p.is_file()]
manifest = {'version': 'ace-cost-curves-preparation-custody-v1', 'members': members,
            'member_count_excluding_manifest': len(members),
            'member_bytes_excluding_manifest': sum(m['bytes'] for m in members),
            'object_count_including_manifest': len(members)+1}
manifest_raw = encoded(manifest)
save(LOCAL/'manifest.json', manifest_raw)
save(EXTERNAL/'manifest.json', manifest_raw)
expected_paths = {m['path'] for m in members} | {'manifest.json'}
for root in (LOCAL, EXTERNAL):
    assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} == expected_paths
    for m in members:
        assert pin(root/m['path']) == {k: m[k] for k in ('bytes', 'sha256')}
    assert (root/'manifest.json').read_bytes() == manifest_raw
print(json.dumps({'files': len(expected_paths),
    'bytes_including_manifest': manifest['member_bytes_excluding_manifest']+len(manifest_raw),
    'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
    'closeout_sha256': hashlib.sha256(closeout_raw).hexdigest(),
    'test_child_cpu_seconds': closeout['test_child_cpu_seconds'],
    'test_child_wall_seconds': closeout['test_child_wall_seconds'],
    'saved_reports': len(reports), 'retained_rejected_bundles': closeout['retained_rejected_bundles']}, indent=2))

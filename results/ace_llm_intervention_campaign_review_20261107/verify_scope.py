"""Read-only custody check: never imports or executes reviewed modules/tests."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

REPO = Path('/Users/pat/code/ACE')
LOCAL = REPO/'results/ace_llm_intervention_campaign_preparation_20261107'
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-preparation-20261107-01')

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = read(LOCAL/'manifest.json')
members = manifest['members']
expected = {m['path'] for m in members} | {'manifest.json'}
assert len(members) == manifest['member_count_excluding_manifest']
assert len(expected) == len(members)+1
for root in (LOCAL, EXTERNAL):
    paths = list(root.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    actual = {str(p.relative_to(root)) for p in paths if p.is_file()}
    assert actual == expected, (str(root), actual ^ expected)
    for member in members:
        p = root/member['path']
        assert p.stat().st_size == member['bytes'], str(p)
        assert sha(p) == member['sha256'], str(p)
assert (LOCAL/'manifest.json').read_bytes() == (EXTERNAL/'manifest.json').read_bytes()

sources = read(LOCAL/'source_manifest.json')
for spec in sources.values():
    original = Path(spec['original'])
    assert sha(original) == spec['sha256'], str(original)
    assert sha(LOCAL/'source'/original.name) == spec['sha256'], str(original)

receipt = read(LOCAL/'receipt.json')
closeout = read(LOCAL/'closeout.json')
assert closeout['test_receipt_sha256'] == sha(LOCAL/'receipt.json')
assert closeout['source_pins'] == sources
assert receipt['returncode'] == 0 and receipt['exception'] is None
assert receipt['execution_invocations'] == 1
test_ast = ast.parse((LOCAL/'source/test_llm_intervention_campaign.py').read_text())
test_names = sorted(n.name for n in ast.walk(test_ast)
                    if isinstance(n, ast.FunctionDef) and n.name.startswith('test_'))
stderr = (LOCAL/'stderr.log').read_text()
assert len(test_names) == 8
assert stderr.count(' ... ok\n') == 8 and 'Ran 8 tests' in stderr and stderr.endswith('OK\n')
for name in test_names:
    assert name+' (__main__.CampaignTests) ... ok' in stderr
records = [read(p) for p in sorted((LOCAL/'fixtures').glob('*/artificial-response-record.json'))]
assert len(records) == receipt['response_records'] == 8
for record in records:
    assert record['count'] == len(record['responses'])
    assert all(r['world_id'] in (51, 52) for r in record['responses'])
assert sum(r['count'] for r in records) == receipt['artificial_response_returns'] == 27
for field in ('gpu_allocations', 'old_test_suites_repeated', 'pretrained_models',
              'remote_operations', 'scientific_worlds', 'training_updates'):
    assert receipt[field] == 0

result = {
    'verified_utc': datetime.now(timezone.utc).isoformat(),
    'verification_scope': 'Read-only byte/custody and recorded-scope audit, no module imports, tests, worlds, models, subprocesses or remote work.',
    'local_and_external_custody_files': len(expected),
    'local_and_external_bytes': sum((LOCAL/p).stat().st_size for p in expected),
    'manifest_sha256': sha(LOCAL/'manifest.json'),
    'receipt_sha256': sha(LOCAL/'receipt.json'),
    'source_manifest_sha256': sha(LOCAL/'source_manifest.json'),
    'current_source_hashes_match': {Path(s['original']).name: s['sha256'] for s in sources.values()},
    'recorded_passed_test_names': test_names,
    'recorded_execution_invocations': receipt['execution_invocations'],
    'verified_artificial_response_returns': sum(r['count'] for r in records),
    'recorded_child_cpu_seconds': receipt['child_cpu_seconds'],
    'recorded_elapsed_seconds': receipt['elapsed_seconds'],
    'recorded_peak_child_rss_bytes_macos': receipt['peak_child_rss_bytes_macos'],
    'resource_measurement_scope': receipt['scope_of_cost'],
}
print(json.dumps(result, indent=2, sort_keys=True))

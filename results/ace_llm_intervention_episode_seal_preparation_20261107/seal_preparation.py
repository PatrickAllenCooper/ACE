"""One local inventory/copy; does not execute adapter, reporter or tests."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/pat/code/ACE')
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-episode-seal-preparation-20261107-01')

def metadata(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

if any(EXTERNAL.iterdir()) or (ROOT/'manifest.json').exists():
    raise FileExistsError('nonexclusive custody seal/copy destination')
receipt = json.loads((ROOT/'test-attempt01/receipt.json').read_text())
for relative, pin in receipt['source'].items():
    if metadata(REPO/relative) != pin or metadata(ROOT/'source01'/relative) != pin:
        raise ValueError('tested source/closure changed: '+relative)
references = {}
for name in ('llm_intervention_cost_curves.py', 'llm_intervention_episode.py',
             'llm_intervention_replicated_branches.py', 'llm_intervention_task.py', 'llm_intervention_scm.py'):
    dest = ROOT/'reference'/name
    dest.parent.mkdir(exist_ok=True)
    shutil.copyfile(REPO/'scripts/research'/name, dest)
    references[name] = metadata(dest)
closeout = {'status': 'prepared_for_independent_review', 'source_fixed': True,
    'scientific_provenance_qualified': False, 'scientific_acceptance': False,
    'tests_passed': 6, 'test_runs': 1, 'failed_test_attempts': 0,
    'failed_preliminary_read_commands': 1,
    'failed_read_explanation': 'guessed separate response-adapter file absent; actual class found in world source; command retained',
    'unittest_elapsed_output': 'Ran 6 tests in 0.097s', 'test_receipt': 'test-attempt01/receipt.json',
    'child_cpu_seconds': receipt['child_total_cpu_seconds'], 'elapsed_seconds': receipt['elapsed_seconds'],
    'source': receipt['source'], 'read_only_schema_reference': references,
    'pending': ['independent review', 'parent actual-runner/finalizer new-layer integration',
                'caller source/runtime/independent pins and outer provenance receipt qualification'],
    'execution_scope': 'handcrafted JSON/hash-chain adapter fixtures only; no old tests or upstream execution',
    'seal_command': '/Users/pat/code/ACE/.venv_test/bin/python -B '+str(ROOT/'seal_preparation.py'),
    'seal_cwd': '/tmp', 'external_custody': str(EXTERNAL)}
(ROOT/'closeout.json').write_text(json.dumps(closeout, indent=2, sort_keys=True)+'\n')
members = {str(p.relative_to(ROOT)): metadata(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}
manifest = {'scope': 'exclusive artificial adapter preparation, not scientific protocol freeze', 'excluded_self': 'manifest.json',
    'member_count': len(members), 'member_bytes': sum(v['bytes'] for v in members.values()), 'members': members}
(ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
files = [p for p in sorted(ROOT.rglob('*')) if p.is_file()]
for p in files:
    target = EXTERNAL/p.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(str(target))
    shutil.copyfile(p, target)
local = {str(p.relative_to(ROOT)): metadata(p) for p in files}
external = {str(p.relative_to(EXTERNAL)): metadata(p) for p in EXTERNAL.rglob('*') if p.is_file()}
if local != external:
    raise ValueError('copy member/hash mismatch')
print(json.dumps({'files_including_manifest': len(files), 'bytes_including_manifest': sum(v['bytes'] for v in local.values()),
    'manifest': metadata(ROOT/'manifest.json'), 'copy_member_hash_equality': True}, indent=2, sort_keys=True))

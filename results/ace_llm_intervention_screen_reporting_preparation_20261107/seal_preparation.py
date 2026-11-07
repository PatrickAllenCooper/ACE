"""Single-use local preparation inventory/copy; no tests or source execution."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/pat/code/ACE')
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-screen-reporting-preparation-20261107-01')

def metadata(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

if any(EXTERNAL.iterdir()) or (ROOT/'manifest.json').exists():
    raise FileExistsError('custody seal/copy destination is not exclusive')
receipt = json.loads((ROOT/'test-attempt01/receipt.json').read_text())
for relative, pin in receipt['source'].items():
    if metadata(REPO/relative) != pin or metadata(ROOT/'source01'/relative) != pin:
        raise ValueError('tested source changed: '+relative)
commands = json.loads((ROOT/'command_receipts.json').read_text())
reference = next(c['output'] for c in commands if c.get('label') == 'reporter')
(ROOT/'reference').mkdir()
(ROOT/'reference/llm_intervention_cost_curves_read_only.py').write_text(reference)
closeout = {'status': 'prepared_for_independent_review', 'source_fixed': True,
    'scientific_provenance_qualified': False, 'scientific_acceptance': False,
    'test_methods_passed': 6, 'test_runs': 1, 'failed_test_attempts': 0, 'failed_preliminary_executions': 0,
    'unittest_elapsed_output': 'Ran 6 tests in 0.076s', 'test_receipt': 'test-attempt01/receipt.json',
    'child_cpu_seconds': receipt['child_total_cpu_seconds'], 'elapsed_seconds': receipt['elapsed_seconds'],
    'source': receipt['source'], 'schema_reference': metadata(ROOT/'reference/llm_intervention_cost_curves_read_only.py'),
    'reference_scope': 'exact source text read for interface; not imported/executed/re-reviewed',
    'pending': ['independent review', 'parent integration', 'caller source/receipt/execution provenance',
                'caller prospective roster/strata and independent byte pins'],
    'limitations': ['stipulated artificial report errors and receipt-pin labels only',
        'no generated worlds, scored responses, fits, models or scientific roster',
        'no upstream reporter or old tests executed', 'physical cache spending is not inferable from logical per-arm charges'],
    'seal_command': '/Users/pat/code/ACE/.venv_test/bin/python -B '+str(ROOT/'seal_preparation.py'),
    'seal_cwd': '/tmp', 'external_custody': str(EXTERNAL)}
(ROOT/'closeout.json').write_text(json.dumps(closeout, indent=2, sort_keys=True)+'\n')
members = {str(p.relative_to(ROOT)): metadata(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}
manifest = {'scope': 'exclusive artificial reporting preparation, not scientific freeze',
    'excluded_self': 'manifest.json', 'member_count': len(members),
    'member_bytes': sum(m['bytes'] for m in members.values()), 'members': members}
(ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
files = [p for p in sorted(ROOT.rglob('*')) if p.is_file()]
for p in files:
    target = EXTERNAL/p.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(str(target))
    shutil.copyfile(p, target)
external_files = {str(p.relative_to(EXTERNAL)): metadata(p) for p in EXTERNAL.rglob('*') if p.is_file()}
local_files = {str(p.relative_to(ROOT)): metadata(p) for p in files}
if local_files != external_files:
    raise ValueError('external copy inventory/hash mismatch')
print(json.dumps({'all_files_including_manifest': len(files), 'all_bytes_including_manifest': sum(v['bytes'] for v in local_files.values()),
    'manifest': metadata(ROOT/'manifest.json'), 'copy_member_hash_equality': True, 'source': receipt['source']}, indent=2, sort_keys=True))

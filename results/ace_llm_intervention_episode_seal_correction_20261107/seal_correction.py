"""Single-use correction custody inventory/copy; no test or project execution."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/pat/code/ACE')
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-episode-seal-correction-20261107-01')
ORIGINAL = REPO/'results/ace_llm_intervention_episode_seal_preparation_20261107'
OLD_EXTERNAL = EXTERNAL.parent/'ACE-LLM-intervention-episode-seal-preparation-20261107-01'

def metadata(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

if any(EXTERNAL.iterdir()) or (ROOT/'manifest.json').exists():
    raise FileExistsError('correction seal destination not exclusive')
receipt = json.loads((ROOT/'test-attempt01/receipt.json').read_text())
for relative, pin in receipt['source'].items():
    if metadata(ROOT/'source02'/relative) != pin or metadata(REPO/relative) != pin:
        raise ValueError('frozen correction source/context differs: '+relative)
for root in (ORIGINAL, OLD_EXTERNAL):
    if metadata(root/'manifest.json')['sha256'] != receipt['original_capsule_manifest_sha256']:
        raise ValueError('original custody manifest changed')
test_path = 'scripts/research/test_llm_intervention_episode_seal.py'
def old_methods(path):
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'SealTests')
    return {m.name: ast.dump(m, include_attributes=False) for m in cls.body if isinstance(m, ast.FunctionDef) and m.name.startswith('test_')}
old, current = old_methods(ORIGINAL/'source01'/test_path), old_methods(REPO/test_path)
if len(old) != 6 or old != current:
    raise ValueError('original test method bodies changed')
closeout = {'status': 'fixed_for_narrow_independent_recheck', 'original_required_finding': 'one P2 terminal posterior reconciliation gap',
    'scientific_acceptance': False, 'test_runs': 1, 'new_methods_passed': 4, 'failed_executions': 0,
    'original_six_methods_executed': False, 'original_six_method_ast_unchanged': True,
    'fixture_helper_change': 'placeholder means/covariances replaced by exact rational small-matrix oracle',
    'child_cpu_seconds': receipt['child_total_cpu_seconds'], 'elapsed_seconds': receipt['elapsed_seconds'],
    'unittest_elapsed_output': 'Ran 4 tests in 0.038s', 'test_receipt': 'test-attempt01/receipt.json', 'source': receipt['source'],
    'original_local_external_manifest_pins_unchanged': True,
    'original_capsule_manifest_sha256': receipt['original_capsule_manifest_sha256'],
    'original_capsule_write_operations': 0, 'original_member_audit_repeated': False,
    'pending': ['Zeno narrow correction review', 'Sagan separate saved-artifact affected paired-finalization recheck'],
    'sagan_ready_notification_sent': True,
    'seal_command': '/Users/pat/code/ACE/.venv_test/bin/python -B '+str(ROOT/'seal_correction.py'),
    'seal_cwd': '/tmp', 'external_custody': str(EXTERNAL)}
(ROOT/'closeout.json').write_text(json.dumps(closeout, indent=2, sort_keys=True)+'\n')
members = {str(p.relative_to(ROOT)): metadata(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}
manifest = {'scope': 'exclusive posterior correction custody, not scientific freeze', 'excluded_self': 'manifest.json',
    'member_count': len(members), 'member_bytes': sum(v['bytes'] for v in members.values()), 'members': members}
(ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
files = [p for p in sorted(ROOT.rglob('*')) if p.is_file()]
for p in files:
    target = EXTERNAL/p.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): raise FileExistsError(str(target))
    shutil.copyfile(p, target)
local = {str(p.relative_to(ROOT)): metadata(p) for p in files}
external = {str(p.relative_to(EXTERNAL)): metadata(p) for p in EXTERNAL.rglob('*') if p.is_file()}
if local != external: raise ValueError('correction copy member/hash mismatch')
print(json.dumps({'files_including_manifest': len(files), 'bytes_including_manifest': sum(v['bytes'] for v in local.values()),
    'manifest': metadata(ROOT/'manifest.json'), 'copy_member_hash_equality': True,
    'original_manifest_pins_unchanged': True, 'original_six_method_ast_unchanged': True}, indent=2, sort_keys=True))

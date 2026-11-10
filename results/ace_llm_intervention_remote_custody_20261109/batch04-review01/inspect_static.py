"""Reviewer byte/AST inspection only: never imports or calls reviewed code."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import stat
import time

START = time.monotonic()
CPU = time.process_time()
ROOT = Path('/Users/pat/code/ACE/results/ace_llm_intervention_remote_custody_20261109/gpu-audit-batch04')
OUT = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
raw = lambda value: (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()

def need(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    st = path.lstat()
    need(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, 'ordinary single-link source/evidence required: ' + str(path))
    need(st.st_size <= 8 * 1024**2, 'source preparation member unexpectedly large')
    data = path.read_bytes()
    after = path.lstat()
    need((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'source changed while read')
    return data

pins = json.loads(read(ROOT / 'source-pins.json'))
for name, expected in pins.items():
    need(sha(read(ROOT / name)) == expected, 'recorded source pin mismatch: ' + name)
need(pins['batch_gpu_audit.py'] == 'a1b00ed9d65aeb834b386c684c40aeba46c673b19e53d1b716abb7d224edfdd7', 'independent requested batch pin')
need(pins['references/aid-audit_saved_gpu72.py'] == 'b76e13e2b70de3e1f638ef163427186e03ac6f69634a0d0c6914821db25b6d3a', 'aid pin')
need(pins['references/aid-independent-review.md'] == 'cc2f9c4a82f9f85907cceb2f38a04c5512e5d0ebcc59ed894767830c1543c6ac', 'aid review pin')
need(pins['dependencies/remote_custody.py'] == '4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f', 'backend pin')
old = read(ROOT / 'predecessor/batch03/batch_gpu_audit.py')
new = read(ROOT / 'batch_gpu_audit.py')
need(sha(old) == '555b571486500043118b730eb1a46f67886c01ef7af5734313da2b5c3df7abfd', 'closed predecessor pin')
trees = [ast.parse(data) for data in (old, new)]
defs = [{n.name: ast.dump(n, include_attributes=False) for n in t.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} for t in trees]
changed = sorted(n for n in defs[0].keys() & defs[1].keys() if defs[0][n] != defs[1][n])
added = sorted(defs[1].keys() - defs[0].keys())
need(changed == sorted(['select_members', '_selection', '_result', 'fetch_audit', '_fetch_audit']), 'unexpected changed definition')
need(added == ['_control_binding'] and not (defs[0].keys() - defs[1].keys()), 'unexpected function addition/removal')
new_assignments = {'CONTROL_ROOT', 'FAILED_GPU_ROOT', 'GPU_OUTPUT', 'RECOVERY_FILES', 'SERVICES'}
def stripped(tree, successor):
    retained = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in set(changed + added):
            continue
        if successor and isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in new_assignments for t in node.targets):
            continue
        retained.append(ast.dump(node, include_attributes=False))
    return retained
need(stripped(trees[0], False) == stripped(trees[1], True), 'undisclosed top-level/source change')
diff = ''.join(difflib.unified_diff(old.decode().splitlines(keepends=True), new.decode().splitlines(keepends=True), fromfile='batch03/batch_gpu_audit.py', tofile='batch04/batch_gpu_audit.py')).encode()
need(diff == read(ROOT / 'source.diff'), 'provided source diff not exact')
run = json.loads(read(ROOT / 'evidence/attempt01/run.json'))
need(run['returncode'] == 0 and run['new_methods'] == 3 and run['old_methods_repeated'] == 0, 'retained test disposition')
for name, expected in run['source_pins'].items():
    need(sha(read(ROOT / name)) == expected == sha(read(ROOT / 'evidence/attempt01/source' / name)), 'tested snapshot mismatch: ' + name)
stderr = read(ROOT / 'evidence/attempt01/stderr.txt').decode()
need('Ran 3 tests' in stderr and stderr.rstrip().endswith('OK'), 'test transcript disposition')
copies = json.loads(read(ROOT / 'copy-receipt.json'))
mirror = Path(copies['copies'][1])
local_members = {str(p.relative_to(ROOT)): p for p in ROOT.rglob('*') if p.is_file() or p.is_symlink()}
external_members = {str(p.relative_to(mirror)): p for p in mirror.rglob('*') if p.is_file() or p.is_symlink()}
need(set(local_members) == set(external_members), 'author mirror file membership differs')
local_dirs = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_dir())
external_dirs = sorted(str(p.relative_to(mirror)) for p in mirror.rglob('*') if p.is_dir())
need(local_dirs == external_dirs, 'author mirror directory membership differs')
inventory = {}
for name, path in sorted(local_members.items()):
    other = external_members[name]
    a, b = read(path), read(other)
    need(a == b and not path.samefile(other), 'mirror not independent matching bytes: ' + name)
    inventory[name] = {'bytes': len(a), 'sha256': sha(a)}
(OUT / 'author-preparation-inventory.json').write_bytes(raw({'copies': [str(ROOT), str(mirror)], 'files': inventory, 'directories': local_dirs}))
result = dict(status='static_inspection_complete', source_sha256=sha(new), predecessor_sha256=sha(old),
    changed_functions=changed, added_functions=added,
    unchanged_functions=sorted(n for n in defs[0] if n not in changed),
    other_top_level_AST_identical=True, provided_diff_exact=True, all_source_pins_verified=True,
    tested_source_snapshots_equal=True, original_preparation_copies_independent_and_identical=True,
    original_preparation_files=len(inventory), original_preparation_bytes=sum(v['bytes'] for v in inventory.values()),
    original_preparation_inventory_sha256=sha((OUT / 'author-preparation-inventory.json').read_bytes()),
    retained_test_receipt=run, reviewed_source_executions=0, test_executions=0,
    remote_actions=0, real_locks_acquired=0, acceptance=False,
    local_static_inspection_process_cpu_seconds=time.process_time()-CPU,
    local_static_inspection_elapsed_seconds=time.monotonic()-START)
(OUT / 'static-inspection.json').write_bytes(raw(result))
print(json.dumps({k: result[k] for k in ['status','source_sha256','original_preparation_files','original_preparation_bytes','local_static_inspection_process_cpu_seconds','local_static_inspection_elapsed_seconds']}))

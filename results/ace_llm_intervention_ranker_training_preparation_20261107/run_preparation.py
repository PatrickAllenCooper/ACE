"""Single new artificial test invocation; exclusive, retained source and evidence."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

REPO = Path('/Users/pat/code/ACE')
LOCAL = REPO/'results/ace_llm_intervention_ranker_training_preparation_20261107'
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-ranker-training-preparation-20261107-01')


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def pin(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def copy_exclusive(source, destination):
    with destination.open('xb') as handle:
        handle.write(source.read_bytes())


EXTERNAL.mkdir(exist_ok=False)
attempt = EXTERNAL/'attempt01'
attempt.mkdir()
source = attempt/'source'
source.mkdir()
save(EXTERNAL/'preparation.json', {
    'started_utc': datetime.now(timezone.utc).isoformat(), 'local': str(LOCAL), 'external': str(EXTERNAL),
    'scope': 'Matched numerical training control using new tiny artificial rankers, no pretrained model or scientific admission.',
    'test_invocation_limit': 1, 'old_tests_run': 0, 'scientific_generation': 0, 'pretrained_models': 0,
    'tiny_artificial_numeric_optimization': True, 'remote_operations': 0, 'installs': 0, 'commits': 0,
    'cost_scope': 'Test child CPU is metered below; implementation/inspection/custody unmetered. No allowance reset.'})
pending = ['llm_intervention_ranker_training', 'test_llm_intervention_ranker_training']
copied = {}
while pending:
    module = pending.pop()
    if module in copied:
        continue
    original = REPO/'scripts/research'/(module+'.py')
    target = source/original.name
    copy_exclusive(original, target)
    copied[module] = {'original': str(original), **pin(target)}
    for node in ast.walk(ast.parse(target.read_text())):
        names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                 [item.name for item in node.names] if isinstance(node, ast.Import) else [])
        pending.extend(name for name in names if name.startswith('llm_intervention_'))
guide = REPO/'docs/development/guidance/ace_llm_intervention_ranker_training_2026-11-07.md'
copy_exclusive(guide, source/guide.name)
copied[guide.name] = {'original': str(guide), **pin(source/guide.name)}
save(attempt/'source_manifest.json', copied)
copy_exclusive(Path(__file__), attempt/'run_preparation.py')
copy_exclusive(Path(__file__), EXTERNAL/'run_preparation.py')
command = [sys.executable, '-B', str(source/'test_llm_intervention_ranker_training.py')]
environment = {'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '', 'ACE_NUMERIC_TRAINING_TEST_OUTPUT': str(attempt/'fixtures')}
invocation = {'command': command, 'cwd': str(attempt), 'environment_overrides': environment,
    'preparation_command': [sys.executable, '-B', str(Path(__file__))], 'timeout_seconds': 60,
    'python': sys.version, 'platform': platform.platform(), 'numpy': importlib.metadata.version('numpy'), 'torch': importlib.metadata.version('torch'),
    'started_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Six NEW tiny numerical training methods plus their independent literal optimizer oracle only.'}
save(attempt/'command.json', invocation)
before = resource.getrusage(resource.RUSAGE_CHILDREN)
started = time.monotonic()
with (attempt/'stdout.log').open('xb') as stdout, (attempt/'stderr.log').open('xb') as stderr:
    try:
        completed = subprocess.run(command, cwd=attempt, env=dict(os.environ, **environment),
            stdout=stdout, stderr=stderr, timeout=60, check=False)
        returncode, exception = completed.returncode, None
    except BaseException as exc:
        returncode, exception = None, type(exc).__name__+': '+str(exc)
elapsed = time.monotonic()-started
after = resource.getrusage(resource.RUSAGE_CHILDREN)
receipt = {**invocation, 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'returncode': returncode, 'runner_exception': exception, 'elapsed_seconds': elapsed,
    'child_user_cpu_seconds': after.ru_utime-before.ru_utime,
    'child_system_cpu_seconds': after.ru_stime-before.ru_stime, 'child_max_rss_bytes_macos': after.ru_maxrss,
    'files': {str(p.relative_to(attempt)): pin(p) for p in sorted(attempt.rglob('*')) if p.is_file()}}
save(attempt/'receipt.json', receipt)
save(EXTERNAL/'closeout.json', {'status': 'new_artificial_methods_passed_independent_review_pending' if returncode == 0 else 'failed_attempt_retained',
    'finished_utc': datetime.now(timezone.utc).isoformat(), 'test_invocations': 1,
    'failed_test_invocations': int(returncode != 0), 'new_test_methods': 6,
    'completed_numeric_fits': len(list((attempt/'fixtures').glob('*/fit/terminal.json'))),
    'failed_numeric_fits': len(list((attempt/'fixtures').glob('*/fit/failure.json'))),
    'independent_literal_optimizer_oracle_present': True, 'source_pins': copied,
    'attempt_receipt_sha256': pin(attempt/'receipt.json')['sha256'],
    'test_child_cpu_seconds': receipt['child_user_cpu_seconds']+receipt['child_system_cpu_seconds'],
    'test_child_wall_seconds': elapsed, 'prospective_training_component_only': True, 'scientific_adoption': False,
    'old_tests_run': 0, 'pretrained_model_loads': 0, 'scientific_generation': 0, 'scientific_training_updates': 0,
    'gpu_seconds': 0, 'remote_operations': 0, 'installs': 0, 'commits': 0,
    'cost_scope': 'Original test child only; implementation, inspection and custody CPU not fully metered.',
    'caller_obligations': 'Independent source/record/receipt/full-roster provenance, review/adoption, exact target runtime and outer resource supervision.'})
members = [{'path': str(p.relative_to(EXTERNAL)), **pin(p)} for p in sorted(EXTERNAL.rglob('*')) if p.is_file()]
save(EXTERNAL/'manifest.json', {'version': 'ace-ranker-training-preparation-custody-v1',
    'member_count_excluding_manifest': len(members), 'member_bytes_excluding_manifest': sum(p['bytes'] for p in members),
    'object_count_including_manifest': len(members)+1, 'members': members})
for path in sorted(EXTERNAL.iterdir()):
    target = LOCAL/path.name
    if path.is_dir():
        shutil.copytree(path, target)
    elif path.name == 'run_preparation.py':
        if path.read_bytes() != target.read_bytes():
            raise RuntimeError('preparation source changed during execution')
    else:
        copy_exclusive(path, target)
print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}, indent=2))
print((attempt/'stderr.log').read_text())
print(json.dumps({'custody_files': len(members)+1,
                  'custody_bytes': sum(p.stat().st_size for p in EXTERNAL.rglob('*') if p.is_file())}))
raise SystemExit(0 if returncode == 0 else 1)

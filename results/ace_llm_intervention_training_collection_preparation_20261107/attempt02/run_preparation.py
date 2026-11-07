"""Exclusive local CPU evidence for NEW collection tests; no old suite or remote work."""
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
LOCAL = REPO/'results/ace_llm_intervention_training_collection_preparation_20261107'
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-collection-preparation-20261107-01')


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def pin(path):
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def copy(source, destination):
    with destination.open('xb') as handle:
        handle.write(source.read_bytes())


number = int(sys.argv[1])
if number == 1:
    EXTERNAL.mkdir(exist_ok=False)
    save(EXTERNAL/'preparation.json', {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'local': str(LOCAL), 'external': str(EXTERNAL),
        'scope': 'NEW supplied-world CPU collection tests; handwritten artificial descriptors only.',
        'old_tests_run': 0, 'scientific_generation': 0, 'models': 0, 'policy_training_updates': 0,
        'private_evaluations': 0, 'remote_operations': 0, 'installs': 0, 'commits': 0,
        'cost_scope': 'Test child CPU/wall only; implementation/inspection/custody unmetered. No allowance reset.'})
    copy(Path(__file__), EXTERNAL/'run_preparation.py')
    copy(EXTERNAL/'preparation.json', LOCAL/'preparation.json')
attempt = EXTERNAL/f'attempt{number:02d}'
attempt.mkdir(exist_ok=False)
source = attempt/'source'
source.mkdir()
pending = ['llm_intervention_training_collection', 'test_llm_intervention_training_collection']
copied = {}
while pending:
    name = pending.pop()
    if name in copied:
        continue
    original = REPO/'scripts/research'/(name+'.py')
    copy(original, source/original.name)
    copied[name] = {'original': str(original), **pin(source/original.name)}
    for node in ast.walk(ast.parse(original.read_text())):
        names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                 [item.name for item in node.names] if isinstance(node, ast.Import) else [])
        pending.extend(n for n in names if n.startswith('llm_intervention_'))
inputs = [REPO/'docs/development/guidance/ace_llm_intervention_training_collection_2026-11-07.md',
    REPO/'docs/development/guidance/ace_llm_intervention_execution_2026-11-02.md',
    REPO/'docs/development/guidance/ace_llm_intervention_screen_protocol_2026-11-07.md',
    REPO/'docs/development/guidance/ace_llm_intervention_training_exposure_2026-11-07.md',
    REPO/'docs/development/guidance/ace_llm_intervention_training_dataset_2026-11-04.md',
    REPO/'results/ace_llm_intervention_screen_protocol_preparation_20261107/test-attempt01/fixtures/prospective-proposal.json']
for original in inputs:
    copy(original, source/original.name)
    copied[original.name] = {'original': str(original), **pin(source/original.name)}
save(attempt/'source_manifest.json', copied)
copy(Path(__file__), attempt/'run_preparation.py')
command = [sys.executable, '-B', str(source/'test_llm_intervention_training_collection.py'), str(attempt/'fixtures'), *sys.argv[2:]]
environment = {'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
invocation = {'command': command, 'cwd': str(attempt), 'environment_overrides': environment,
    'preparation_command': [sys.executable, '-B', str(Path(__file__)), *sys.argv[1:]],
    'timeout_seconds': 60, 'python': sys.version, 'platform': platform.platform(),
    'numpy': importlib.metadata.version('numpy'), 'started_utc': datetime.now(timezone.utc).isoformat()}
save(attempt/'command.json', invocation)
before = resource.getrusage(resource.RUSAGE_CHILDREN)
began = time.monotonic()
with (attempt/'stdout.log').open('xb') as stdout, (attempt/'stderr.log').open('xb') as stderr:
    try:
        completed = subprocess.run(command, cwd=attempt, env=dict(os.environ, **environment),
            stdout=stdout, stderr=stderr, timeout=60, check=False)
        returncode, exception = completed.returncode, None
    except BaseException as exc:
        returncode, exception = None, type(exc).__name__+': '+str(exc)
elapsed = time.monotonic()-began
after = resource.getrusage(resource.RUSAGE_CHILDREN)
receipt = {**invocation, 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'returncode': returncode, 'runner_exception': exception, 'elapsed_seconds': elapsed,
    'child_user_cpu_seconds': after.ru_utime-before.ru_utime,
    'child_system_cpu_seconds': after.ru_stime-before.ru_stime, 'child_max_rss_bytes_macos': after.ru_maxrss}
save(attempt/'receipt.json', receipt)
members = [{'path': str(p.relative_to(attempt)), **pin(p)} for p in sorted(attempt.rglob('*')) if p.is_file()]
save(attempt/'manifest.json', {'members': members, 'count_excluding_manifest': len(members),
    'bytes_excluding_manifest': sum(p['bytes'] for p in members)})
shutil.copytree(attempt, LOCAL/attempt.name)
print(json.dumps(receipt, indent=2))
print((attempt/'stderr.log').read_text())
raise SystemExit(0 if returncode == 0 else 1)

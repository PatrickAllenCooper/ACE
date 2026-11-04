"""Once-only artificial CPU fixture run with exclusive source/output custody."""
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
LOCAL = REPO/'results/ace_llm_intervention_world_preparation_20261104'
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-world-preparation-20261104-01')
attempt = EXTERNAL/'attempt01'
attempt.mkdir(exist_ok=False)
source = attempt/'source'
source.mkdir()

def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())

def pin(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

pending = ['llm_intervention_world', 'test_llm_intervention_world']
copied = {}
while pending:
    module = pending.pop()
    if module in copied:
        continue
    original = REPO/'scripts/research'/(module + '.py')
    target = source/original.name
    with target.open('xb') as handle:
        handle.write(original.read_bytes())
    copied[module] = {'original': str(original), **pin(target)}
    for node in ast.walk(ast.parse(target.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('llm_intervention_'):
            pending.append(node.module)

for relative in ('docs/development/guidance/ace_llm_intervention_world_2026-11-04.md',
                 'docs/development/guidance/ace_llm_intervention_agenda_2026-11-02.md',
                 'docs/development/guidance/ace_llm_intervention_execution_2026-11-02.md',
                 'docs/development/guidance/reviews/ace_llm_intervention_first_screen_design_2026-11-02.md'):
    original = REPO/relative
    with (source/original.name).open('xb') as handle:
        handle.write(original.read_bytes())
    copied[original.name] = {'original': str(original), **pin(source/original.name)}
with (attempt/'run_preparation.py').open('xb') as handle:
    handle.write(Path(__file__).read_bytes())
save(attempt/'source_manifest.json', copied)

command = [sys.executable, '-B', str(source/'test_llm_intervention_world.py'), str(attempt/'fixtures')]
env_changes = {'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
environment = dict(os.environ, **env_changes)
record = {'command': command, 'cwd': str(attempt), 'environment_overrides': env_changes,
          'python': sys.version, 'platform': platform.platform(),
          'numpy_version': importlib.metadata.version('numpy'), 'timeout_seconds': 60,
          'start_utc': datetime.now(timezone.utc).isoformat(),
          'preparation_command': [sys.executable, '-B', str(Path(__file__))],
          'scope': 'new artificial CPU tests only; no existing tests, scientific worlds, models, installs, downloads or remote operations',
          'cost_scope': 'child CPU and elapsed test process only; inspection/implementation/copying partly unmetered; no scheduler allocation or GPU; cumulative allowance not duplicated'}
save(attempt/'command.json', record)
before = resource.getrusage(resource.RUSAGE_CHILDREN)
started = time.monotonic()
with (attempt/'stdout.log').open('xb') as stdout, (attempt/'stderr.log').open('xb') as stderr:
    try:
        result = subprocess.run(command, cwd=attempt, env=environment, stdout=stdout,
                                stderr=stderr, timeout=60, check=False)
        returncode, failure = result.returncode, None
    except BaseException as exc:
        returncode, failure = None, type(exc).__name__ + ': ' + str(exc)
after = resource.getrusage(resource.RUSAGE_CHILDREN)
receipt = {**record, 'end_utc': datetime.now(timezone.utc).isoformat(),
           'elapsed_seconds': time.monotonic()-started, 'returncode': returncode, 'runner_exception': failure,
           'child_user_cpu_seconds': after.ru_utime-before.ru_utime,
           'child_system_cpu_seconds': after.ru_stime-before.ru_stime,
           'child_max_rss_bytes_macos': after.ru_maxrss, 'scientific_descriptors_generated': 0,
           'gpu_seconds': 0, 'remote_operations': 0, 'model_loads': 0,
           'files': {str(p.relative_to(attempt)): pin(p) for p in sorted(attempt.rglob('*')) if p.is_file()}}
save(attempt/'receipt.json', receipt)
shutil.copytree(attempt, LOCAL/'attempt01')
print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}, indent=2))
print((attempt/'stderr.log').read_text())
raise SystemExit(0 if returncode == 0 else 1)

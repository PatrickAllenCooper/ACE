"""One new artificial-only test invocation with exclusive retained custody."""
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
LOCAL = REPO/'results/ace_llm_intervention_cost_curves_preparation_20261107'
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-cost-curves-preparation-20261107-01')


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def pin(path):
    value = path.read_bytes()
    return {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}


EXTERNAL.mkdir(exist_ok=False)
attempt = EXTERNAL/'attempt01'
attempt.mkdir()
source = attempt/'source'
source.mkdir()
save(EXTERNAL/'preparation.json', {
    'started_utc': datetime.now(timezone.utc).isoformat(),
    'local': str(LOCAL), 'external': str(EXTERNAL),
    'scope': 'New pure offline cost-curve primitive; seven new artificial methods only.',
    'read_interfaces': ['first_screen_design', 'world', 'SCM', 'task', 'episode', 'contract'],
    'old_tests_run': 0, 'scientific_generation': 0, 'model_loads': 0,
    'remote_operations': 0, 'installs': 0, 'commits': 0,
    'prior_discovery': 'Read-only no-AGENTS rg returned 1; no prior test invocation.',
    'cost_scope': 'Implementation and inspection CPU unmetered; subprocess CPU below is not whole-cycle CPU.'})
pending = ['llm_intervention_cost_curves', 'test_llm_intervention_cost_curves']
copied = {}
while pending:
    module = pending.pop()
    if module in copied:
        continue
    original = REPO/'scripts/research'/(module+'.py')
    target = source/original.name
    with target.open('xb') as handle:
        handle.write(original.read_bytes())
    copied[module] = {'original': str(original), **pin(target)}
    for node in ast.walk(ast.parse(target.read_text())):
        names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                 [n.name for n in node.names] if isinstance(node, ast.Import) else [])
        pending.extend(n for n in names if n.startswith('llm_intervention_'))
guide = REPO/'docs/development/guidance/ace_llm_intervention_cost_curves_2026-11-07.md'
with (source/guide.name).open('xb') as handle:
    handle.write(guide.read_bytes())
copied[guide.name] = {'original': str(guide), **pin(source/guide.name)}
save(attempt/'source_manifest.json', copied)
shutil.copyfile(Path(__file__), attempt/'run_preparation.py')
command = [sys.executable, '-B', str(source/'test_llm_intervention_cost_curves.py'), str(attempt/'fixtures')]
environment = {'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
record = {'command': command, 'cwd': str(attempt), 'environment_overrides': environment,
          'python': sys.version, 'platform': platform.platform(), 'numpy': importlib.metadata.version('numpy'),
          'preparation_command': [sys.executable, '-B', str(Path(__file__))],
          'timeout_seconds': 60, 'started_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'Only seven new artificial offline methods; no prior tests, responses, generation, models or remote work.'}
save(attempt/'command.json', record)
before = resource.getrusage(resource.RUSAGE_CHILDREN)
started = time.monotonic()
with (attempt/'stdout.log').open('xb') as stdout, (attempt/'stderr.log').open('xb') as stderr:
    try:
        result = subprocess.run(command, cwd=attempt, env=dict(os.environ, **environment),
                                stdout=stdout, stderr=stderr, timeout=60, check=False)
        code, failure = result.returncode, None
    except BaseException as exc:
        code, failure = None, type(exc).__name__+': '+str(exc)
elapsed = time.monotonic()-started
after = resource.getrusage(resource.RUSAGE_CHILDREN)
receipt = {**record, 'finished_utc': datetime.now(timezone.utc).isoformat(),
           'returncode': code, 'runner_exception': failure, 'elapsed_seconds': elapsed,
           'child_user_cpu_seconds': after.ru_utime-before.ru_utime,
           'child_system_cpu_seconds': after.ru_stime-before.ru_stime,
           'child_max_rss_bytes_macos': after.ru_maxrss,
           'files': {str(p.relative_to(attempt)): pin(p) for p in sorted(attempt.rglob('*')) if p.is_file()}}
save(attempt/'receipt.json', receipt)
shutil.copytree(attempt, LOCAL/'attempt01')
shutil.copyfile(EXTERNAL/'preparation.json', LOCAL/'preparation.json')
print(json.dumps({k:v for k,v in receipt.items() if k != 'files'}, indent=2))
print((attempt/'stderr.log').read_text())
raise SystemExit(0 if code == 0 else 1)

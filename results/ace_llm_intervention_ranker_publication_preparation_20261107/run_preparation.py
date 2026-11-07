"""Retain one focused stdlib-only correction check in exclusive custody."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

REPO = Path('/Users/pat/code/ACE')
LOCAL = REPO/'results/ace_llm_intervention_ranker_publication_preparation_20261107'
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-ranker-publication-preparation-20261107-01')


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def pin(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


EXTERNAL.mkdir(exist_ok=False)
source = EXTERNAL/'source'
source.mkdir()
paths = ['scripts/research/llm_intervention_ranker_training.py',
    'scripts/research/llm_intervention_contract.py',
    'scripts/research/test_llm_intervention_ranker_publication.py',
    'docs/development/guidance/ace_llm_intervention_ranker_training_2026-11-07.md',
    'docs/development/guidance/reviews/ace_llm_intervention_ranker_training_review_2026-11-07.md']
sources = {}
for name in paths:
    path = REPO/name
    with (source/path.name).open('xb') as f:
        f.write(path.read_bytes())
    sources[name] = pin(path)
save(EXTERNAL/'sources.json', sources)
shutil.copyfile(__file__, EXTERNAL/'run_preparation.py')
command = [sys.executable, '-B', str(source/'test_llm_intervention_ranker_publication.py')]
env = {'PYTHONDONTWRITEBYTECODE': '1', 'ACE_NUMERIC_PUBLICATION_TEST_OUTPUT': str(EXTERNAL/'fixtures'),
    'ACE_NUMERIC_PUBLICATION_SOURCE': str(source/'llm_intervention_ranker_training.py'),
    'ACE_NUMERIC_PUBLICATION_CONTRACT': str(source/'llm_intervention_contract.py')}
metadata = {'command': command, 'cwd': '/tmp', 'environment_overrides': env,
    'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
    'scope': 'Five new terminal-publication helper controls. AST extraction only, no Torch or trainer import.',
    'prior_preparation_manifest_sha256': pin(REPO/'results/ace_llm_intervention_ranker_training_preparation_20261107/manifest.json')['sha256'],
    'old_tests_repeated': 0, 'models': 0, 'optimizer_updates': 0, 'scientific_responses': 0}
save(EXTERNAL/'command.json', metadata)
before = resource.getrusage(resource.RUSAGE_CHILDREN)
start = time.monotonic()
with (EXTERNAL/'stdout.log').open('xb') as out, (EXTERNAL/'stderr.log').open('xb') as err:
    try:
        result = subprocess.run(command, cwd='/tmp', env=dict(os.environ, **env), stdout=out,
                                stderr=err, timeout=30, check=False)
        returncode, exception = result.returncode, None
    except BaseException as exc:
        returncode, exception = None, type(exc).__name__+': '+str(exc)
elapsed = time.monotonic()-start
after = resource.getrusage(resource.RUSAGE_CHILDREN)
receipt = {**metadata, 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'returncode': returncode, 'exception': exception, 'elapsed_seconds': elapsed,
    'child_cpu_seconds': after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
    'peak_child_rss_bytes_macos': after.ru_maxrss,
    'cost_scope': 'One focused child including imports. Preparation/review overhead unmetered and no allowance reset.'}
save(EXTERNAL/'receipt.json', receipt)
members = [{'path': str(p.relative_to(EXTERNAL)), **pin(p)} for p in sorted(EXTERNAL.rglob('*')) if p.is_file()]
save(EXTERNAL/'manifest.json', {'member_count_excluding_manifest': len(members), 'members': members})
for path in EXTERNAL.iterdir():
    target = LOCAL/path.name
    if path.is_dir():
        shutil.copytree(path, target)
    elif path.name == 'run_preparation.py':
        assert path.read_bytes() == target.read_bytes()
    else:
        with target.open('xb') as f:
            f.write(path.read_bytes())
print(json.dumps(receipt, indent=2))
print((EXTERNAL/'stderr.log').read_text())
raise SystemExit(0 if returncode == 0 else 1)

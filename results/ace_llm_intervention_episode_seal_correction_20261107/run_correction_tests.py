"""Single invocation of only the four new posterior-correction methods."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/pat/code/ACE')
SOURCES = ('scripts/research/llm_intervention_episode_seal.py',
    'scripts/research/test_llm_intervention_episode_seal.py',
    'docs/development/guidance/ace_llm_intervention_episode_seal_2026-11-07.md',
    'scripts/research/llm_intervention_contract.py', 'scripts/research/llm_intervention_world.py',
    'scripts/research/llm_intervention_scm.py',
    'docs/development/guidance/reviews/ace_llm_intervention_episode_seal_review_2026-11-07.md')
attempt = ROOT/'test-attempt01'
attempt.mkdir(exist_ok=False)
pins = {}
for relative in SOURCES:
    dest = ROOT/'source02'/relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO/relative, dest)
    raw = dest.read_bytes()
    pins[relative] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
env = dict(os.environ, ACE_EPISODE_SEAL_TEST_OUTPUT=str(attempt/'fixtures'))
methods = ['test_positive_hand_oracle_and_pinned_prior_noise', 'test_changed_mean_entries_are_rejected',
    'test_changed_covariance_entries_are_rejected', 'test_atomic_prefixes_and_explicit_unavailable_failed_state']
command = [sys.executable, '-B', str(REPO/SOURCES[1]), '-v']+['PosteriorCorrectionTests.'+m for m in methods]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
before, wall = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
with (attempt/'stdout.txt').open('xb') as out, (attempt/'stderr.txt').open('xb') as err:
    completed = subprocess.run(command, cwd='/tmp', env=env, stdout=out, stderr=err)
elapsed, after = time.perf_counter()-wall, resource.getrusage(resource.RUSAGE_CHILDREN)
receipt = {'command_argv': command, 'command_shell': shlex.join(command), 'cwd': '/tmp',
    'environment_override': {'ACE_EPISODE_SEAL_TEST_OUTPUT': env['ACE_EPISODE_SEAL_TEST_OUTPUT']},
    'interpreter': sys.executable, 'interpreter_version': sys.version, 'started_utc': started,
    'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'exit_code': completed.returncode,
    'elapsed_seconds': elapsed, 'child_user_cpu_seconds': after.ru_utime-before.ru_utime,
    'child_system_cpu_seconds': after.ru_stime-before.ru_stime,
    'child_total_cpu_seconds': after.ru_utime-before.ru_utime+after.ru_stime-before.ru_stime,
    'source': pins, 'scope': 'four new correction methods only; original six methods and parent integration not run',
    'read_only_context': ['scripts/research/llm_intervention_scm.py', SOURCES[-1]],
    'original_capsule_manifest_sha256': 'aa049adf744ca164902ac4777ce6f0cb94e30700e65f50d99a9ead6f3294e05f'}
(attempt/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
print((attempt/'stderr.txt').read_text())
print(json.dumps(receipt, indent=2, sort_keys=True))
raise SystemExit(completed.returncode)

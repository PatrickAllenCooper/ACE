"""Single-use receipt wrapper for ONLY this sidecar's new artificial suite."""
import datetime
import hashlib
import json
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import time
import os

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/pat/code/ACE')
SOURCE = ('scripts/research/llm_intervention_screen_reporting.py',
          'scripts/research/test_llm_intervention_screen_reporting.py',
          'docs/development/guidance/ace_llm_intervention_screen_reporting_2026-11-07.md')
attempt = ROOT/'test-attempt01'
attempt.mkdir(exist_ok=False)
pins = {}
for relative in SOURCE:
    dest = ROOT/'source01'/relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO/relative, dest)
    raw = dest.read_bytes()
    pins[relative] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
env = dict(os.environ, ACE_SCREEN_REPORTING_TEST_OUTPUT=str(attempt/'fixtures'))
command = [sys.executable, '-B', str(REPO/SOURCE[1]), '-v']
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
before = resource.getrusage(resource.RUSAGE_CHILDREN)
wall = time.perf_counter()
with (attempt/'stdout.txt').open('xb') as out, (attempt/'stderr.txt').open('xb') as err:
    completed = subprocess.run(command, cwd='/tmp', env=env, stdout=out, stderr=err)
elapsed = time.perf_counter()-wall
after = resource.getrusage(resource.RUSAGE_CHILDREN)
receipt = {'command_argv': command, 'command_shell': shlex.join(command), 'cwd': '/tmp',
    'environment_override': {'ACE_SCREEN_REPORTING_TEST_OUTPUT': env['ACE_SCREEN_REPORTING_TEST_OUTPUT']},
    'interpreter': sys.executable, 'interpreter_version': sys.version, 'started_utc': started,
    'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'exit_code': completed.returncode,
    'elapsed_seconds': elapsed, 'child_user_cpu_seconds': after.ru_utime-before.ru_utime,
    'child_system_cpu_seconds': after.ru_stime-before.ru_stime,
    'child_total_cpu_seconds': after.ru_utime-before.ru_utime+after.ru_stime-before.ru_stime,
    'source': pins, 'scope': 'new artificial hand-authored report tests only; no upstream reporter execution'}
(attempt/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
print((attempt/'stderr.txt').read_text())
print(json.dumps(receipt, indent=2, sort_keys=True))
raise SystemExit(completed.returncode)

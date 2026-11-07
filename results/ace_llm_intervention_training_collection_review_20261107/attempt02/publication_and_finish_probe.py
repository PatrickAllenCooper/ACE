"""One-shot AST-only fault probes. No collector/module/model imports or responses.

Runs the collector's unchanged final inventory/publication statements and exception
handler with empty synthetic record/world lists. Separately runs its exact finish
body on a truncated plan. This is not a full collect_training_batch invocation.
"""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch


CAPSULE = Path('/Users/pat/code/ACE/results/ace_llm_intervention_training_collection_preparation_20261107/attempt02/source')
ROOT = Path(__file__).resolve().parent
STARTED = datetime.now(timezone.utc).isoformat()
WALL, CPU = time.monotonic(), time.process_time()
SOURCE_PATHS = (
    'scripts/research/llm_intervention_training_collection.py',
    'scripts/research/test_llm_intervention_training_collection.py',
    'docs/development/guidance/ace_llm_intervention_training_collection_2026-11-07.md',
    'scripts/research/llm_intervention_training_dataset.py',
    'scripts/research/llm_intervention_screen_protocol.py',
    'scripts/research/llm_intervention_training_exposure.py',
    'scripts/research/llm_intervention_replicated_branches.py',
    'scripts/research/llm_intervention_contract.py',
)
EXPECTED = {
    SOURCE_PATHS[0]: 'b70367591f5021fd59d9c58ff75f8269b3f8f6301d2a86c2d2238478a8cadc40',
    SOURCE_PATHS[1]: 'f85420d8586b08da1b73b6199c5abf6583467de6df858d513d3ae034ee193ef1',
    SOURCE_PATHS[2]: 'ed19961a163176fdc0ebb6e2094127b72b1c16c33a57af42f9d943ec2042317a',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, obj):
    with path.open('x') as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write('\n')


sources = {}
(ROOT/'source').mkdir(exist_ok=False)
for name in SOURCE_PATHS:
    raw = (CAPSULE/Path(name).name).read_bytes()
    if name in EXPECTED:
        assert sha(raw) == EXPECTED[name], 'Review target changed: '+name
    sources[name] = {'sha256': sha(raw), 'bytes': len(raw)}
    with (ROOT/'source'/Path(name).name).open('xb') as handle:
        handle.write(raw)
save(ROOT/'source_manifest.json', sources)


def tree(name):
    return ast.parse((ROOT/'source'/name).read_text())


def function(module, name):
    return copy.deepcopy(next(node for node in module.body
                              if isinstance(node, ast.FunctionDef) and node.name == name))


collector_tree = tree('llm_intervention_training_collection.py')
collector = function(collector_tree, 'collect_training_batch')
finish_node = function(collector, 'finish')
try_node = next(node for node in collector.body if isinstance(node, ast.Try))
# Source lines 281-293: final inventory and publication, followed by the exact
# original exception handler (294-304). No preflight or collection is executed.
tail = copy.deepcopy([node for node in try_node.body if node.lineno >= 281])
assert tail[0].lineno == 281 and tail[-1].lineno == 293
wrapper = ast.parse('def publication_path():\n    pass\n').body[0]
wrapper.body = [ast.Try(body=tail, handlers=copy.deepcopy(try_node.handlers), orelse=[], finalbody=[])]
env = dict(json=json, hashlib=hashlib, os=os, datetime=datetime, timezone=timezone, time=time)
nodes = [function(tree('llm_intervention_contract.py'), 'canonical'),
         function(tree('llm_intervention_replicated_branches.py'), '_save'),
         function(collector_tree, '_require'), function(collector_tree, '_pin'),
         function(collector_tree, '_inventory'), finish_node, wrapper]
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
             '<exact collector publication/finish AST>', 'exec'), env)
env.update(VERSION='ace-supplied-training-collection-v1', scope='artificial',
           world_roster=(), entries=[], planned=[], worlds={}, active=None,
           stage='inventory', started=time.monotonic(), cpu=time.process_time(),
           started_utc=STARTED)

terminal_dir = ROOT/'terminal-fsync'
terminal_dir.mkdir(exist_ok=False)
save(terminal_dir/'plan.json', {'probe_only': True})
env['output'] = terminal_dir
original_save, real_fsync = env['_save'], os.fsync
injected = []


def terminal_fault_save(path, value):
    if path.name == 'terminal.json':
        def fault(fd):
            injected.append('terminal fsync raises after _save write+flush')
            raise OSError('review-only terminal fsync failure')
        with patch.object(os, 'fsync', fault):
            return original_save(path, value)
    return original_save(path, value)


env['_save'] = terminal_fault_save
try:
    env['publication_path']()
except OSError as exc:
    publication_exception = str(exc)
else:
    raise AssertionError('Expected injected OSError')
terminal = json.loads((terminal_dir/'terminal.json').read_text())
failure = json.loads((terminal_dir/'failure.json').read_text())
assert terminal['status'] == 'complete' and terminal['exposure_ready'] is True
assert failure['status'] == 'failed' and failure['exposure_ready'] is False
assert sha((terminal_dir/'inventory.json').read_bytes()) == terminal['inventory_sha256']

bad_plan_dir = ROOT/'unreadable-plan'
(bad_plan_dir/'world-000').mkdir(parents=True, exist_ok=False)
with (bad_plan_dir/'world-000/plan.json').open('x') as handle:
    handle.write('{"journal_plan":')
env.update(output=bad_plan_dir, entries=[{'world_id': 17, 'path': 'world-000', 'status': 'started'}],
           world_roster=(17,), active=0, stage='collection', _save=original_save)
# Enter the exact except handler through an injected original assembly error.
handler_wrapper = ast.parse('def failure_path():\n    pass\n').body[0]
handler_wrapper.body = [ast.Try(body=ast.parse("raise OSError('original plan write failure')").body,
    handlers=copy.deepcopy(try_node.handlers), orelse=[], finalbody=[])]
exec(compile(ast.fix_missing_locations(ast.Module(body=[handler_wrapper], type_ignores=[])),
             '<exact collector exception handler AST>', 'exec'), env)
try:
    env['failure_path']()
except json.JSONDecodeError as exc:
    finish_error = {'type': type(exc).__name__, 'message': str(exc),
                    'original_context_type': type(exc.__context__).__name__,
                    'original_context_message': str(exc.__context__)}
else:
    raise AssertionError('Expected unhandled plan decode error')
assert not (bad_plan_dir/'failure.json').exists()
assert all(sha((CAPSULE/Path(name).name).read_bytes()) == info['sha256'] for name, info in sources.items())
result = {'scope': 'isolated unchanged AST bodies; not a full batch execution',
    'scientific_worlds': 0, 'world_constructors': 0, 'responses': 0, 'model_imports': 0,
    'training': 0, 'gpu': 0, 'remote': 0, 'old_tests': 0,
    'publication': {'injected': injected, 'raised': publication_exception,
        'terminal_status': terminal['status'], 'terminal_exposure_ready': terminal['exposure_ready'],
        'failure_status': failure['status'], 'failure_exposure_ready': failure['exposure_ready'],
        'terminal_inventory_hash_valid': True},
    'unreadable_plan': {'error': finish_error, 'batch_failure_exists': False,
        'original_plan_evidence_preserved': True},
    'source_stable_through_probe': True, 'started_utc': STARTED,
    'finished_utc': datetime.now(timezone.utc).isoformat(),
    'probe_body_cpu_seconds': time.process_time()-CPU,
    'probe_body_wall_seconds': time.monotonic()-WALL,
    'cost_scope': 'Probe body incl AST, copying and local I/O; excludes Python startup/imports, review and custody.'}
save(ROOT/'probe_result.json', result)
members = {str(p.relative_to(ROOT)): {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size}
           for p in sorted(ROOT.rglob('*')) if p.is_file()}
save(ROOT/'probe_manifest.json', members)
print(json.dumps(result, indent=2, sort_keys=True))

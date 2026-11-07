"""One affected saved-finalization recheck; never rerun episodes or policies.

The driver authenticates the old capsule and freezes the source closure before
starting one measured child. The child preserves all eight original finalizers,
including negative cases, and requires all 17 roster rows to remain byte-identical.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
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
import traceback
from unittest.mock import patch


REPO = Path('/Users/pat/code/ACE')
LOCAL = REPO/'results/ace_llm_intervention_paired_episode_recheck_20261107'
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-paired-episode-recheck-20261107-01')
ORIGINAL = EXTERNAL.parent/'ACE-LLM-intervention-paired-episode-preparation-20261107-01'
ORIGINAL_LOCAL = REPO/'results/ace_llm_intervention_paired_episode_preparation_20261107'
ORIGINAL_MANIFEST = '6878bb3ea550a234658cef89b8c29fe5c96aafa2fc91ad002fb7a642d1100576'
EXPECTED_SEALER = '91b7a40829157cc21d586fd22ae17eee6556d1b73da1dfbf2f4b104564f39811'
READY = REPO/'results/ace_llm_intervention_episode_seal_correction_20261107/test-attempt01/receipt.json'
READY_PIN = '3f72b0c00a23cc96c0fb46ae7610ba24c56bc92adafd8ce61da6fb9cf0710a6b'
CASES = (
    'test_actual_persistent_cleanup_required_before_private_reporting/report',
    'test_actual_persistent_cleanup_required_before_private_reporting/unclean-service',
    'test_actual_public_processes_to_hand_computed_private_costs/report',
    'test_conflicting_initial_histories_never_select_an_authoritative_arm/report',
    'test_conflicting_initial_histories_never_select_an_authoritative_arm/tampered-report',
    'test_invalid_return_and_partial_initial_failure_are_retained/report',
    'test_no_affordable_request_is_a_clean_prior_only_budget_stop/report',
    'test_prepared_outcome_is_not_counted_as_returned/report',
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(path, raw):
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def save(path, value):
    write(path, (canonical(value)+'\n').encode())


def verify_original(root):
    raw = (root/'manifest.json').read_bytes()
    if sha(raw) != ORIGINAL_MANIFEST:
        raise ValueError('original independent manifest pin differs')
    members = json.loads(raw)['members']
    expected = {m['path'] for m in members}|{'manifest.json'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != expected or len(actual) != 621:
        raise ValueError('original complete membership differs')
    size = 0
    for m in members:
        p = root/m['path']
        b = p.read_bytes()
        if p.is_symlink() or len(b) != m['bytes'] or sha(b) != m['sha256']:
            raise ValueError('original member differs: '+m['path'])
        size += len(b)
    size += len(raw)
    if size != 1148353:
        raise ValueError('original byte count differs')
    return {'root': str(root), 'files': len(actual), 'bytes': size,
            'manifest_sha256': sha(raw), 'all_member_hashes_verified': True}


def closure():
    pending, sources = ['llm_intervention_paired_episode'], {}
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        path = REPO/'scripts/research'/(name+'.py')
        raw = path.read_bytes()
        sources[name+'.py'] = raw
        for node in ast.walk(ast.parse(raw)):
            names = ([node.module] if isinstance(node, ast.ImportFrom) else
                     [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            pending.extend(n for n in names if n and n.startswith('llm_intervention_'))
    return sources


def tree_pins(root):
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
            for p in sorted(root.rglob('*')) if p.is_file()}


def differences(old, new, prefix=''):
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys()|new.keys()):
            p = prefix+'/'+key
            if key not in old:
                result.append({'field': p, 'change': 'added', 'new': new[key]})
            elif key not in new:
                result.append({'field': p, 'change': 'removed', 'old': old[key]})
            else:
                result.extend(differences(old[key], new[key], p))
        return result
    return [] if old == new else [{'field': prefix, 'old': old, 'new': new}]


def worker(root):
    """Run only frozen finalizers; no subprocess/response/model work is allowed."""
    unauthorized = []
    def reject(label):
        def blocked(*args, **kwargs):
            unauthorized.append(label)
            raise AssertionError('unauthorized recheck operation: '+label)
        return blocked
    protected = (str(ORIGINAL), str(ORIGINAL_LOCAL))
    def audit(event, args):
        if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.fork', 'socket.connect'):
            reject(event)()
        if event == 'import' and (args[0].split('.')[0] in ('torch', 'transformers') or args[0].startswith('test_llm_')):
            reject('import '+args[0])()
        if event == 'open' and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0])
            if any(name == p or name.startswith(p+'/') for p in protected):
                flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
                mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
                if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND) or any(c in mode for c in 'wax+'):
                    reject('write original '+name)()
    sys.addaudithook(audit)
    source_pins = json.loads((root/'source_manifest.json').read_bytes())
    for name, pin in source_pins.items():
        if sha((root/'source'/name).read_bytes()) != pin['sha256']:
            raise ValueError('frozen source changed before import: '+name)
    sys.path.insert(0, str(root/'source'))
    import llm_intervention_paired_episode as paired
    import llm_intervention_world as world
    import llm_intervention_episode as episode
    from llm_intervention_task import TaskProcessPolicy
    from llm_intervention_persistent_policy import PersistentPolicy

    original_members = {m['path']: m for m in json.loads((ORIGINAL/'manifest.json').read_bytes())['members']}
    def original_bytes(path):
        rel = str(path.relative_to(ORIGINAL))
        raw = path.read_bytes()
        if rel not in original_members or sha(raw) != original_members[rel]['sha256']:
            raise ValueError('input not in authenticated original manifest: '+rel)
        return raw
    outcomes, rows = [], 0
    with ExitStack() as stack:
        for obj, name, label in (
            (paired, 'ResponseAdapter', 'paired.ResponseAdapter'),
            (world, 'ResponseAdapter', 'world.ResponseAdapter'),
            (world, 'generate_world', 'world.generate_world'),
            (world, '_normal', 'world._normal'),
            (world, 'training_callbacks', 'world.training_callbacks'),
            (TaskProcessPolicy, 'choose', 'TaskProcessPolicy.choose'),
            (TaskProcessPolicy, '__init__', 'TaskProcessPolicy.__init__'),
            (PersistentPolicy, 'choose', 'PersistentPolicy.choose'),
            (PersistentPolicy, '__init__', 'PersistentPolicy.__init__'),
            (paired, 'run_world_arm', 'paired.run_world_arm'),
            (paired, 'run_episode', 'paired.run_episode'),
            (episode, 'run_episode', 'episode.run_episode'),
        ):
            stack.enter_context(patch.object(obj, name, reject(label)))
        for index, name in enumerate(CASES):
            old = ORIGINAL/'fixtures'/name
            case = root/'cases'/f'{index:02d}'
            case.mkdir(parents=True, exist_ok=False)
            fresh = case/'finalization'
            record = {'index': index, 'original_directory': str(old), 'started_utc': utc()}
            try:
                plan_raw = original_bytes(old/'plan.json')
                arm_raw = original_bytes(old/'arm-inventory.json')
                previous = original_bytes(old/'summary.json')
                plan, arms = json.loads(plan_raw), json.loads(arm_raw)
                for item in arms.values():
                    if item['path'] is not None:
                        Path(item['path']).resolve().relative_to(ORIGINAL.resolve())
                closed, input_pins = {}, {'plan.json': sha(plan_raw), 'arm-inventory.json': sha(arm_raw), 'summary.json': sha(previous)}
                for j, spec in enumerate(plan['policy_specs']):
                    if spec['kind'] == 'persistent':
                        filename = f'service-{j:02d}.closed.json'
                        raw = original_bytes(old/filename)
                        closed[spec['policy_id']] = {'raw': raw, 'sha256': sha(raw)}
                        input_pins[filename] = sha(raw)
                save(case/'inputs.json', {'original_directory': str(old), 'byte_pins': input_pins,
                    'plan_sha256': sha(canonical(plan).encode()), 'arm_outputs': arms,
                    'persistent_closed_pins': {k: v['sha256'] for k, v in closed.items()}})
                write(case/'original-summary.json', previous)
                result = paired.finalize_paired_world(plan=plan, plan_sha256=sha(canonical(plan).encode()),
                    arm_outputs=arms, persistent_closed=closed, output=fresh)
                observed = (fresh/'summary.json').read_bytes()
                row_count = len(result['rows'])
                rows += row_count
                old_files, new_files = tree_pins(old), tree_pins(fresh)
                file_delta = {n: {'old': old_files.get(n), 'new': new_files.get(n)}
                              for n in sorted(old_files.keys()|new_files.keys()) if old_files.get(n) != new_files.get(n)}
                provenance_delta = {}
                for path in sorted(set(old.glob('arm-*/sealing-provenance.json'))):
                    rel = path.relative_to(old)
                    raw = original_bytes(path)
                    baseline = case/'original-provenance'/rel
                    baseline.parent.mkdir(parents=True, exist_ok=True)
                    write(baseline, raw)
                    target = fresh/rel
                    provenance_delta[str(rel)] = differences(json.loads(raw), json.loads(target.read_bytes())) if target.exists() else {'new_missing': True}
                old_errors = original_bytes(old/'sealing-errors.json')
                write(case/'original-sealing-errors.json', old_errors)
                new_errors = (fresh/'sealing-errors.json').read_bytes()
                save(case/'comparison.json', {
                    'summary_byte_identical': previous == observed,
                    'old_summary_sha256': sha(previous), 'new_summary_sha256': sha(observed),
                    'row_count': row_count, 'file_differences': file_delta,
                    'provenance_differences': provenance_delta,
                    'rejection_byte_identical': old_errors == new_errors,
                    'rejection_differences': differences(json.loads(old_errors), json.loads(new_errors)),
                })
                record.update(status='passed' if previous == observed else 'summary_mismatch',
                              summary_byte_identical=previous == observed, row_count=row_count)
            except BaseException as exc:
                record.update(status='failed', exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            record['finished_utc'] = utc()
            save(case/'outcome.json', record)
            outcomes.append(record)
            print(canonical({k: record[k] for k in ('index', 'status')}), flush=True)
    loaded = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
              if name.startswith('llm_intervention_') and getattr(module, '__file__', None)}
    if any(Path(path).parent != (root/'source').resolve() for path in loaded.values()):
        raise AssertionError('project import escaped the frozen source closure')
    if any(name.split('.')[0] in ('torch', 'transformers') for name in sys.modules):
        raise AssertionError('model runtime imported')
    passed = len(outcomes) == 8 and rows == 17 and not unauthorized and all(x['status'] == 'passed' for x in outcomes)
    save(root/'worker-result.json', {'status': 'passed' if passed else 'failed', 'finalizations': len(outcomes),
        'roster_rows': rows, 'all_summaries_byte_identical': all(x.get('summary_byte_identical') is True for x in outcomes),
        'outcomes': outcomes, 'blocked_operation_attempts': unauthorized, 'loaded_project_modules': loaded,
        'environment_calls': 0, 'policy_calls': 0, 'models_loaded': 0, 'scientific_data_generated': 0,
        'independent_sealer_review_still_required': True})
    return 0 if passed else 1


def driver():
    beginning, cpu_start = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    # Authenticate the original supplied manifest BEFORE reading its inventory.
    initial_checks = [verify_original(ORIGINAL), verify_original(ORIGINAL_LOCAL)]
    ready = READY.read_bytes()
    if sha(ready) != READY_PIN or json.loads(ready)['exit_code'] != 0:
        raise ValueError('ready correction receipt differs')
    sources = closure()
    if sha(sources['llm_intervention_episode_seal.py']) != EXPECTED_SEALER:
        raise ValueError('Mencius ready sealer pin differs')
    original_sources = {p.name for p in (ORIGINAL/'source').glob('llm_intervention_*.py')}
    if set(sources) != original_sources:
        raise ValueError('unexpected project dependency membership change')
    for name, raw in sources.items():
        if name != 'llm_intervention_episode_seal.py' and raw != (ORIGINAL/'source'/name).read_bytes():
            raise ValueError('unexpected non-sealer source change: '+name)
    if {p.name for p in LOCAL.iterdir()} != {'recheck_finalizations.py'}:
        raise ValueError('local output must contain only this unexecuted script')
    EXTERNAL.mkdir(exist_ok=False)
    (EXTERNAL/'source').mkdir()
    write(EXTERNAL/'recheck_finalizations.py', Path(__file__).read_bytes())
    write(EXTERNAL/'original-manifest.json', (ORIGINAL/'manifest.json').read_bytes())
    write(EXTERNAL/'sealer-ready-receipt.json', ready)
    save(EXTERNAL/'original-verification-before.json', initial_checks)
    source_pins = {}
    for name, raw in sorted(sources.items()):
        write(EXTERNAL/'source'/name, raw)
        source_pins[name] = {'bytes': len(raw), 'sha256': sha(raw),
                             'original_sha256': sha((ORIGINAL/'source'/name).read_bytes())}
    save(EXTERNAL/'source_manifest.json', source_pins)
    save(EXTERNAL/'readiness.json', {'corrected_sealer_sha256': EXPECTED_SEALER,
        'handoff': 'Mencius ready message received in this chat; four new correction controls passed; source frozen',
        'ready_receipt_path': str(READY), 'ready_receipt_sha256': READY_PIN,
        'original_manifest_sha256': ORIGINAL_MANIFEST, 'expected_finalizations': 8, 'expected_roster_rows': 17,
        'no_prior_suite_environment_policy_or_model_reruns': True})
    command = [str(REPO/'.venv_test/bin/python'), '-B', str(EXTERNAL/'recheck_finalizations.py'), '--worker', str(EXTERNAL)]
    overrides = {'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': '', 'CUDA_VISIBLE_DEVICES': '',
                 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
    save(EXTERNAL/'command.json', {'command': command, 'cwd': '/tmp', 'environment_overrides': overrides,
                                  'timeout_seconds': 60, 'source_manifest_sha256': sha((EXTERNAL/'source_manifest.json').read_bytes())})
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start, started_utc = time.monotonic(), utc()
    exception, returncode = None, None
    with (EXTERNAL/'stdout.log').open('xb') as out, (EXTERNAL/'stderr.log').open('xb') as err:
        try:
            result = subprocess.run(command, cwd='/tmp', env={**os.environ, **overrides},
                stdout=out, stderr=err, timeout=60, check=False, start_new_session=True)
            returncode = result.returncode
        except BaseException as exc:
            exception = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    save(EXTERNAL/'receipt.json', {'command': command, 'cwd': '/tmp', 'started_utc': started_utc, 'finished_utc': utc(),
        'returncode': returncode, 'exception': exception, 'elapsed_seconds': time.monotonic()-start,
        'child_user_cpu_seconds': after.ru_utime-before.ru_utime,
        'child_system_cpu_seconds': after.ru_stime-before.ru_stime,
        'child_cpu_seconds': after.ru_utime-before.ru_utime+after.ru_stime-before.ru_stime,
        'peak_child_rss_bytes_macos': after.ru_maxrss,
        'scope': 'One affected eight-finalization saved-input child, including imports; no environment/policy/model calls',
        'cost_scope': 'Child CPU/wall and child RSS, not whole development cost or a renewed preparation allowance'})
    integrity_error = None
    try:
        save(EXTERNAL/'original-verification-after.json', [verify_original(ORIGINAL), verify_original(ORIGINAL_LOCAL)])
        for name, pin in source_pins.items():
            if sha((EXTERNAL/'source'/name).read_bytes()) != pin['sha256']:
                raise ValueError('frozen source changed: '+name)
    except BaseException as exc:
        integrity_error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    result_path = EXTERNAL/'worker-result.json'
    report = json.loads(result_path.read_bytes()) if result_path.exists() else None
    okay = returncode == 0 and report and report['status'] == 'passed' and integrity_error is None
    cpu_end = resource.getrusage(resource.RUSAGE_SELF)
    save(EXTERNAL/'closeout.json', {'status': 'affected_finalizations_passed_sealer_review_pending' if okay else 'failed_preserved_no_retry',
        'returncode': returncode, 'integrity_error': integrity_error, 'execution_invocations': 1,
        'worker_result_sha256': sha(result_path.read_bytes()) if result_path.exists() else None,
        'source_manifest_sha256': sha((EXTERNAL/'source_manifest.json').read_bytes()),
        'original_manifest_sha256': ORIGINAL_MANIFEST, 'independent_sealer_review_still_required': True,
        'scientific_acceptance': False, 'scientific_provenance_qualified': False,
        'driver_cpu_seconds_before_manifest': cpu_end.ru_utime+cpu_end.ru_stime-cpu_start.ru_utime-cpu_start.ru_stime,
        'driver_elapsed_seconds_before_manifest': time.monotonic()-beginning,
        'driver_measurement_scope': 'Self CPU only; child cost separately in receipt; excludes startup/imports and final manifest/mirroring'})
    pins = tree_pins(EXTERNAL)
    save(EXTERNAL/'manifest.json', {'member_count_excluding_manifest': len(pins),
        'member_bytes_excluding_manifest': sum(p['bytes'] for p in pins.values()),
        'members': [{'path': name, **pin} for name, pin in pins.items()]})
    for path in sorted(EXTERNAL.rglob('*')):
        if path.is_file():
            target = LOCAL/path.relative_to(EXTERNAL)
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = path.read_bytes()
            if target.exists():
                if target.read_bytes() != raw:
                    raise ValueError('exclusive local mirror collision: '+str(target))
            else:
                write(target, raw)
    if tree_pins(LOCAL) != tree_pins(EXTERNAL):
        raise ValueError('new capsule mirror differs')
    print(canonical({'status': 'passed' if okay else 'failed', 'files': len(tree_pins(EXTERNAL)),
        'bytes': sum(p['bytes'] for p in tree_pins(EXTERNAL).values()),
        'manifest_sha256': sha((EXTERNAL/'manifest.json').read_bytes()),
        'external': str(EXTERNAL), 'local': str(LOCAL)}))
    return 0 if okay else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.worker))
    raise SystemExit(driver())

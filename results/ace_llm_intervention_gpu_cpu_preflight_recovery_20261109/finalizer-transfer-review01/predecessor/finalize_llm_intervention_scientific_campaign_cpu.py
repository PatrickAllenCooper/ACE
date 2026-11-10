"""Once-only CPU reporting after independent scientific development CPU/GPU acceptance.

No submission, training, policy execution, model loading or new responses. The
packet captures the accepted GPU input roster plus this entry and the independently
produced GPU acceptance. Hash inventories before decoding the private bundle.
See GPU_ACCEPTANCE_KEYS for the deliberately data-only upstream admission record.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import time
import traceback
import types

VERSION = 'ace-scientific-campaign-finalizer-phase-clock-v3'
ENTRY = 'finalize_llm_intervention_scientific_campaign_cpu.py'
WORLD_IDS = tuple(range(202611070100, 202611070124))
SCIENTIFIC_CPU_ENTRY = 'prepare_llm_intervention_scientific_campaign_cpu.py'
SCIENTIFIC_GPU_ENTRY = 'qualify_llm_intervention_scientific_campaign_gpu.py'
CPU_ENTRY = 'prepare_llm_intervention_campaign_cpu.py'
GPU_ENTRY = 'qualify_llm_intervention_campaign_gpu.py'
CPU_SHA = '850ca3d4d7ac4799f68d74bd2d0b1196d10809751a4bdf878633122fc6a9c2c4'
GPU_SHA = '58f64d3eea63ae56cc7d14d9188c5d1f9e6e4126fc5b4512abc9758006a803bc'
RUNTIME_SHA = '117dc22bd56889906612bff0c975095918e062974fa537757b3e93c981dbfbc0'
CAMPAIGN_SHA = '9a2717dd68c37ddde6d7fb63c6cc519c067f7b1d457d4f49f919c18342424bbe'
POLICIES = ('uniform_random', 'menu_coverage', 'local_information', 'target_risk',
            'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm')
RESOURCES = dict(account='ucb736_asc2', partition='acpu', qos='cpu-normal', cpus=2,
                 memory_mib=7680, wall_seconds=4200, gpus=0, requeue=False)
PREPARATION, REPORT_STARTUP, REPORT_STARTUP_CAP, WORK, CLEANUP, HARD = 1500, 120, 1620, 3320, 3350, 4190
DEADLINES = dict(preparation=PREPARATION, report_startup=REPORT_STARTUP,
    report_startup_cap=REPORT_STARTUP_CAP, work=WORK, cleanup=CLEANUP, hard=HARD)
PREFLIGHT = 'llm_intervention_cpu_preflight_receipt.py'
HISTORY = 'llm_intervention_cpu_history_admission.py'
RECOVERY_FILES = {'authority': 'gpu-recovery-authority.json', 'context': 'cpu-history-expected.json',
    'receipt': 'cpu-history-preflight.json', 'read_set': 'cpu-history-readset.json'}
RECOVERY_ADDITIONS = {'gpu-recovery-acceptance.json', 'gpu-success-baseline.json',
    'gpu-failure-acceptance.json', 'gpu-failure-baseline.json', 'finalizer-clock-adoption.json'}
RECOVERY_AUTHORITY_SHA = 'b3630d400649ecd186e35232b24e9020e5bced7ad616379c7eaf5eef84498603'
GPU_CONTROL_ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02'
GPU_ORIGINAL_ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'
HISTORICAL_MARKER = 'controller/historical-postcustody-required.json'

GPU_ACCEPTANCE_KEYS = {'schema', 'status', 'scope', 'gpu_packet_sha256', 'cpu_packet_sha256',
    'cpu_acceptance_sha256', 'job_id', 'allocation', 'gpu_output', 'campaign_root',
    'gpu_output_files', 'campaign_files',
    'bundle_sha256', 'attempt_pins', 'accepted_checks', 'terminal_pins', 'scientific_qualified'}
PACKET_KEYS = {'schema', 'scope', 'files', 'source_names', 'paths', 'resources',
    'cpu_packet_sha256', 'gpu_packet_sha256', 'cpu_acceptance_sha256', 'gpu_acceptance_sha256',
    'source_commit', 'freeze_sha256', 'namespace_admission_sha256',
    'operational_source_commit', 'gpu_recovery_acceptance_sha256', 'clock_adoption_sha256'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def decode(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            need(key not in result, 'duplicate input key')
            result[key] = item
        return result
    result = json.loads(value, object_pairs_hook=unique)
    raw(result)
    return result


def read(path, expected=None, maximum=256*1024**2):
    # Same no-follow, bounded, stable capture as the accepted CPU/GPU launchers.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, 'bounded regular input required')
        value = handle.read(maximum+1)
        after = os.fstat(handle.fileno())
    need(len(value) <= maximum and (before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'input changed during capture')
    if expected is not None:
        need(pin(expected) and sha(value) == expected, 'independent input pin differs: '+str(path))
    return value


def save(path, value):
    value = raw(value)
    with Path(path).open('xb') as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    return sha(value)


def clock_deadlines(started, authenticated=None):
    """One allocation origin; the worker's completed authentication opens startup."""
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    need(finite(started) and started >= 0, 'finite original monotonic origin required')
    result = {k: started+v for k, v in DEADLINES.items() if k != 'report_startup'}
    if authenticated is not None:
        need(finite(authenticated) and started <= authenticated < result['preparation'],
             'authentication did not finish within original preparation budget')
        result['report_startup'] = min(authenticated+REPORT_STARTUP, result['report_startup_cap'])
    return result


def arm_until(deadline):
    remaining = deadline-time.monotonic()
    need(remaining > 0, 'absolute phase deadline exhausted')
    signal.setitimer(signal.ITIMER_REAL, remaining)


def authentication_deadlines(value, started, packet_pin, pid):
    need(value['schema'] == VERSION and value['packet_sha256'] == packet_pin
         and type(value['pid']) is int and value['pid'] == pid
         and type(value['started_monotonic']) in (int, float) and value['started_monotonic'] == started,
         'owned worker authentication receipt differs')
    return clock_deadlines(started, value['monotonic'])


class PhaseTimings:
    """Exclusive diagnostic records; nested CPU durations must not be summed."""
    def __init__(self, output, started, role):
        self.output, self.started, self.role, self.sequence = Path(output), started, role, 0

    @contextlib.contextmanager
    def phase(self, name):
        index = self.sequence
        self.sequence += 1
        wall, cpu = time.monotonic(), time.process_time()
        details = {}
        record = dict(schema='ace-finalizer-phase-timing-v1', phase=name, role=self.role,
            pid=os.getpid(), sequence=index, origin_monotonic=self.started, started_monotonic=wall,
            started_elapsed_seconds=wall-self.started, started_process_cpu_seconds=cpu,
            started_utc=datetime.now(timezone.utc).isoformat(), scientific_acceptance=False)
        save(self.output/f'timing-{index:03d}.start.json', record)
        failure = None
        try:
            yield details
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            end, end_cpu = time.monotonic(), time.process_time()
            try:
                save(self.output/f'timing-{index:03d}.end.json', dict(record,
                    finished_monotonic=end, finished_elapsed_seconds=end-self.started,
                    elapsed_seconds=end-wall, process_cpu_seconds=end_cpu-cpu,
                    completed_utc=datetime.now(timezone.utc).isoformat(), details=details,
                    failure=failure, completed=failure is None))
            except BaseException:
                if failure is None:
                    raise
                traceback.print_exc(file=sys.stderr)


def admit_clock(packet, data):
    authority = decode(data['finalizer-clock-adoption.json'])
    need(sha(data['finalizer-clock-adoption.json']) == packet['clock_adoption_sha256']
         and authority['schema'] == 'ace-finalizer-clock-adoption-v1' and authority['status'] == 'adopted'
         and authority['scope'] == 'operational_finalization'
         and raw(authority['resources']) == raw(RESOURCES) and raw(authority['deadlines']) == raw(DEADLINES)
         and authority['entry_sha256'] == packet['files'][ENTRY]
         and authority['source_commit'] == packet['operational_source_commit']
         and authority['freeze_sha256'] == packet['freeze_sha256']
         and authority['cpu_acceptance_sha256'] == packet['cpu_acceptance_sha256']
         and authority['gpu_acceptance_sha256'] == packet['gpu_acceptance_sha256']
         and authority['gpu_recovery_acceptance_sha256'] == packet['gpu_recovery_acceptance_sha256']
         and authority['baseline_sha256'] == sha(data['gpu-success-baseline.json'])
         and authority['full_path_timing_supported'] is True
         and type(authority['required_findings']) is int and authority['required_findings'] == 0
         and all(pin(authority[k]) for k in ('timing_evidence_sha256', 'timing_review_sha256'))
         and authority['scientific_inputs_unchanged'] is True
         and authority['scientific_qualified'] is False and authority['execution_qualified'] is False,
         'independently adopted, timing-supported operational clock required')


def module(name, path, value):
    item = types.ModuleType(name)
    item.__file__ = str(path)
    exec(compile(value, str(path), 'exec'), vars(item))
    return item


def absolute(path):
    return (type(path) is str and path.startswith('/') and '\0' not in path
            and str(Path(path)) == path and '..' not in Path(path).parts)


def relative(path):
    return (type(path) is str and path not in ('', '.') and not Path(path).is_absolute()
            and '..' not in Path(path).parts and str(Path(path)) == path)


def separate(paths):
    # Preserve contractual mount aliases, but never permit aliases to hide overlap.
    paths = [Path(p).resolve() for p in paths]
    need(all(a != b and a not in b.parents and b not in a.parents
             for i, a in enumerate(paths) for b in paths[i+1:]), 'input/output roots overlap')


def packet_destination(path, expected, output):
    root = Path(path).absolute().parent
    p = decode(read(path, expected, 256*1024))
    need(type(p) is dict and set(p) == PACKET_KEYS and p['schema'] == VERSION
         and p['scope'] == 'development' and raw(p['resources']) == raw(RESOURCES), 'fixed finalizer packet required')
    paths = p['paths']
    need(type(paths) is dict and set(paths) == {'cpu_inputs', 'cpu_output', 'campaign', 'gpu_inputs',
         'gpu_output', 'output'} and all(absolute(x) for x in paths.values()),
         'fixed absolute finalizer paths required')
    need(paths['campaign'] == str(Path(paths['cpu_output'])/'worker/artifacts/campaign')
         and output == paths['output'] and type(p['source_commit']) is str
         and re.fullmatch('[0-9a-f]{40}', p['source_commit'])
         and type(p['operational_source_commit']) is str
         and re.fullmatch('[0-9a-f]{40}', p['operational_source_commit']), 'original campaign/output/source identity differs')
    separate([root]+[v for k, v in paths.items() if k != 'campaign'])
    old = decode(read(root/'cpu-packet.json', p['cpu_packet_sha256']))
    protected = [old['runtime_source'], old['model_directory'], old['archive']['path'], old['manifest']['path']]
    need(all(absolute(x) for x in protected), 'absolute protected runtime inputs required')
    for other in protected:
        separate([paths['output'], other])
    need(not root.is_symlink() and all(not Path(x).is_symlink() for x in paths.values()),
         'symlink evidence root forbidden')
    return root, p


def checked_inventory(api, root, expected, **kwargs):
    need(type(expected) is dict and expected and all(relative(n) and pin(p) for n, p in expected.items()),
         'complete independent inventory required')
    observed = api.inventory(root, **kwargs)
    need(observed == expected, 'independent immutable inventory differs: '+str(root))
    return observed


def inventory(root, *, exclude=None):
    """Bounded complete scientific custody, including every failed attempt."""
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), 'ordinary custody root required')
    need(exclude is None or relative(exclude), 'relative excluded subtree required')
    result, total, count = {}, 0, 0
    for path in sorted(root.rglob('*')):
        count += 1
        need(count <= 262144 and not path.is_symlink(), 'custody membership bound or symlink')
        name = path.relative_to(root).as_posix()
        if path.is_dir():
            continue
        need(path.is_file(), 'ordinary custody object required')
        if exclude is not None and (name == exclude or name.startswith(exclude+'/')):
            continue
        value = read(path)
        total += len(value)
        need(total <= 8*1024**3, 'complete custody byte bound exceeded')
        result[name] = sha(value)
    return result


def admit_evidence(packet, old, gpu_packet, data, api):
    """Read private inputs as opaque bytes until ALL upstream custody gates pass.

    The pinned acceptance is the external review authority for scheduler/runtime/
    cleanup. Their original evidence is covered by the full output inventories;
    this finalizer does not introduce another review or qualification hierarchy.
    """
    paths = packet['paths']
    cpu = decode(data['cpu-acceptance.json'])
    gpu = decode(data['gpu-acceptance.json'])
    cpu_keys = {'schema', 'status', 'cpu_packet_sha256', 'job_id', 'allocation', 'cpu_output',
        'campaign_root', 'cpu_output_files', 'cpu_output_excluded_directory', 'campaign_files',
        'bundle_path', 'bundle_sha256', 'phase_terminal_path', 'phase_terminal_sha256',
        'attempt_pins', 'scientific_qualified'}
    need(type(cpu) is dict and set(cpu) == cpu_keys and cpu['schema'] == 'ace-scientific-campaign-cpu-acceptance-v1'
         and cpu['status'] == 'development_cpu_phase_accepted' and cpu['scientific_qualified'] is False
         and cpu['cpu_packet_sha256'] == packet['cpu_packet_sha256']
         and re.fullmatch('[1-9][0-9]*', cpu['job_id']), 'independent CPU acceptance differs')
    need(cpu['allocation'] == dict(account='ucb736_asc2', partition='acpu', qos='cpu-normal',
         cpus=2, memory_mib=7680, gpus=0, state='COMPLETED', exit_code='0:0'), 'accepted CPU scheduler differs')
    need(type(gpu) is dict and set(gpu) == GPU_ACCEPTANCE_KEYS
         and gpu['schema'] == 'ace-scientific-campaign-gpu-acceptance-v1'
         and gpu['status'] == 'development_gpu_phase_accepted' and gpu['scope'] == 'development'
         and gpu['scientific_qualified'] is False and re.fullmatch('[1-9][0-9]*', gpu['job_id'])
         and all(gpu[k] == packet[k] for k in ('cpu_packet_sha256', 'gpu_packet_sha256', 'cpu_acceptance_sha256')),
         'independent GPU acceptance differs')
    need(gpu['accepted_checks'] == dict(scheduler=True, runtime=True, cleanup=True, custody=True)
         and all(type(v) is bool for v in gpu['accepted_checks'].values()), 'independent upstream checks missing')
    allocation = gpu['allocation']
    count = allocation.get('allocated_cpus')
    need(type(count) is int and count == 4 and allocation == dict(account='ucb736_asc2',
         partition='aa100', qos='gpu-testing', requested_cpus=4, allocated_cpus=count, memory_mib=15360,
         wall_seconds=gpu_packet['resources']['wall_seconds'], gpus=1, gpu_type='a100_3g.20gb', state='COMPLETED', exit_code='0:0'),
         'accepted GPU scheduler differs')
    need(cpu['cpu_output'] == paths['cpu_output'] == old['cpu_output']
         and cpu['campaign_root'] == gpu['campaign_root'] == paths['campaign']
         and gpu['gpu_output'] == paths['gpu_output']
         and cpu['cpu_output_excluded_directory'] == 'worker/artifacts/campaign', 'accepted evidence roots differ')
    checked_inventory(api, paths['cpu_output'], cpu['cpu_output_files'], exclude='worker/artifacts/campaign')
    checked_inventory(api, paths['gpu_output'], gpu['gpu_output_files'])
    complete = checked_inventory(api, paths['campaign'], gpu['campaign_files'])
    need(type(cpu['campaign_files']) is dict and cpu['campaign_files']
         and all(complete.get(n) == p for n, p in cpu['campaign_files'].items()), 'accepted CPU campaign bytes changed')
    need({n: p for n, p in complete.items() if n.split('/')[0] not in {'policy-05', 'policy-06', 'policy-07'}}
         == cpu['campaign_files'], 'campaign gained files outside three GPU attempts')
    campaign_root = Path(paths['campaign'])
    need({p.name for p in campaign_root.iterdir()} == {'campaign.json', 'source-plan-pins.json', 'start.json'} |
         {f'policy-{i:02d}' for i in range(8)}, 'complete untouched campaign required; no finalization/retry')
    need(set(cpu['attempt_pins']) == set(POLICIES[:5]) and set(gpu['attempt_pins']) == set(POLICIES)
         and all(pin(v) for v in gpu['attempt_pins'].values())
         and all(gpu['attempt_pins'][k] == v for k, v in cpu['attempt_pins'].items()), 'all eight independent attempt pins required')
    for i, pid in enumerate(POLICIES):
        name = f'policy-{i:02d}/attempt.json'
        need(complete.get(name) == gpu['attempt_pins'][pid]
             and complete.get(f'policy-{i:02d}/attempt.candidate.json') == gpu['attempt_pins'][pid],
             'attempt publication/custody pin differs')
    terminals = gpu['terminal_pins']
    need(type(terminals) is dict and set(terminals) == {'terminal.json', 'controller/phase/terminal.json'}
         and all(pin(p) and gpu['gpu_output_files'].get(n) == p for n, p in terminals.items()),
         'existing GPU terminal pins must match full output inventory')
    for field in ('bundle_path', 'phase_terminal_path'):
        need(relative(cpu[field]), 'relative CPU terminal/bundle path required')
        need(cpu['cpu_output_files'].get(cpu[field]) == cpu[field.replace('_path', '_sha256')],
             'CPU terminal/bundle absent from inventory')
    bundle_raw = read(Path(paths['cpu_output'])/cpu['bundle_path'], cpu['bundle_sha256'])
    need(gpu['bundle_sha256'] == cpu['bundle_sha256'], 'accepted bundle pin differs')
    ct = decode(read(Path(paths['cpu_output'])/cpu['phase_terminal_path'], cpu['phase_terminal_sha256']))
    gt = decode(read(Path(paths['gpu_output'])/'terminal.json', terminals['terminal.json']))
    phase = decode(read(Path(paths['gpu_output'])/'controller/phase/terminal.json', terminals['controller/phase/terminal.json']))
    need(ct['status'] == 'cpu_phase_complete' and ct['completed_cpu_cells'] == 120 and ct['declared_cells'] == 192
         and ct['attempt_pins'] == cpu['attempt_pins'] and ct['numeric_training_updates'] == expected_numeric_steps(data)
         and ct['llm_training_updates'] == 0 and ct['gpu_services_started'] == 0
         and ct['private_scores_computed'] is False, 'accepted CPU terminal differs')
    gpu_attempts = {p: gpu['attempt_pins'][p] for p in POLICIES[5:]}
    need(gt['schema'] == gpu_packet['schema'] and gt['status'] == 'development_gpu_campaign_complete'
         and gt['packet_sha256'] == packet['gpu_packet_sha256']
         and gt['cpu_acceptance_sha256'] == packet['cpu_acceptance_sha256']
         and gt['attempt_pins'] == phase['attempt_pins'] == gpu_attempts
         and gt['new_training_updates'] == 0 and gt['private_scores_computed'] is False
         and gt['scientific_qualified'] is False and phase['status'] == 'gpu_phase_complete'
         and phase['signals_restored'] is True and phase['failure'] is None
         and phase['errors'] == [] and phase['unstarted_not_recorded'] == [], 'accepted GPU terminal/cleanup differs')
    admit_recovery(packet, gpu_packet, data, gpu, gt, api)
    # First private bundle decode: every source, inventory, attempt and external
    # scheduler/runtime/cleanup acceptance has now been authenticated.
    bundle = decode(bundle_raw)
    c = bundle['campaign']
    need(c['scope'] == 'development' and c['policy_ids'] == list(POLICIES)
         and [s['plan']['world']['world_id'] for s in c['sources']] == list(WORLD_IDS)
         and all(s['plan']['acquisition_cap_units'] == 24 for s in c['sources'])
         and c['scientific_provenance_qualified'] is False and c['retry_allowed'] is False
         and bundle['campaign_sha256'] == sha(raw(c).rstrip(b'\n'))
         and ct['campaign_sha256'] == gt['campaign_sha256'] == phase['campaign_sha256'] == bundle['campaign_sha256'],
         'fixed 24-world development bundle differs')
    need(read(campaign_root/'campaign.json') == raw(c)
         and read(campaign_root/'source-plan-pins.json') == raw(bundle['source_plan_pins']), 'original campaign root differs')
    return cpu, gpu, bundle


def admit_recovery(packet, gp, data, gpu, terminal, api):
    """Independent GPU02 postcustody acceptance, before the first private decode.

    The unchanged17-field GPU acceptance remains the scheduler/runtime/cleanup
    authority. This separate byte-pinned supplement closes the historical check
    deliberately deferred by GPU02; its absence can never mean acceptance.
    """
    files=packet['files'];authority=decode(data[RECOVERY_FILES['authority']])
    expected=decode(data[RECOVERY_FILES['context']]);context=expected['execution_context']
    supplement=decode(data['gpu-recovery-acceptance.json'])
    failed=decode(data['gpu-failure-acceptance.json']);prior=decode(data['gpu-failure-baseline.json'])
    baseline=decode(data['gpu-success-baseline.json'])
    need(gp.get('recovery_files')==RECOVERY_FILES
         and all(sha(data[n])==files[n]==gp['files'][n] for n in RECOVERY_FILES.values())
         and files[RECOVERY_FILES['authority']]==RECOVERY_AUTHORITY_SHA,
         'independent original GPU02 recovery input bytes required')
    need(authority['schema']=='ace-gpu-cpu-history-recovery-authority-v1'
         and authority['status']=='operationally_adopted'
         and authority['historical_postcustody_required'] is authority['unchanged_scientific_inputs'] is True
         and authority['control_root']==GPU_CONTROL_ROOT and authority['failed_gpu01_root']==GPU_ORIGINAL_ROOT
         and authority['allowed_new_output']==gp['paths']['gpu_output']==GPU_ORIGINAL_ROOT+'/gpu'
         and gp['paths']['gpu_inputs']==GPU_CONTROL_ROOT+'/inputs'
         and authority['cpu_acceptance_sha256']==packet['cpu_acceptance_sha256']
         and authority['original_source_commit']==packet['source_commit'], 'original recovery authority/root differs')
    need(gp['source_names']==sorted(authority['operational_source_files'])
         and len(gp['source_names'])==56
         and {n:gp['files'][n] for n in gp['source_names']}==authority['operational_source_files']
         and context['source_files_sha256']==sha(raw(authority['operational_source_files']).rstrip(b'\n')),
         'exact operational source closure differs')
    scoped=api.check_preflight(data[RECOVERY_FILES['receipt']],files[RECOVERY_FILES['receipt']],
        data[RECOVERY_FILES['context']],files[RECOVERY_FILES['context']])
    need(scoped['full_external_postcustody_required'] is True,
         'preflight receipt cannot establish historical postcustody')
    need(authority['failed_gpu01_acceptance_sha256']==files['gpu-failure-acceptance.json']
         and authority['baseline']==dict(sha256=files['gpu-failure-baseline.json'],terminal_jobs=47,submission_journals=96)
         and failed['schema']=='ace-scientific-campaign-gpu-failure-acceptance-v1'
         and failed['status']=='accepted_failed_prepolicy_attempt' and failed['acceptance'] is True
         and type(failed['required_findings']) is int and failed['required_findings']==0
         and failed['job_id']==authority['original_gpu01_job_id']=='35498218'
         and prior['terminal_jobs']==47 and prior['submission_journals']==96
         and prior['input_pins']['acceptance']==files['gpu-failure-acceptance.json']
         and baseline['terminal_jobs']==48 and baseline['submission_journals']==98
         and baseline['outstanding_reservation']==0
         and baseline['input_pins']['predecessor']==files['gpu-failure-baseline.json']
         and baseline['input_pins']['acceptance']==packet['gpu_acceptance_sha256'],
         'independently accepted failed47/96 then successful48/98 lineage required')
    required=dict(schema='ace-scientific-campaign-gpu-recovery-acceptance-v1',status='accepted',
        scope='development',acceptance=True,required_findings=0,scientific_qualified=False,
        historical_postcustody_verified=True,originals_preserved=True,job_id=gpu['job_id'],
        gpu_acceptance_sha256=packet['gpu_acceptance_sha256'],gpu_packet_sha256=packet['gpu_packet_sha256'],
        cpu_acceptance_sha256=packet['cpu_acceptance_sha256'],authority_sha256=files[RECOVERY_FILES['authority']],
        preflight_context_sha256=files[RECOVERY_FILES['context']],preflight_receipt_sha256=files[RECOVERY_FILES['receipt']],
        readset_sha256=files[RECOVERY_FILES['read_set']],failure_acceptance_sha256=files['gpu-failure-acceptance.json'],
        failure_baseline_sha256=files['gpu-failure-baseline.json'],baseline_sha256=files['gpu-success-baseline.json'],
        source_files_sha256=context['source_files_sha256'],gpu_terminal_sha256=gpu['terminal_pins']['terminal.json'],
        phase_terminal_sha256=gpu['terminal_pins']['controller/phase/terminal.json'],
        control_root=GPU_CONTROL_ROOT,output_root=GPU_ORIGINAL_ROOT+'/gpu',
        raw_root_labels=dict(control=GPU_CONTROL_ROOT,launch=GPU_ORIGINAL_ROOT),cpu_campaign_root=gpu['campaign_root'],
        original_source_commit=packet['source_commit'],original_gpu_entry_sha256=authority['original_entry_sha256'],
        failed_gpu01_job_id=authority['original_gpu01_job_id'])
    need(type(supplement) is dict and set(required)<=set(supplement)
         and raw({k:supplement[k] for k in required})==raw(required),
         'independent actual GPU02 historical postcustody acceptance required')
    for name in ('before_observation_sha256','after_observation_sha256','custody_sha256','gpu_launch_packet_sha256',
                 'archive_receipt_sha256','failed_archive_receipt_sha256'):
        need(pin(supplement[name]),'independent full postcustody evidence pin required: '+name)
    need(type(supplement['operational_source_commit']) is str
         and re.fullmatch('[0-9a-f]{40}',supplement['operational_source_commit'])
         and type(supplement['evidence_pins']) is dict and all(pin(v) for v in supplement['evidence_pins'].values())
         and supplement['evidence_pins']['operational-source-review']==authority['source_review_sha256'],
         'independent captured operational source/review provenance required')
    marker_pin=gpu['gpu_output_files'].get(HISTORICAL_MARKER)
    need(pin(marker_pin) and supplement['postcustody_marker_sha256']==marker_pin,
         'actual deferred-postcustody marker must be in the full accepted GPU output inventory')
    marker=decode(read(Path(packet['paths']['gpu_output'])/HISTORICAL_MARKER,marker_pin))
    need(marker==dict(preflight=scoped,authority_sha256=RECOVERY_AUTHORITY_SHA,
         historical_postcheck_performed_on_gpu=False,full_external_postcustody_required=True,acceptance=False),
         'original GPU deferred-postcustody marker differs')
    need(terminal['operational_recovery_authority_sha256']==RECOVERY_AUTHORITY_SHA
         and terminal['historical_preflight_receipt_sha256']==files[RECOVERY_FILES['receipt']]
         and terminal['historical_postcheck_performed_on_gpu'] is False
         and terminal['full_external_postcustody_required'] is True,
         'GPU terminal must retain its original deferred historical-check boundary')


def expected_numeric_steps(data):
    plan = decode(data['numerical-plan.json'])
    d = plan['exposure']['counts']['dpo_steps']
    need(type(d) is int and 1 <= d <= 128
         and plan['exposure']['counts']['sft_steps'] == 160,
         'complete matched numerical exposure required')
    return 160+d


def upstream_shape(packet, old, gp, files, cpu, service, diagnostic):
    """Authenticate declarations without replaying completed upstream execution."""
    need(old['schema'] == 'ace-scientific-campaign-cpu-v1' and old['scope'] == 'development'
         and old['runtime'] == service.RUNTIME and old['runtime_source'] == diagnostic.RUNTIME_SOURCE
         and old['model_files'] == cpu.MODEL_FILES, 'original CPU runtime/model differs')
    for key, expected in (('archive', diagnostic.ARCHIVE_SHA), ('manifest', diagnostic.MANIFEST_SHA)):
        need(old[key]['sha256'] == expected and absolute(old[key]['path']), 'original runtime archive differs')
    need(old['freeze_sha256'] == packet['freeze_sha256'] == files['scientific-freeze.json']
         and old['namespace_admission_sha256'] == packet['namespace_admission_sha256']
         and pin(packet['namespace_admission_sha256']), 'original scientific freeze/namespace differs')
    need(gp['schema'] == 'ace-scientific-paired-campaign-gpu-cpu-history-v2' and gp['scope'] == 'development'
         and gp['cpu_packet_sha256'] == packet['cpu_packet_sha256']
         and gp['cpu_acceptance_sha256'] == packet['cpu_acceptance_sha256']
         and all(gp['files'].get(n) == h for n,h in old['files'].items())
         and set(old['source_names']) <= set(gp['source_names'])
         and {SCIENTIFIC_CPU_ENTRY, SCIENTIFIC_GPU_ENTRY} <= set(gp['source_names']),
         'accepted GPU source/input closure differs')
    r = gp['resources']
    need(type(r) is dict and set(r) == {'account','partition','qos','requested_cpus','max_allocated_cpus',
        'memory_mib','wall_seconds','gpus','gpu_type','requeue'}
        and r == dict(account='ucb736_asc2',partition='aa100',qos='gpu-testing',requested_cpus=4,
        max_allocated_cpus=4,memory_mib=15360,wall_seconds=r['wall_seconds'],gpus=1,
        gpu_type='a100_3g.20gb',requeue=False)
        and type(r['wall_seconds']) is int and 300 <= r['wall_seconds'] <= 3600,
        'independently accepted GPU resource envelope differs')
    need(set(files) == set(gp['files']) | {ENTRY, GPU_ENTRY, 'gpu-packet.json', 'gpu-acceptance.json'} | RECOVERY_ADDITIONS
         and all(files.get(n) == h for n,h in gp['files'].items())
         and packet['source_names'] == sorted(set(gp['source_names']) | {ENTRY, GPU_ENTRY})
         and all(n.endswith('.py') and n in files for n in packet['source_names']),
         'exact captured finalizer/source closure required')


def authenticate(path, expected, output, timings):
    root, packet = packet_destination(path, expected, output)
    files = packet['files']
    need(type(files) is dict and len(files) <= 256 and all(Path(n).name == n and
         n.endswith(('.py', '.json', '.pt')) and pin(p) for n, p in files.items()), 'flat pinned input roster required')
    fixed = {CPU_ENTRY: CPU_SHA, GPU_ENTRY: GPU_SHA, 'llm_intervention_service_runtime.py': RUNTIME_SHA,
             'llm_intervention_campaign.py': CAMPAIGN_SHA}
    need(all(files.get(n) == p for n, p in fixed.items()), 'accepted helper/source pins differ')
    for name in ('cpu-packet', 'gpu-packet', 'cpu-acceptance', 'gpu-acceptance', 'gpu-recovery-acceptance'):
        need(files.get(name+'.json') == packet[name.replace('-', '_')+'_sha256'], 'upstream authority pin differs')
    data = {}
    total = 0
    for n,p in files.items():
        data[n] = read(root/n, p)
        total += len(data[n])
        need(total <= 768*1024**2, 'total captured input bound exceeded')
    admit_clock(packet, data)
    need(Path(__file__).absolute() == root/ENTRY, 'captured finalizer required')
    cpu = module('finalizer_cpu_helpers', root/CPU_ENTRY, data[CPU_ENTRY])
    gpu = module('finalizer_inventory_helpers', root/GPU_ENTRY, data[GPU_ENTRY])
    old, gp = decode(data['cpu-packet.json']), decode(data['gpu-packet.json'])
    need(all(files.get(n) == p for n, p in cpu.HELPER_PINS.items()), 'exact accepted runtime/helper pins required')
    diagnostic = module('finalizer_diagnostic', root/cpu.DIAGNOSTIC, data[cpu.DIAGNOSTIC])
    service = module('finalizer_service_contract', root/cpu.SERVICE, data[cpu.SERVICE])
    training = module('finalizer_environment', root/cpu.TRAINING, data[cpu.TRAINING])
    upstream_shape(packet, old, gp, files, cpu, service, diagnostic)
    paths = packet['paths']
    need(all(paths[k] == v for k, v in gp['paths'].items()), 'original GPU paths differ')
    for key, name, previous in (('cpu_inputs', 'cpu-packet.json', old), ('gpu_inputs', 'gpu-packet.json', gp)):
        origin = Path(paths[key])
        need(read(origin/'packet.json', files[name]) == data[name], 'original upstream packet differs')
        for n, p in previous['files'].items():
            need(read(origin/n, p) == data[n], 'original upstream input changed: '+n)
    checker = module('finalizer_preflight_checker', root/PREFLIGHT, data[PREFLIGHT])
    def timed_inventory(tree, **kwargs):
        label = next((k for k in ('cpu_output', 'gpu_output', 'campaign') if str(tree) == paths[k]), None)
        need(label is not None, 'explicit inventory timing root required')
        with timings.phase('inventory_'+label) as metrics:
            observed = inventory(tree, **kwargs)
            metrics['files'] = len(observed)
            return observed
    api = types.SimpleNamespace(inventory=timed_inventory, check_preflight=checker.validate_receipt)
    ca, ga, bundle = admit_evidence(packet, old, gp, data, api)
    return dict(root=root, packet=packet, data=data, cpu=cpu, api=api, old=old, diagnostic=diagnostic,
                training=training, cpu_acceptance=ca, gpu_acceptance=ga, bundle=bundle, timings=timings)


def allocation(text, job):
    pairs = re.findall(r'([^\s=]+)=(\S+)', text)
    fields = dict(pairs)
    need(len(fields) == len(pairs) and re.fullmatch('[1-9][0-9]*', job), 'unique scheduler identity required')
    wanted = dict(JobId=job, JobState='RUNNING', Account='ucb736_asc2', Partition='acpu', QOS='cpu-normal',
                  NumCPUs='2', NumTasks='1', NumNodes='1', TimeLimit='01:10:00', Requeue='0', Restarts='0')
    need(all(fields.get(k) == v for k, v in wanted.items()) and fields.get('UserId', '').startswith('paco0228('),
         'fixed CPU-only finalizer allocation differs')
    for name in ('AllocTRES', 'ReqTRES'):
        rows = [x.split('=') for x in fields.get(name, '').split(',')]
        need(all(len(x) == 2 for x in rows), 'malformed resource list')
        tres = dict(rows)
        need(len(tres) == len(rows) and tres.get('cpu') == '2' and tres.get('node') == '1'
             and tres.get('mem') in ('7680M', '7.5G', '7.50G')
             and not any(k.startswith('gres/') for k in tres), 'requested/allocated resources differ')
    need(not any(os.environ.get(k) for k in ('SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES')),
         'CPU-only environment required')
    return fields


def runtime_record(ctx, loader, stage, *, after):
    """Use the exact pinned runtime filters/checks; no new mapping exemptions."""
    import llm_intervention_service_runtime as checks
    import qualify_llm_intervention_runtime_metadata as diagnostic
    prefix = Path(stage['staged_prefix'])
    need(sys.flags.isolated == 1 and sys.dont_write_bytecode and sys.version_info[:3] == (3, 10, 19)
         and os.getpgrp() == os.getpid(), 'isolated owned NumPy worker required')
    checks.check_stage(stage, ctx['old']['runtime'], prefix)
    controls = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1', MKL_THREADING_LAYER='GNU', LD_LIBRARY_PATH=str(prefix/'lib'))
    need(all(os.environ.get(k) == v for k, v in controls.items()) and not any(os.environ.get(k) for k in
         ('PYTHONHOME', 'PYTHONPATH', 'LD_PRELOAD', 'MKL_SERVICE_FORCE_INTEL', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS')),
         'fixed CPU environment differs')
    forbidden = {'torch', 'transformers', 'tokenizers', 'scipy', 'safetensors'}
    need(not any(n.split('.')[0] in forbidden for n in sys.modules), 'non-NumPy runtime import forbidden')
    captured = {n: ctx['data'][n] for n in ctx['packet']['source_names']}
    modules, projects = checks.runtime_modules(sys.modules, loader, captured, ctx['root']/ENTRY, globals())
    snapshot = diagnostic.runtime_snapshot(prefix, ctx['old']['runtime_source'], modules=modules)
    diagnostic.check_prefix(prefix, snapshot)
    need(not diagnostic.inside(prefix, ctx['old']['runtime_source'])
         and not snapshot['generated_provenance']['present'], 'original/generated runtime forbidden')
    if after:
        need(snapshot['packages']['numpy']['version'] == ctx['old']['runtime']['imported_versions']['numpy']
             and any(n.startswith('numpy.') and p.endswith('.so') for n, p in snapshot['module_paths'].items()),
             'actual pinned NumPy extension required')
    else:
        need('numpy' not in sys.modules, 'NumPy imported before runtime admission')
    return dict(snapshot=snapshot, captured_project_modules=projects, environment=controls)


def validate_report(ctx):
    """Retain complete engine artifacts; only validate the fixed reporting envelope."""
    root = Path(ctx['packet']['paths']['campaign'])
    observed = ctx['api'].inventory(root)
    before = ctx['gpu_acceptance']['campaign_files']
    need(all(observed.get(n) == p for n, p in before.items()), 'original campaign evidence changed during finalization')
    added = set(observed)-set(before)
    need(all(n.startswith('private/') or n in {'finalization-intent.json', 'finalization.json',
         'finalization.candidate.json'} for n in added), 'finalizer wrote outside private reporting surface')
    summary = decode(read(root/'private/summary.json'))
    receipt = decode(read(root/'finalization.json'))
    need(read(root/'finalization.candidate.json') == raw(receipt)
         and receipt['summary_sha256'] == observed['private/summary.json']
         and receipt['campaign_sha256'] == ctx['bundle']['campaign_sha256']
         and receipt['attempt_pins'] == ctx['gpu_acceptance']['attempt_pins']
         and receipt['cell_count'] == 192 and receipt['scientific_acceptance'] is False
         and receipt['new_environment_responses'] == receipt['new_training_updates'] == 0
         and receipt['retry_allowed'] is False, 'campaign finalization receipt differs')
    need(summary['scope'] == 'development' and summary['cell_count'] == 192
         and [(r['world_id'], r['policy_id']) for r in summary['rows']] == list(itertools.product(WORLD_IDS, POLICIES))
         and [(p['left'], p['right']) for p in summary['policy_pairs']] == list(itertools.combinations(POLICIES, 2))
         and all(p['world_ids'] == list(WORLD_IDS) for p in summary['policy_pairs'])
         and all(summary[k] is False for k in ('scientific_acceptance', 'scientific_protocol_frozen',
             'scientific_provenance_qualified', 'complete_case_filtering_allowed')), '192-cell/28-pair descriptive envelope differs')
    for i, row in enumerate(summary['rows']):
        if row['cell_status'] == 'reported':
            report = decode(read(root/f'private/world-{i//8:02d}/arm-{i%8:02d}/cost-report.json'))
            need(type(report['curve']) is list and 'curve_complete' in report, 'full curve evidence missing')
    return dict(cell_count=192, descriptive_policy_comparisons=28,
        failed_or_unresolved_cells=sum(bool(r['failed_or_unresolved']) for r in summary['rows']),
        finalization_sha256=observed['finalization.json'], summary_sha256=observed['private/summary.json'],
        campaign_files=observed, new_environment_responses=0, new_training_updates=0, scientific_acceptance=False)


def worker(args, ctx, handoff):
    out = Path(args.output)/'worker'
    stage = decode(read(Path(args.output)/'stage.json', handoff['stage_sha256']))
    need(handoff['parent_pid'] == os.getppid() and handoff['job_id'] == os.environ.get('SLURM_JOB_ID'),
         'owned worker handoff differs')
    captured = {n: ctx['data'][n] for n in ctx['packet']['source_names']}
    loader = ctx['cpu'].CapturedSources(captured, ctx['root'])
    sys.meta_path.insert(0, loader)
    try:
        with ctx['timings'].phase('report_runtime_startup'):
            save(out/'runtime-before.json', runtime_record(ctx, loader, stage, after=False))
            import llm_intervention_campaign as campaign
            save(out/'runtime-ready.json', runtime_record(ctx, loader, stage, after=True))
        deadlines = clock_deadlines(args.started, args.authenticated)
        need(time.monotonic() < deadlines['report_startup'], 'finalizer reporting startup deadline exceeded')
        save(out/'work-start.json', dict(packet_sha256=args.packet_sha256, utc=datetime.now(timezone.utc).isoformat(),
            monotonic=time.monotonic(), pid=os.getpid(), private_finalization_calls=1,
            authentication_sha256=args.authentication_sha256, report_startup_deadline=deadlines['report_startup']))
        arm_until(deadlines['work'])
        bundle = ctx['bundle']
        # The existing engine gates all attempts/closures and claims its own
        # once-only intent. Never rebuild or relocate the completed campaign.
        with ctx['timings'].phase('scientific_reporting'):
            campaign.finalize_campaign(campaign=bundle['campaign'], campaign_sha256=bundle['campaign_sha256'],
                source_plan_pins=bundle['source_plan_pins'], output=ctx['packet']['paths']['campaign'],
                attempt_pins=ctx['gpu_acceptance']['attempt_pins'])
        with ctx['timings'].phase('runtime_after'):
            save(out/'runtime-after.json', runtime_record(ctx, loader, stage, after=True))
        with ctx['timings'].phase('worker_validate_report'):
            report = validate_report(ctx)
        save(out/'terminal.json', dict(schema=VERSION, status='descriptive_finalization_complete',
            packet_sha256=args.packet_sha256, completed_utc=datetime.now(timezone.utc).isoformat(),
            report=report, retry_allowed=False, independent_acceptance_required=True))
    finally:
        sys.meta_path.remove(loader)


class OwnedChild:
    """Keep the live process handle until both leader and group have closed."""
    def __init__(self, started):
        self.started, self.process = started, None

    def confirm_closed(self):
        if self.process is None:
            return True
        try:
            exited = self.process.poll() is not None
            try:
                os.killpg(self.process.pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
        except BaseException:
            return False
        if exited and not alive:
            self.process = None
            return True
        return False

    def hard_exit(self, sig=None, frame=None):
        try:
            if self.process is not None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except OSError:
                    pass
        finally:
            os._exit(124)

    def hold_until_closed(self):
        if self.confirm_closed():
            return
        signal.signal(signal.SIGALRM, self.hard_exit)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL, max(.001, self.started+HARD-time.monotonic()))
        while not self.confirm_closed():
            if time.monotonic() >= self.started+HARD:
                self.hard_exit()
            time.sleep(.05)


def supervise(command, env, out, started, packet_pin, kill_owned, owned):
    """Parent independently observes preparation, reporting startup and work."""
    proc, cleanup, ready = None, None, False
    authenticated, authentication_pin = None, None
    deadlines = clock_deadlines(started)
    primary, cleanup_error, record_error = None, None, None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        with (out/'worker.stdout.log').open('xb') as stdout, (out/'worker.stderr.log').open('xb') as stderr:
            arm_until(deadlines['preparation'])
            proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            owned.process = proc
            save(out/'worker.started.json', dict(command=command, pid=proc.pid, pgid=proc.pid,
                started_monotonic=started, preparation_deadline=deadlines['preparation'],
                report_startup_cap=deadlines['report_startup_cap'], work_deadline=deadlines['work']))
            while True:
                authentication = out/'worker/authentication-complete.json'
                if authenticated is None and authentication.exists():
                    captured = read(authentication)
                    value = decode(captured)
                    deadlines = authentication_deadlines(value, started, packet_pin, proc.pid)
                    authenticated, authentication_pin = value['monotonic'], sha(captured)
                    arm_until(deadlines['report_startup'])
                marker = out/'worker/work-start.json'
                if not ready and marker.exists():
                    value = decode(read(marker))
                    need(authenticated is not None and value['packet_sha256'] == packet_pin and value['pid'] == proc.pid
                         and type(value['monotonic']) in (int, float)
                         and authenticated <= value['monotonic'] < deadlines['report_startup']
                         and value['authentication_sha256'] == authentication_pin
                         and value['report_startup_deadline'] == deadlines['report_startup'], 'worker startup receipt differs')
                    ready = True
                    arm_until(deadlines['work'])
                limit = deadlines['work'] if ready else deadlines['report_startup'] if authenticated is not None else deadlines['preparation']
                need(time.monotonic() < limit, 'finalizer preparation/startup/work deadline exceeded')
                rc = proc.poll()
                if rc is not None:
                    need(rc == 0 and ready, 'finalizer worker failed or never began reporting')
                    break
                time.sleep(.05)
    except BaseException as exc:
        primary = exc
    # No return or exception may discard the actual owned handle. main() holds
    # it through every subsequent publication failure until closure or HARD.
    cleanup_started, cleanup_cpu_started = time.monotonic(), time.process_time()
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, max(.001, started+CLEANUP-time.monotonic()))
    try:
        if proc is not None:
            cleanup = kill_owned(proc)
            save(out/'worker.cleanup.json', cleanup)
            need(cleanup['group_still_exists'] is False and cleanup['errors'] == []
                 and cleanup['returncode'] == 0 and cleanup['kill_sent'] is False,
                 'clean natural worker closure required')
        need(owned.confirm_closed(), 'owned leader/group closure unconfirmed')
    except BaseException as exc:
        cleanup_error = exc
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        signal.setitimer(signal.ITIMER_REAL, max(.001, started+HARD-time.monotonic()))
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        def error(exc):
            return None if exc is None else dict(type=type(exc).__name__, message=str(exc))
        try:
            save(out/'worker.resources.json', dict(elapsed_seconds=time.monotonic()-started,
                cleanup_elapsed_seconds=time.monotonic()-cleanup_started,
                cleanup_parent_cpu_seconds=time.process_time()-cleanup_cpu_started,
                cleanup_deadline=started+CLEANUP, postcheck_deadline=started+HARD,
                child_and_cleanup_cpu_seconds=after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime,
                cumulative_child_peak_rss_bytes=after.ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
                first_failure=error(primary), cleanup_failure=error(cleanup_error)))
        except BaseException as exc:
            record_error = exc
    if primary is not None:
        raise primary from (cleanup_error or record_error)
    if cleanup_error is not None:
        raise cleanup_error from record_error
    if record_error is not None:
        raise record_error
    return cleanup


def parent(args, ctx):
    out, d = Path(args.output), ctx['diagnostic']
    args.owned_child = OwnedChild(args.started)
    job = os.environ.get('SLURM_JOB_ID', '')
    query = subprocess.run(['scontrol', 'show', 'job', '-o', job], capture_output=True, text=True, timeout=10)
    save(out/'allocation.json', dict(stdout=query.stdout, stderr=query.stderr, returncode=query.returncode))
    need(query.returncode == 0, 'allocation query failed')
    allocation(query.stdout, job)
    scratch = Path(os.environ.get('SLURM_SCRATCH', ''))
    need(scratch.is_absolute() and scratch.is_dir() and scratch.stat().st_uid == os.getuid(), 'owned node scratch required')
    local = scratch.resolve()/('ace-campaign-finalizer-'+job)
    local.mkdir(mode=0o700, exist_ok=False)
    archive = module('finalizer_archive', ctx['root']/'llm_intervention_runtime_archive.py',
                     ctx['data']['llm_intervention_runtime_archive.py'])
    probe = module('finalizer_cleanup', ctx['root']/'probe.py', ctx['data']['probe.py'])
    event = d.events(out, args.started, args.process_started, 7680*1024**2)
    event('stage_start')
    with ctx['timings'].phase('runtime_staging'):
        stage = d.stage_runtime(ctx['old'], archive, local/'runtime', event)
    stage_pin = save(out/'stage.json', stage)
    event('stage_complete', staged_prefix=stage['staged_prefix'])
    need(time.monotonic() < args.started+PREPARATION, 'staging consumed preparation budget')
    handoff = dict(schema=VERSION, packet_sha256=args.packet_sha256, stage_sha256=stage_pin,
                   started_monotonic=args.started, parent_pid=os.getpid(), job_id=job)
    handoff_pin = save(out/'handoff.json', handoff)
    prefix = Path(stage['staged_prefix'])
    command = [str(prefix/'bin/python'), '-I', '-B', '-c', d.BOOTSTRAP, str(ctx['root']/ENTRY),
        ctx['packet']['files'][ENTRY], '--packet', str(Path(args.packet).absolute()), '--packet-sha256', args.packet_sha256,
        '--output', args.output, '--worker', '--handoff-sha256', handoff_pin]
    env = ctx['training'].cpu_environment(prefix, os.environ)
    arm_until(args.started+PREPARATION)
    with ctx['timings'].phase('worker_supervision_and_cleanup'):
        cleanup = supervise(command, env, out, args.started, args.packet_sha256, probe.kill_owned, args.owned_child)
    terminal = decode(read(out/'worker/terminal.json'))
    with ctx['timings'].phase('parent_validate_report'):
        report = validate_report(ctx)
    need(terminal['schema'] == VERSION and terminal['status'] == 'descriptive_finalization_complete'
         and terminal['packet_sha256'] == args.packet_sha256 and terminal['report'] == report,
         'worker terminal differs from retained report')
    # Frozen upstream inventories must also remain unchanged after reporting.
    ca, ga, paths = ctx['cpu_acceptance'], ctx['gpu_acceptance'], ctx['packet']['paths']
    with ctx['timings'].phase('parent_postchecks'):
        checked_inventory(ctx['api'], paths['cpu_output'], ca['cpu_output_files'], exclude='worker/artifacts/campaign')
        checked_inventory(ctx['api'], paths['gpu_output'], ga['gpu_output_files'])
    need(time.monotonic() < args.started+HARD, 'finalizer hard deadline exceeded')
    return dict(job_id=job, report=report, cleanup=cleanup, stage_sha256=stage_pin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'packet-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--handoff-sha256')
    args = parser.parse_args()
    args.started, args.process_started = time.monotonic(), time.process_time()
    claimed, error, result = None, None, None

    def hard_stop(sig=None, frame=None):
        owned = getattr(args, 'owned_child', None)
        if owned is not None:
            owned.hard_exit()
        os._exit(124)

    def reserve_publication():
        signal.signal(signal.SIGALRM, hard_stop)
        remaining = args.started+HARD-time.monotonic()
        if remaining <= 0:
            hard_stop()
        signal.setitimer(signal.ITIMER_REAL, remaining)

    def interrupted(sig, frame):
        reserve_publication()
        raise TimeoutError('finalizer signal '+str(sig))

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}
    arm_until(args.started+PREPARATION)
    try:
        _, packet = packet_destination(args.packet, args.packet_sha256, args.output)
        handoff = None
        if args.worker:
            need(pin(args.handoff_sha256), 'pinned worker handoff required')
            handoff = decode(read(Path(args.output)/'handoff.json', args.handoff_sha256))
            start = handoff['started_monotonic']
            need(handoff['schema'] == VERSION and handoff['packet_sha256'] == args.packet_sha256
                 and type(start) in (int, float) and math.isfinite(start)
                 and 0 <= args.started-start < PREPARATION, 'original absolute preparation clock required')
            args.started = start
            arm_until(start+PREPARATION)
        else:
            need(args.handoff_sha256 is None, 'parent cannot accept a runtime handoff')
        candidate = Path(args.output)/'worker' if args.worker else Path(args.output)
        candidate.mkdir(mode=0o700, exist_ok=False)
        claimed = candidate
        save(claimed/'start.json', dict(schema=VERSION, packet_sha256=args.packet_sha256, resources=RESOURCES,
            started_utc=datetime.now(timezone.utc).isoformat(), started_monotonic=args.started,
            deadlines=clock_deadlines(args.started), report_startup_seconds=REPORT_STARTUP, retry_allowed=False))
        timings = PhaseTimings(claimed, args.started, 'worker' if args.worker else 'parent')
        with timings.phase('full_authentication'):
            ctx = authenticate(args.packet, args.packet_sha256, args.output, timings)
        if args.worker:
            args.authenticated = time.monotonic()
            deadlines = clock_deadlines(args.started, args.authenticated)
            arm_until(deadlines['report_startup'])
            args.authentication_sha256 = save(claimed/'authentication-complete.json', dict(schema=VERSION,
                packet_sha256=args.packet_sha256, started_monotonic=args.started, monotonic=args.authenticated,
                pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(),
                process_cpu_seconds=time.process_time()-args.process_started,
                report_startup_deadline=deadlines['report_startup']))
        result = worker(args, ctx, handoff) if args.worker else parent(args, ctx)
    except BaseException as exc:
        reserve_publication()
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if claimed is not None:
            try:
                save(claimed/'failure.json', dict(schema=VERSION, status='terminal_failure', error=error,
                    retry_allowed=False, scientific_acceptance=False,
                    campaign_publication_authority='original campaign/finalization.json; reconcile separately if present'))
            except BaseException:
                traceback.print_exc(file=sys.stderr)
        raise
    finally:
        reserve_publication()
        try:
            if claimed is not None:
                save(claimed/'attempt.json', dict(schema=VERSION, status='failed' if error else 'completed',
                    packet_sha256=args.packet_sha256, error=error, result=result,
                    completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-args.started,
                    process_cpu_seconds=time.process_time()-args.process_started, retry_allowed=False,
                    scientific_acceptance=False, independent_acceptance_required=True))
        finally:
            owned = getattr(args, 'owned_child', None)
            if owned is not None:
                owned.hold_until_closed()
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in previous.items():
                signal.signal(s, handler)


if __name__ == '__main__':
    main()

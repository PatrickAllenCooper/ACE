"""Once-only CPU reporting after independent artificial CPU/GPU acceptance.

No submission, training, policy execution, model loading or new responses. The
packet captures the accepted GPU input roster plus this entry and the independently
produced GPU acceptance. Hash inventories before decoding the private bundle.
See GPU_ACCEPTANCE_KEYS for the deliberately data-only upstream admission record.
"""
from __future__ import annotations

import argparse
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

VERSION = 'ace-artificial-campaign-finalizer-cpu-v1'
ENTRY = 'finalize_llm_intervention_campaign_cpu.py'
CPU_ENTRY = 'prepare_llm_intervention_campaign_cpu.py'
GPU_ENTRY = 'qualify_llm_intervention_campaign_gpu.py'
CPU_SHA = '850ca3d4d7ac4799f68d74bd2d0b1196d10809751a4bdf878633122fc6a9c2c4'
GPU_SHA = '58f64d3eea63ae56cc7d14d9188c5d1f9e6e4126fc5b4512abc9758006a803bc'
RUNTIME_SHA = '117dc22bd56889906612bff0c975095918e062974fa537757b3e93c981dbfbc0'
CAMPAIGN_SHA = '9a2717dd68c37ddde6d7fb63c6cc519c067f7b1d457d4f49f919c18342424bbe'
POLICIES = ('uniform_random', 'menu_coverage', 'local_information', 'target_risk',
            'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm')
RESOURCES = dict(account='ucb736_asc2', partition='acpu', qos='cpu-normal', cpus=2,
                 memory_mib=7680, wall_seconds=300, gpus=0, requeue=False)
STARTUP, WORK, CLEANUP, HARD = 120, 260, 280, 290
GPU_ACCEPTANCE_KEYS = {'schema', 'status', 'scope', 'gpu_packet_sha256', 'cpu_packet_sha256',
    'cpu_acceptance_sha256', 'job_id', 'allocation', 'gpu_output', 'campaign_root',
    'qualification_root', 'gpu_output_files', 'campaign_files', 'qualification_files',
    'bundle_sha256', 'attempt_pins', 'accepted_checks', 'terminal_pins', 'scientific_qualified'}
PACKET_KEYS = {'schema', 'scope', 'files', 'source_names', 'paths', 'resources',
    'cpu_packet_sha256', 'gpu_packet_sha256', 'cpu_acceptance_sha256', 'gpu_acceptance_sha256'}


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


def read(path, expected=None, maximum=32*1024**2):
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
         and p['scope'] == 'artificial' and raw(p['resources']) == raw(RESOURCES), 'fixed finalizer packet required')
    paths = p['paths']
    need(type(paths) is dict and set(paths) == {'cpu_inputs', 'cpu_output', 'campaign', 'gpu_inputs',
         'gpu_output', 'gpu_qualification', 'output'} and all(absolute(x) for x in paths.values()),
         'fixed absolute finalizer paths required')
    need(paths['campaign'] == str(Path(paths['cpu_output'])/'worker/artifacts/campaign')
         and output == paths['output'], 'original campaign/output path differs')
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


def admit_evidence(packet, old, gpu_packet, data, api):
    """Read private inputs as opaque bytes until ALL upstream custody gates pass.

    The pinned acceptance is the external review authority. scheduler/runtime/
    cleanup evidence is kept as opaque, fully inventoried records, not invented
    locally or accepted from a phase's self-qualification flags.
    """
    paths = packet['paths']
    cpu = decode(data['cpu-acceptance.json'])
    gpu = decode(data['gpu-acceptance.json'])
    cpu_keys = {'schema', 'status', 'cpu_packet_sha256', 'job_id', 'allocation', 'cpu_output',
        'campaign_root', 'cpu_output_files', 'cpu_output_excluded_directory', 'campaign_files',
        'bundle_path', 'bundle_sha256', 'phase_terminal_path', 'phase_terminal_sha256',
        'attempt_pins', 'scientific_qualified'}
    need(type(cpu) is dict and set(cpu) == cpu_keys and cpu['schema'] == 'ace-paired-campaign-cpu-acceptance-v1'
         and cpu['status'] == 'artificial_cpu_phase_accepted' and cpu['scientific_qualified'] is False
         and cpu['cpu_packet_sha256'] == packet['cpu_packet_sha256']
         and re.fullmatch('[1-9][0-9]*', cpu['job_id']), 'independent CPU acceptance differs')
    need(cpu['allocation'] == dict(account='ucb736_asc2', partition='acpu', qos='cpu-normal',
         cpus=2, memory_mib=7680, gpus=0, state='COMPLETED', exit_code='0:0'), 'accepted CPU scheduler differs')
    need(type(gpu) is dict and set(gpu) == GPU_ACCEPTANCE_KEYS
         and gpu['schema'] == 'ace-paired-campaign-gpu-acceptance-v1'
         and gpu['status'] == 'artificial_gpu_phase_accepted' and gpu['scope'] == 'artificial'
         and gpu['scientific_qualified'] is False and re.fullmatch('[1-9][0-9]*', gpu['job_id'])
         and all(gpu[k] == packet[k] for k in ('cpu_packet_sha256', 'gpu_packet_sha256', 'cpu_acceptance_sha256')),
         'independent GPU acceptance differs')
    need(gpu['accepted_checks'] == dict(scheduler=True, runtime=True, cleanup=True, custody=True)
         and all(type(v) is bool for v in gpu['accepted_checks'].values()), 'independent upstream checks missing')
    allocation = gpu['allocation']
    count = allocation.get('allocated_cpus')
    need(type(count) is int and 4 <= count <= 8 and allocation == dict(account='ucb736_asc2',
         partition='aa100', qos='gpu-testing', requested_cpus=4, allocated_cpus=count, memory_mib=15360,
         wall_seconds=600, gpus=1, gpu_type='a100_3g.20gb', state='COMPLETED', exit_code='0:0'),
         'accepted GPU scheduler differs')
    need(cpu['cpu_output'] == paths['cpu_output'] == old['cpu_output']
         and cpu['campaign_root'] == gpu['campaign_root'] == paths['campaign']
         and gpu['gpu_output'] == paths['gpu_output'] and gpu['qualification_root'] == paths['gpu_qualification']
         and cpu['cpu_output_excluded_directory'] == 'worker/artifacts/campaign', 'accepted evidence roots differ')
    checked_inventory(api, paths['cpu_output'], cpu['cpu_output_files'], exclude='worker/artifacts/campaign')
    checked_inventory(api, paths['gpu_output'], gpu['gpu_output_files'])
    checked_inventory(api, paths['gpu_qualification'], gpu['qualification_files'])
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
    need(type(terminals) is dict and set(terminals) == {'scheduler', 'runtime', 'cleanup', 'supervisor', 'phase'},
         'full upstream terminal pins required')
    captured = {}
    for role, ref in terminals.items():
        need(type(ref) is dict and set(ref) == {'root', 'path', 'sha256'} and relative(ref['path'])
             and ref['root'] in ('gpu_output', 'gpu_qualification') and pin(ref['sha256']), 'terminal reference differs')
        inv = gpu['gpu_output_files' if ref['root'] == 'gpu_output' else 'qualification_files']
        need(inv.get(ref['path']) == ref['sha256'], 'terminal not in immutable inventory')
        captured[role] = read(Path(paths[ref['root']])/ref['path'], ref['sha256'])
    for role, name in (('supervisor', 'terminal.json'), ('phase', 'controller/phase/terminal.json')):
        need(terminals[role]['root'] == 'gpu_output' and terminals[role]['path'] == name, 'fixed GPU terminal differs')
    for field in ('bundle_path', 'phase_terminal_path'):
        need(relative(cpu[field]), 'relative CPU terminal/bundle path required')
        need(cpu['cpu_output_files'].get(cpu[field]) == cpu[field.replace('_path', '_sha256')],
             'CPU terminal/bundle absent from inventory')
    bundle_raw = read(Path(paths['cpu_output'])/cpu['bundle_path'], cpu['bundle_sha256'])
    need(gpu['bundle_sha256'] == cpu['bundle_sha256'], 'accepted bundle pin differs')
    ct = decode(read(Path(paths['cpu_output'])/cpu['phase_terminal_path'], cpu['phase_terminal_sha256']))
    gt, phase = decode(captured['supervisor']), decode(captured['phase'])
    need(ct['status'] == 'cpu_phase_complete' and ct['failed_policies'] == []
         and ct['attempt_pins'] == cpu['attempt_pins'] and ct['numeric_training_updates'] == 6
         and ct['llm_training_updates'] == 0 and ct['gpu_services_started'] == 0
         and ct['private_scores_computed'] is False, 'accepted CPU terminal differs')
    gpu_attempts = {p: gpu['attempt_pins'][p] for p in POLICIES[5:]}
    need(gt['schema'] == gpu_packet['schema'] and gt['status'] == 'artificial_gpu_campaign_complete'
         and gt['packet_sha256'] == packet['gpu_packet_sha256']
         and gt['cpu_acceptance_sha256'] == packet['cpu_acceptance_sha256']
         and gt['attempt_pins'] == phase['attempt_pins'] == gpu_attempts
         and gt['new_training_updates'] == 0 and gt['private_scores_computed'] is False
         and gt['scientific_qualified'] is False and phase['status'] == 'gpu_phase_complete'
         and phase['signals_restored'] is True and phase['failure'] is None
         and phase['errors'] == [] and phase['unstarted_not_recorded'] == [], 'accepted GPU terminal/cleanup differs')
    # First private bundle decode: every source, inventory, attempt and external
    # scheduler/runtime/cleanup acceptance has now been authenticated.
    bundle = decode(bundle_raw)
    c = bundle['campaign']
    need(c['scope'] == 'artificial' and c['policy_ids'] == list(POLICIES)
         and [s['plan']['world']['world_id'] for s in c['sources']] == [9101, 9102]
         and all(s['plan']['acquisition_cap_units'] == 4 for s in c['sources'])
         and c['scientific_provenance_qualified'] is False and c['retry_allowed'] is False
         and bundle['campaign_sha256'] == sha(raw(c).rstrip(b'\n'))
         and ct['campaign_sha256'] == gt['campaign_sha256'] == phase['campaign_sha256'] == bundle['campaign_sha256'],
         'fixed artificial bundle differs')
    need(read(campaign_root/'campaign.json') == raw(c)
         and read(campaign_root/'source-plan-pins.json') == raw(bundle['source_plan_pins']), 'original campaign root differs')
    return cpu, gpu, bundle


def authenticate(path, expected, output):
    root, packet = packet_destination(path, expected, output)
    files = packet['files']
    need(type(files) is dict and len(files) <= 128 and all(Path(n).name == n and
         n.endswith(('.py', '.json', '.pt')) and pin(p) for n, p in files.items()), 'flat pinned input roster required')
    fixed = {CPU_ENTRY: CPU_SHA, GPU_ENTRY: GPU_SHA, 'llm_intervention_service_runtime.py': RUNTIME_SHA,
             'llm_intervention_campaign.py': CAMPAIGN_SHA}
    need(all(files.get(n) == p for n, p in fixed.items()), 'accepted helper/source pins differ')
    for name in ('cpu-packet', 'gpu-packet', 'cpu-acceptance', 'gpu-acceptance'):
        need(files.get(name+'.json') == packet[name.replace('-', '_')+'_sha256'], 'upstream authority pin differs')
    data = {n: read(root/n, p) for n, p in files.items()}
    cpu = module('finalizer_cpu_helpers', root/CPU_ENTRY, data[CPU_ENTRY])
    gpu = module('finalizer_gpu_helpers', root/GPU_ENTRY, data[GPU_ENTRY])
    old, gp = decode(data['cpu-packet.json']), decode(data['gpu-packet.json'])
    need(all(files.get(n) == p for n, p in cpu.HELPER_PINS.items()), 'exact accepted runtime/helper pins required')
    diagnostic = module('finalizer_diagnostic', root/cpu.DIAGNOSTIC, data[cpu.DIAGNOSTIC])
    service = module('finalizer_service_contract', root/cpu.SERVICE, data[cpu.SERVICE])
    training = module('finalizer_environment', root/cpu.TRAINING, data[cpu.TRAINING])
    cpu.packet_shape(old, service, diagnostic, training)
    need(gp['schema'] == gpu.VERSION and gp['resources'] == gpu.RESOURCES
         and gp['cpu_packet_sha256'] == packet['cpu_packet_sha256']
         and gp['cpu_acceptance_sha256'] == packet['cpu_acceptance_sha256']
         and set(gp['files']) == set(old['files']) | gpu.EXTRA_SOURCES | {'cpu-packet.json', 'cpu-acceptance.json'}
         and all(gp['files'].get(n) == p for n, p in old['files'].items())
         and gp['source_names'] == sorted(set(old['source_names']) | gpu.EXTRA_SOURCES), 'accepted GPU source closure differs')
    need(set(files) == set(gp['files']) | {ENTRY, 'gpu-packet.json', 'gpu-acceptance.json'}
         and all(files.get(n) == p for n, p in gp['files'].items())
         and packet['source_names'] == sorted(set(gp['source_names']) | {ENTRY})
         and Path(__file__).absolute() == root/ENTRY, 'exact captured finalizer/source closure required')
    paths = packet['paths']
    need(all(paths[k] == v for k, v in gp['paths'].items()), 'original GPU paths differ')
    for key, name, previous in (('cpu_inputs', 'cpu-packet.json', old), ('gpu_inputs', 'gpu-packet.json', gp)):
        origin = Path(paths[key])
        need(read(origin/'packet.json', files[name]) == data[name], 'original upstream packet differs')
        for n, p in previous['files'].items():
            need(read(origin/n, p) == data[n], 'original upstream input changed: '+n)
    ca, ga, bundle = admit_evidence(packet, old, gp, data, gpu)
    return dict(root=root, packet=packet, data=data, cpu=cpu, api=gpu, old=old, diagnostic=diagnostic,
                training=training, cpu_acceptance=ca, gpu_acceptance=ga, bundle=bundle)


def allocation(text, job):
    pairs = re.findall(r'(\S+)=(\S+)', text)
    fields = dict(pairs)
    need(len(fields) == len(pairs) and re.fullmatch('[1-9][0-9]*', job), 'unique scheduler identity required')
    wanted = dict(JobId=job, JobState='RUNNING', Account='ucb736_asc2', Partition='acpu', QOS='cpu-normal',
                  NumCPUs='2', NumTasks='1', NumNodes='1', TimeLimit='00:05:00', Requeue='0', Restarts='0')
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
         and receipt['cell_count'] == 16 and receipt['scientific_acceptance'] is False
         and receipt['new_environment_responses'] == receipt['new_training_updates'] == 0
         and receipt['retry_allowed'] is False, 'campaign finalization receipt differs')
    need(summary['scope'] == 'artificial' and summary['cell_count'] == 16
         and [(r['world_id'], r['policy_id']) for r in summary['rows']] == list(itertools.product((9101, 9102), POLICIES))
         and [(p['left'], p['right']) for p in summary['policy_pairs']] == list(itertools.combinations(POLICIES, 2))
         and all(p['world_ids'] == [9101, 9102] for p in summary['policy_pairs'])
         and all(summary[k] is False for k in ('scientific_acceptance', 'scientific_protocol_frozen',
             'scientific_provenance_qualified', 'complete_case_filtering_allowed')), '16-cell/28-pair descriptive envelope differs')
    for i, row in enumerate(summary['rows']):
        if row['cell_status'] == 'reported':
            report = decode(read(root/f'private/world-{i//8:02d}/arm-{i%8:02d}/cost-report.json'))
            need(type(report['curve']) is list and 'curve_complete' in report, 'full curve evidence missing')
    return dict(cell_count=16, descriptive_policy_comparisons=28,
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
        save(out/'runtime-before.json', runtime_record(ctx, loader, stage, after=False))
        import llm_intervention_campaign as campaign
        save(out/'runtime-ready.json', runtime_record(ctx, loader, stage, after=True))
        need(time.monotonic() < handoff['started_monotonic']+STARTUP, 'finalizer startup deadline exceeded')
        save(out/'work-start.json', dict(packet_sha256=args.packet_sha256, utc=datetime.now(timezone.utc).isoformat(),
            monotonic=time.monotonic(), pid=os.getpid(), private_finalization_calls=1))
        signal.setitimer(signal.ITIMER_REAL, max(.001, handoff['started_monotonic']+WORK-time.monotonic()))
        bundle = ctx['bundle']
        # The existing engine gates all attempts/closures and claims its own
        # once-only intent. Never rebuild or relocate the completed campaign.
        campaign.finalize_campaign(campaign=bundle['campaign'], campaign_sha256=bundle['campaign_sha256'],
            source_plan_pins=bundle['source_plan_pins'], output=ctx['packet']['paths']['campaign'],
            attempt_pins=ctx['gpu_acceptance']['attempt_pins'])
        save(out/'runtime-after.json', runtime_record(ctx, loader, stage, after=True))
        report = validate_report(ctx)
        save(out/'terminal.json', dict(schema=VERSION, status='descriptive_finalization_complete',
            packet_sha256=args.packet_sha256, completed_utc=datetime.now(timezone.utc).isoformat(),
            report=report, retry_allowed=False, independent_acceptance_required=True))
    finally:
        sys.meta_path.remove(loader)


def supervise(command, env, out, started, packet_pin, kill_owned):
    """One owned process group; independent startup/work deadlines and cleanup."""
    proc, cleanup, ready, clean_first = None, None, False, False
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        with (out/'worker.stdout.log').open('xb') as stdout, (out/'worker.stderr.log').open('xb') as stderr:
            need(time.monotonic() < started+STARTUP, 'no startup budget remains')
            proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            save(out/'worker.started.json', dict(command=command, pid=proc.pid, pgid=proc.pid,
                started_monotonic=started, startup_deadline=started+STARTUP, work_deadline=started+WORK))
            while True:
                marker = out/'worker/work-start.json'
                if not ready and marker.exists():
                    value = decode(read(marker))
                    need(value['packet_sha256'] == packet_pin and value['pid'] == proc.pid
                         and started <= value['monotonic'] < started+STARTUP, 'worker startup receipt differs')
                    ready = True
                need(time.monotonic() < started+(WORK if ready else STARTUP), 'finalizer startup/work deadline exceeded')
                rc = proc.poll()
                if rc is not None:
                    need(rc == 0 and ready, 'finalizer worker failed or never began reporting')
                    break
                time.sleep(.05)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        old = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
        signal.setitimer(signal.ITIMER_REAL, max(.001, started+CLEANUP-time.monotonic()))
        try:
            if proc is not None:
                cleanup = kill_owned(proc)
                save(out/'worker.cleanup.json', cleanup)
                clean_first = (cleanup['group_still_exists'] is False and not cleanup['errors']
                               and cleanup['returncode'] == 0 and cleanup['kill_sent'] is False)
                # Only the live handle created here authorizes further cleanup.
                # Do not exit voluntarily with an unconfirmed owned group.
                index = 0
                while cleanup['group_still_exists']:
                    need(time.monotonic() < started+CLEANUP, 'owned cleanup deadline exceeded')
                    cleanup = kill_owned(proc)
                    save(out/f'cleanup-hold-{index:03d}.json', cleanup)
                    index += 1
                    if cleanup['group_still_exists']:
                        time.sleep(.05)
        finally:
            for s, handler in old.items():
                signal.signal(s, handler)
            signal.setitimer(signal.ITIMER_REAL, max(.001, started+HARD-time.monotonic()))
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            save(out/'worker.resources.json', dict(elapsed_seconds=time.monotonic()-started,
                child_and_cleanup_cpu_seconds=after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime,
                cumulative_child_peak_rss_bytes=after.ru_maxrss*(1 if sys.platform == 'darwin' else 1024)))
    need(clean_first and cleanup is not None and cleanup['group_still_exists'] is False and not cleanup['errors']
         and cleanup['returncode'] == 0 and cleanup['kill_sent'] is False, 'clean natural worker closure required')
    return cleanup


def parent(args, ctx):
    out, d = Path(args.output), ctx['diagnostic']
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
    stage = d.stage_runtime(ctx['old'], archive, local/'runtime', event)
    stage_pin = save(out/'stage.json', stage)
    event('stage_complete', staged_prefix=stage['staged_prefix'])
    need(time.monotonic() < args.started+STARTUP, 'staging consumed startup budget')
    handoff = dict(schema=VERSION, packet_sha256=args.packet_sha256, stage_sha256=stage_pin,
                   started_monotonic=args.started, parent_pid=os.getpid(), job_id=job)
    handoff_pin = save(out/'handoff.json', handoff)
    prefix = Path(stage['staged_prefix'])
    command = [str(prefix/'bin/python'), '-I', '-B', '-c', d.BOOTSTRAP, str(ctx['root']/ENTRY),
        ctx['packet']['files'][ENTRY], '--packet', str(Path(args.packet).absolute()), '--packet-sha256', args.packet_sha256,
        '--output', args.output, '--worker', '--handoff-sha256', handoff_pin]
    env = ctx['training'].cpu_environment(prefix, os.environ)
    signal.setitimer(signal.ITIMER_REAL, max(.001, args.started+WORK-time.monotonic()))
    cleanup = supervise(command, env, out, args.started, args.packet_sha256, probe.kill_owned)
    terminal = decode(read(out/'worker/terminal.json'))
    report = validate_report(ctx)
    need(terminal['schema'] == VERSION and terminal['status'] == 'descriptive_finalization_complete'
         and terminal['packet_sha256'] == args.packet_sha256 and terminal['report'] == report,
         'worker terminal differs from retained report')
    # Frozen upstream inventories must also remain unchanged after reporting.
    ca, ga, paths = ctx['cpu_acceptance'], ctx['gpu_acceptance'], ctx['packet']['paths']
    checked_inventory(ctx['api'], paths['cpu_output'], ca['cpu_output_files'], exclude='worker/artifacts/campaign')
    checked_inventory(ctx['api'], paths['gpu_output'], ga['gpu_output_files'])
    checked_inventory(ctx['api'], paths['gpu_qualification'], ga['qualification_files'])
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

    def interrupted(sig, frame):
        raise TimeoutError('finalizer signal '+str(sig))

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, STARTUP)
    try:
        _, packet = packet_destination(args.packet, args.packet_sha256, args.output)
        handoff = None
        if args.worker:
            need(pin(args.handoff_sha256), 'pinned worker handoff required')
            handoff = decode(read(Path(args.output)/'handoff.json', args.handoff_sha256))
            start = handoff['started_monotonic']
            need(handoff['schema'] == VERSION and handoff['packet_sha256'] == args.packet_sha256
                 and type(start) in (int, float) and math.isfinite(start)
                 and 0 <= args.started-start < STARTUP, 'original absolute startup clock required')
            args.started = start
            signal.setitimer(signal.ITIMER_REAL, max(.001, start+STARTUP-time.monotonic()))
        else:
            need(args.handoff_sha256 is None, 'parent cannot accept a runtime handoff')
        candidate = Path(args.output)/'worker' if args.worker else Path(args.output)
        candidate.mkdir(mode=0o700, exist_ok=False)
        claimed = candidate
        save(claimed/'start.json', dict(schema=VERSION, packet_sha256=args.packet_sha256, resources=RESOURCES,
            started_utc=datetime.now(timezone.utc).isoformat(), started_monotonic=args.started,
            deadlines=dict(startup=args.started+STARTUP, work=args.started+WORK,
                           cleanup=args.started+CLEANUP, hard=args.started+HARD), retry_allowed=False))
        ctx = authenticate(args.packet, args.packet_sha256, args.output)
        result = worker(args, ctx, handoff) if args.worker else parent(args, ctx)
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if claimed is not None:
            save(claimed/'failure.json', dict(schema=VERSION, status='terminal_failure', error=error,
                retry_allowed=False, scientific_acceptance=False,
                campaign_publication_authority='original campaign/finalization.json; reconcile separately if present'))
        raise
    finally:
        try:
            if claimed is not None:
                save(claimed/'attempt.json', dict(schema=VERSION, status='failed' if error else 'completed',
                    packet_sha256=args.packet_sha256, error=error, result=result,
                    completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-args.started,
                    process_cpu_seconds=time.process_time()-args.process_started, retry_allowed=False,
                    scientific_acceptance=False, independent_acceptance_required=True))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in previous.items():
                signal.signal(s, handler)


if __name__ == '__main__':
    main()

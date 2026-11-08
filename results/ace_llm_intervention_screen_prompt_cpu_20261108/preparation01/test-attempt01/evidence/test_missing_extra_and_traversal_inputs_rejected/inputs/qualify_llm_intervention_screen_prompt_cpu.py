"""Once-only CPU capacity measurement of the 120 frozen artificial fixtures.

Use the accepted diagnostic BOOTSTRAP with independent entry/packet SHA256s:
  <python> -I -B -c <BOOTSTRAP> <flat-inputs/this.py> <entry-sha256>
      --packet <flat-inputs/packet.json> --packet-sha256 <pin>
      --source-commit <independent-40-hex-commit> --output <new-absolute-directory>

Packet keys are PACKET_KEYS; source_names is the service SOURCE_NAMES plus this
entry. files contains those sources, HELPER_PINS, fixture-manifest.json and every
manifest envelope/prompt (flat names, exact pins). The source commit is external
provenance, not Git ancestry attestation; executed bytes use the existing captured
bootstrap/loader. No submission, installation, model construction or inference.
Overflow is a retained capacity rejection, never truncation or a study change.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import types

VERSION = 'ace-screen-prompt-cpu-capacity-v1'
ENTRY = 'qualify_llm_intervention_screen_prompt_cpu.py'
FINALIZER = 'finalize_llm_intervention_campaign_cpu.py'
CAMPAIGN = 'prepare_llm_intervention_campaign_cpu.py'
TRAINING = 'prepare_llm_intervention_training_cpu.py'
CPU = 'qualify_llm_intervention_service_cpu.py'
SERVICE = 'llm_intervention_model_service.py'
DIAGNOSTIC = 'qualify_llm_intervention_runtime_metadata.py'
HELPER_PINS = {
    FINALIZER: '88102baf56395e7ef06fc7e96aae3ed2d7b8eef4ecb8b71f77acfc7e4b8e813f',
    CAMPAIGN: '850ca3d4d7ac4799f68d74bd2d0b1196d10809751a4bdf878633122fc6a9c2c4',
    TRAINING: 'b2b1425bcd010d5ddc5deeda0dc3bd89817c9fe4cae6c4cc6465ec9f7f998f43',
    CPU: 'd658bedd192d5c3e92f8fc22a052c67d6524df296623ef74e2c5f38bf02fbeb5',
    SERVICE: 'a7735dd35f56d7653a5ff42520db7fd67b98395e62333cccce31583317f7134e',
    DIAGNOSTIC: '31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329',
    'llm_intervention_service_runtime.py': '117dc22bd56889906612bff0c975095918e062974fa537757b3e93c981dbfbc0',
    'llm_intervention_runtime_archive.py': '7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e',
    'probe.py': '8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951',
}
FIXTURE_MANIFEST_SHA = '73ff4ca7b441cd37c2e59ea75b35ba096fede360cc60d81f7843642f98fe493e'
PACKET_KEYS = {'schema', 'scope', 'request_mode', 'source_commit', 'files', 'source_names',
    'fixture_manifest_sha256', 'archive', 'manifest', 'runtime_source', 'runtime',
    'model_directory', 'model_files', 'resources', 'output'}
FLAGS = dict(model_weights_loaded=False, explicit_cuda_api_called=False, inference_calls=0,
    training_updates=0, environment_responses=0, scientific_qualified=False,
    training_execution_qualified=False, universal_context_bound=False, retry_allowed=False)


def _base():
    # Sole bootstrap read; all subsequent capture/writes reuse the pinned helper.
    path = Path(__file__).with_name(FINALIZER)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as handle:
        value = handle.read(1024**2+1)
    if len(value) > 1024**2 or hashlib.sha256(value).hexdigest() != HELPER_PINS[FINALIZER]:
        raise ValueError('independent finalizer helper pin differs')
    result = types.ModuleType('screen_cpu_base')
    result.__file__ = str(path)
    exec(compile(value, str(path), 'exec'), vars(result))
    return result


B = _base()
RESOURCES = B.RESOURCES
STARTUP, WORK, CLEANUP, HARD = B.STARTUP, B.WORK, B.CLEANUP, B.HARD


class Deadline(BaseException):
    """Must escape per-fixture error preservation and reach owned cleanup."""


def fixture_inputs(packet, data):
    manifest = B.decode(data['fixture-manifest.json'])
    B.need(B.sha(data['fixture-manifest.json']) == FIXTURE_MANIFEST_SHA
        and packet['fixture_manifest_sha256'] == FIXTURE_MANIFEST_SHA,
        'fixed 120-fixture manifest pin differs')
    rows = manifest['fixtures']
    B.need(manifest['version'] == 'ace-screen-artificial-prompt-fixtures-v1'
        and manifest['actual_tokenization_performed'] is False
        and manifest['scientific_execution_authorized'] is False
        and len(rows) == len({r['fixture_id'] for r in rows}) == 120, 'complete artificial roster required')
    names = {'fixture-manifest.json'}
    for row in rows:
        for field in ('envelope_file', 'prompt_file'):
            item = row[field]
            name = item['path']
            B.need(type(name) is str and Path(name).name == name and name not in names
                and packet['files'].get(name) == item['sha256'], 'unique flat fixture binding required')
            value = data[name]
            B.need(B.sha(value) == item['sha256'] and len(value) == item['utf8_bytes'], 'fixture bytes differ')
            names.add(name)
        envelope = B.decode(data[row['envelope_file']['path']])
        prompt = data[row['prompt_file']['path']]
        B.need(B.sha(B.raw(envelope)[:-1]) == row['envelope_sha256']
            and B.sha(prompt) == row['prompt_sha256'] and len(prompt) == row['prompt_utf8_bytes']
            and len(B.raw(envelope))-1 == row['envelope_utf8_bytes'], 'canonical fixture binding differs')
    return rows, names


def authenticate(args):
    root = Path(args.packet).absolute().parent
    packet = B.decode(B.read(args.packet, args.packet_sha256, 256*1024))
    B.need(type(packet) is dict and set(packet) == PACKET_KEYS and packet['schema'] == VERSION
        and packet['scope'] == 'artificial' and packet['request_mode'] == 'all_120_frozen_fixtures'
        and B.raw(packet['resources']) == B.raw(RESOURCES)
        and re.fullmatch('[0-9a-f]{40}', args.source_commit)
        and packet['source_commit'] == args.source_commit, 'fixed capacity-only request required')
    files = packet['files']
    B.need(type(files) is dict and 241 < len(files) < 320
        and all(type(n) is str and Path(n).name == n and n.endswith(('.py', '.json', '.txt'))
                and B.pin(p) for n, p in files.items())
        and all(files.get(n) == p for n, p in HELPER_PINS.items()), 'flat pinned source/input roster required')
    data = {n: B.read(root/n, p, 8*1024**2) for n, p in files.items()}
    helpers = {role: B.module('screen_'+role, root/name, data[name]) for role, name in
        (('service', SERVICE), ('cpu', CPU), ('diagnostic', DIAGNOSTIC), ('training', TRAINING), ('campaign', CAMPAIGN))}
    service, d = helpers['service'], helpers['diagnostic']
    rows, fixture_names = fixture_inputs(packet, data)
    B.need(packet['source_names'] == sorted(service.SOURCE_NAMES | {ENTRY})
        and set(files) == set(packet['source_names']) | set(HELPER_PINS) | fixture_names
        and files['llm_intervention_gpu_mappings.py'] == service.RUNTIME['gpu_mapping_source_sha256']
        and packet['runtime'] == service.RUNTIME and packet['runtime_source'] == d.RUNTIME_SOURCE
        and packet['model_files'] == helpers['campaign'].MODEL_FILES, 'exact source/model/runtime closure differs')
    for key, expected in (('archive', d.ARCHIVE_SHA), ('manifest', d.MANIFEST_SHA)):
        item = packet[key]
        B.need(type(item) is dict and set(item) == {'path', 'sha256'} and item['sha256'] == expected
            and B.absolute(item['path']), 'original archive/manifest required')
    B.need(B.absolute(args.packet) and B.absolute(args.output) and args.output == packet['output']
        and B.absolute(packet['model_directory']) and Path(__file__).absolute() == root/ENTRY
        and not root.is_symlink() and not Path(args.output).is_symlink(), 'captured entry/output differs')
    for protected in (root, packet['model_directory'], packet['runtime_source'],
                      packet['archive']['path'], packet['manifest']['path']):
        B.separate([args.output, protected])
    return dict(root=root, packet=packet, packet_path=args.packet, packet_sha256=args.packet_sha256,
        manifest_pins=dict(fixture_manifest_file_sha256=FIXTURE_MANIFEST_SHA,
            fixture_manifest_canonical_sha256=B.sha(B.raw(B.decode(data['fixture-manifest.json']))[:-1])),
        data=data, rows=rows, **helpers)


def check_inputs(ctx, *, archives=False):
    """Opaque streaming weight hash only; never deserialize or construct a model."""
    p, service = ctx['packet'], ctx['service']
    B.need(B.decode(B.read(ctx['packet_path'], ctx['packet_sha256'])) == p, 'request packet changed')
    for name, value in ctx['data'].items():
        B.need(B.read(ctx['root']/name, p['files'][name], 8*1024**2) == value, 'captured input changed')
    model = Path(p['model_directory'])
    B.need(not model.is_symlink() and model.is_dir() and {x.name for x in model.iterdir()} == service.MODEL_NAMES,
           'exact model/tokenizer directory required')
    models = {n: service.read_pinned(model/n, pin) for n, pin in p['model_files'].items()}
    if archives:
        for name in ('archive', 'manifest'):
            service.read_pinned(p[name]['path'], p[name]['sha256'])
    return models


def no_accelerators():
    names = ('SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'SLURM_GPUS', 'CUDA_VISIBLE_DEVICES',
             'NVIDIA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')
    environment = {n: os.environ.get(n, '') for n in names}
    devices = sorted({p for pattern in ('/dev/nvidia[0-9]*', '/dev/kfd', '/dev/dri/renderD*', '/dev/accel/accel*')
                      for p in glob.glob(pattern)})
    B.need(not any(environment.values()) and not devices, 'CPU allocation exposes accelerator environment/devices')
    return dict(environment=environment, devices=devices)


def cpu_controls(prefix, observed=None):
    controls = dict(CUDA_VISIBLE_DEVICES='', LD_LIBRARY_PATH=str(prefix/'lib'),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1', MKL_THREADING_LAYER='GNU')
    environment = os.environ if observed is None else observed
    B.need((observed is None or set(observed) == set(controls))
        and all(environment.get(k) == v for k, v in controls.items())
        and not any(environment.get(k) for k in ('PYTHONHOME', 'PYTHONPATH', 'LD_PRELOAD', 'MKL_SERVICE_FORCE_INTEL')),
        'fixed CPU thread/offline/library controls differ')
    return controls


class TokenCapture:
    """Record helper-returned tokens even when its final context guard raises."""
    def __init__(self, tokenizer, prompt):
        self.tokenizer, self.prompt = tokenizer, prompt
        self.eos_token_id, self.pad_token_id = tokenizer.eos_token_id, tokenizer.pad_token_id
        self.prompt_ids, self.commands = None, {}

    def apply_chat_template(self, messages, **kwargs):
        B.need(messages == [{'role': 'user', 'content': self.prompt}] and kwargs == dict(
            tokenize=True, add_generation_prompt=True, enable_thinking=False), 'shared chat formatting differs')
        # Explicit False is the existing template API default; never truncate.
        ids = self.tokenizer.apply_chat_template(messages, truncation=False, **kwargs)
        self.prompt_ids = list(ids)
        return ids

    def encode(self, text, **kwargs):
        B.need(kwargs == {'add_special_tokens': False} and text not in self.commands, 'command encoding differs')
        ids = self.tokenizer.encode(text, **kwargs)
        self.commands[text] = list(ids)+[self.eos_token_id]
        return ids

    def decode(self, ids, **kwargs):
        B.need(kwargs == {'skip_special_tokens': False}, 'command decoding differs')
        return self.tokenizer.decode(ids, **kwargs)


def validate_tokens(record, row):
    prompt, commands = record['prompt_ids'], record['commands']
    B.need(type(prompt) is list and prompt and all(type(t) is int and t >= 0 for t in prompt)
        and list(commands) == row['legal_commands'] and len({tuple(v) for v in commands.values()}) == len(commands),
        'exact prompt/ordered legal-command roster differs')
    for ids in commands.values():
        B.need(type(ids) is list and len(ids) >= 2 and ids[-1] == 151645 and 151645 not in ids[:-1]
            and all(type(t) is int and t >= 0 for t in ids), 'complete command plus one EOS required')
    return len(prompt)+32 > 8192 or max(map(len, commands.values())) > 32


def validate_metadata(record, row, envelope):
    meta = record['metadata']
    expected = dict(version='ace-shared-task-language-policy-v1', envelope_sha256=row['envelope_sha256'],
        prompt_sha256=row['prompt_sha256'], task_sha256=B.sha(B.raw(envelope['task'])[:-1]),
        prompt_tokens=len(record['prompt_ids']), longest_command_tokens=max(map(len, record['commands'].values())),
        max_tokens=8192, max_new_tokens=32, enable_thinking=False, legal_commands=len(record['commands']))
    B.need(B.raw(meta) == B.raw(expected), 'shared helper metadata differs')


def tokenize_all(ctx, tokenizer, prepare, prompt_api, journal, deadline):
    """All records retained, including errors/overflows; no fallback or retry."""
    B.need(tokenizer.eos_token_id == 151645 and tokenizer.pad_token_id == 151643, 'fixed special token IDs required')
    rows = ctx['rows']
    records = [dict(fixture_id=r['fixture_id'], status='not_attempted', prompt_ids=None, commands={}) for r in rows]
    journal.save('requested-fixtures.json', dict(rows=rows, request_mode='all_120_frozen_fixtures', **FLAGS))
    try:
        for index, row in enumerate(rows):
            if time.monotonic() >= deadline:
                raise Deadline('absolute CPU work deadline exceeded')
            result = records[index]
            result.update(status='interrupted', started_utc=datetime.now(timezone.utc).isoformat())
            capture = TokenCapture(tokenizer, ctx['data'][row['prompt_file']['path']].decode('utf-8'))
            try:
                envelope = B.decode(ctx['data'][row['envelope_file']['path']])
                B.need(prompt_api(envelope) == capture.prompt, 'actual task prompt differs from frozen fixture')
                grammar, metadata = prepare(envelope, capture, max_tokens=8192, max_new_tokens=32)
                result.update(prompt_ids=capture.prompt_ids, commands=capture.commands, metadata=metadata)
                B.need(not validate_tokens(result, row)
                    and list(grammar.prompt_ids) == capture.prompt_ids
                    and [list(x) for x in grammar.completion_ids] == list(capture.commands.values()),
                    'shared helper result/metadata differs')
                validate_metadata(result, row, envelope)
                for command, ids in capture.commands.items():
                    B.need(tokenizer.decode(ids[:-1], skip_special_tokens=False) == command, 'command round trip differs')
                result['status'] = 'fits'
            except Exception as exc:
                result.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
                result.update(prompt_ids=capture.prompt_ids, commands=capture.commands)
                if type(exc) is ValueError and str(exc) == 'public task or complete action exceeds frozen token limits':
                    try:
                        if validate_tokens(result, row):
                            result['status'] = 'overflow'
                    except ValueError:
                        pass  # Malformed/incomplete tokens remain a failure, not an overflow.
            finally:
                result.update(prompt_ids=capture.prompt_ids, commands=capture.commands,
                              completed_utc=datetime.now(timezone.utc).isoformat())
                journal.save(f'fixture-{index:03d}.json', result)
    finally:
        journal.save('tokenization.json', dict(rows=records, **FLAGS))
    return dict(fixtures=len(records), fits=sum(r['status'] == 'fits' for r in records),
        overflows=sum(r['status'] == 'overflow' for r in records), failures=sum(r['status'] == 'failed' for r in records),
        capacity_qualified=all(r['status'] == 'fits' for r in records))


def worker(args, ctx, handoff):
    out, service = Path(args.output)/'worker', ctx['service']
    B.need(handoff['parent_pid'] == os.getppid() and handoff['job_id'] == os.environ.get('SLURM_JOB_ID')
        and os.getpgrp() == os.getpid(), 'owned worker identity differs')
    journal = service.Journal(out/'artifacts')
    captured = {n: ctx['data'][n] for n in ctx['packet']['source_names']}
    loader = ctx['training'].CapturedSources({n: v for n, v in captured.items() if n != ENTRY}, ctx['root'])
    sys.meta_path.insert(0, loader)
    try:
        from llm_intervention_service_runtime import ServiceRuntime
        observer = ServiceRuntime(ctx['packet'], captured, loader, ctx['root']/ENTRY, globals(),
            Path(args.output)/'stage.json', handoff['stage_sha256'], journal, service.read_pinned,
            service.decode, allow_gpu_mappings=False)
        journal.save('inputs-before.json', dict(models=check_inputs(ctx), accelerators=no_accelerators(),
                                               controls=cpu_controls(observer.prefix)))
        observer.import_cpu_closure()
        from transformers import AutoTokenizer
        from llm_intervention_language_policy import prepare_task_tokens
        from llm_intervention_task import task_prompt
        journal.event('cpu_tokenizer_load_start')
        tokenizer = AutoTokenizer.from_pretrained(ctx['packet']['model_directory'], local_files_only=True, trust_remote_code=False)
        B.need(time.monotonic() < args.started+STARTUP, 'tokenizer startup deadline exceeded')
        B.save(out/'work-start.json', dict(packet_sha256=args.packet_sha256, pid=os.getpid(), monotonic=time.monotonic()))
        signal.setitimer(signal.ITIMER_REAL, max(.001, args.started+WORK-time.monotonic()))
        summary = tokenize_all(ctx, tokenizer, prepare_task_tokens, task_prompt, journal, args.started+WORK)
        journal.event('cpu_tokenization_complete', **summary)
        observer.observe('after-cpu-tokenization')
        journal.save('inputs-after.json', dict(models=check_inputs(ctx, archives=True), accelerators=no_accelerators(),
                                              controls=cpu_controls(observer.prefix)))
        journal.save('terminal.json', dict(schema=VERSION, packet_sha256=args.packet_sha256,
            stage_sha256=handoff['stage_sha256'], source_commit=args.source_commit,
            tokenization_sha256=B.sha(B.read(journal.output/'tokenization.json')), **ctx['manifest_pins'], **summary, **FLAGS))
    finally:
        sys.meta_path.remove(loader)


def validate_worker(ctx, args, stage, stage_pin, provenance):
    out = Path(args.output)/'worker/artifacts'
    read = lambda n: B.decode(B.read(out/n))
    checks = B.module('screen_runtime_checks', ctx['root']/'llm_intervention_service_runtime.py', ctx['data']['llm_intervention_service_runtime.py'])
    ctx['cpu'].validate_runtime_records(read, stage, stage_pin, ctx['service'], ctx['diagnostic'], checks, provenance)
    B.need(read('runtime-stage-binding.json')['gpu_mapping_enabled'] is False, 'GPU mapping exemption forbidden')
    expected = {Path(n).stem: ctx['packet']['files'][n] for n in ctx['packet']['source_names'] if n != ENTRY}
    for phase in ('after-service-imports', 'after-cpu-tokenization'):
        projects = read('runtime-'+phase+'.json')['captured_project_modules']
        B.need(set(expected) <= set(projects) <= set(expected) | {'__main__', '__mp_main__'}
            and all(projects[n]['sha256'] == p for n, p in expected.items()), 'captured runtime source closure differs')
    ctx['training'].validate_phases([read(p.name) for p in sorted(out.glob('phase-*.json'))])
    result, terminal = read('tokenization.json'), read('terminal.json')
    request = read('requested-fixtures.json')
    B.need(request['rows'] == ctx['rows'] and request['request_mode'] == 'all_120_frozen_fixtures', 'saved request roster differs')
    B.need([r['fixture_id'] for r in result['rows']] == [r['fixture_id'] for r in ctx['rows']]
        and all(B.raw(result.get(k)) == B.raw(v) == B.raw(terminal.get(k)) for k, v in FLAGS.items()), 'complete result scope differs')
    for i, (record, row) in enumerate(zip(result['rows'], ctx['rows'])):
        B.need(read(f'fixture-{i:03d}.json') == record and record['status'] in ('fits', 'overflow', 'failed'), 'incomplete fixture attempt')
        if record['status'] != 'failed':
            B.need(validate_tokens(record, row) == (record['status'] == 'overflow'), 'saved capacity classification differs')
        if record['status'] == 'fits':
            validate_metadata(record, row, B.decode(ctx['data'][row['envelope_file']['path']]))
        else:
            B.need(type(record.get('error')) is dict and record['error'].get('message'), 'failure evidence missing')
    B.need(terminal['schema'] == VERSION and terminal['packet_sha256'] == args.packet_sha256
        and terminal['stage_sha256'] == stage_pin and terminal['source_commit'] == args.source_commit
        and all(terminal.get(k) == v for k, v in ctx['manifest_pins'].items())
        and terminal['tokenization_sha256'] == B.sha(B.read(out/'tokenization.json'))
        and terminal['fixtures'] == 120 and terminal['fits'] == sum(r['status'] == 'fits' for r in result['rows'])
        and terminal['overflows'] == sum(r['status'] == 'overflow' for r in result['rows'])
        and terminal['failures'] == sum(r['status'] == 'failed' for r in result['rows'])
        and terminal['capacity_qualified'] is (terminal['fits'] == 120), 'terminal capacity counts differ')
    before = read('inputs-before.json')
    B.need(before == read('inputs-after.json') and set(before['models']) == set(ctx['packet']['model_files'])
        and all(r['sha256'] == ctx['packet']['model_files'][n] and type(r['bytes']) is int and r['bytes'] > 0
                for n, r in before['models'].items())
        and before['accelerators']['devices'] == [] and not any(before['accelerators']['environment'].values()),
        'model/input/device checks changed or differ from independent pins')
    cpu_controls(Path(stage['staged_prefix']), before['controls'])
    return terminal


def parent(args, ctx):
    out, d = Path(args.output), ctx['diagnostic']
    job = os.environ.get('SLURM_JOB_ID', '')
    query = subprocess.run(['scontrol', 'show', 'job', '-o', job], capture_output=True, text=True, timeout=10)
    B.save(out/'allocation.json', dict(stdout=query.stdout, stderr=query.stderr, returncode=query.returncode))
    B.need(query.returncode == 0, 'CPU allocation query failed')
    B.allocation(query.stdout, job)
    B.save(out/'accelerators.json', no_accelerators())
    B.save(out/'inputs-before.json', dict(models=check_inputs(ctx)))
    scratch = Path(os.environ.get('SLURM_SCRATCH', ''))
    B.need(scratch.is_absolute() and scratch.is_dir() and scratch.stat().st_uid == os.getuid(), 'owned scratch required')
    local = scratch.resolve()/('ace-screen-prompt-cpu-'+job)
    B.separate([local, args.output, ctx['root']])
    for protected in (ctx['packet']['model_directory'], ctx['packet']['runtime_source'],
                      ctx['packet']['archive']['path'], ctx['packet']['manifest']['path']):
        B.separate([local, protected])
    local.mkdir(mode=0o700, exist_ok=False)
    archive = B.module('screen_archive', ctx['root']/'llm_intervention_runtime_archive.py', ctx['data']['llm_intervention_runtime_archive.py'])
    probe = B.module('screen_cleanup', ctx['root']/'probe.py', ctx['data']['probe.py'])
    event = d.events(out, args.started, args.process_started, 7680*1024**2)
    stage = d.stage_runtime(ctx['packet'], archive, local/'runtime', event)
    stage_pin = B.save(out/'stage.json', stage)
    prefix = Path(stage['staged_prefix'])
    provenance = d.provenance_contract(prefix)
    B.save(out/'independent-provenance-sources.json', provenance)
    handoff = dict(schema=VERSION, packet_sha256=args.packet_sha256, stage_sha256=stage_pin,
                   started_monotonic=args.started, parent_pid=os.getpid(), job_id=job)
    handoff_pin = B.save(out/'handoff.json', handoff)
    command = [str(prefix/'bin/python'), '-I', '-B', '-c', d.BOOTSTRAP, str(ctx['root']/ENTRY),
        ctx['packet']['files'][ENTRY], '--packet', args.packet, '--packet-sha256', args.packet_sha256,
        '--source-commit', args.source_commit, '--output', args.output, '--worker', '--handoff-sha256', handoff_pin]
    # Reuse exactly the finalizer's 120/260/280/290 owned-process supervisor.
    signal.setitimer(signal.ITIMER_REAL, max(.001, args.started+WORK-time.monotonic()))
    cleanup = B.supervise(command, ctx['training'].cpu_environment(prefix, os.environ), out,
                          args.started, args.packet_sha256, probe.kill_owned)
    terminal = validate_worker(ctx, args, stage, stage_pin, provenance)
    models = check_inputs(ctx)
    B.need(B.decode(B.read(out/'inputs-before.json')) == dict(models=models), 'parent input checks changed')
    B.save(out/'inputs-after.json', dict(models=models))
    B.need(time.monotonic() < args.started+HARD, 'hard deadline exceeded')
    B.save(out/'terminal.json', dict(schema=VERSION, status='capacity_measured' if terminal['capacity_qualified']
        else 'capacity_rejected', packet_sha256=args.packet_sha256, source_commit=args.source_commit,
        worker_terminal_sha256=B.sha(B.read(out/'worker/artifacts/terminal.json')),
        cleanup=cleanup, capacity_qualified=terminal['capacity_qualified'], independent_acceptance_required=True,
        **ctx['manifest_pins'], **FLAGS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'packet-sha256', 'source-commit', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--handoff-sha256')
    args = parser.parse_args()
    args.started, args.process_started = time.monotonic(), time.process_time()
    claimed = None
    def interrupted(sig, frame):
        raise Deadline('CPU capacity signal '+str(sig))
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, STARTUP)
    try:
        B.need(sys.flags.isolated == 1 and sys.dont_write_bytecode, 'captured isolated -I -B bootstrap required')
        ctx = authenticate(args)
        handoff = None
        if args.worker:
            B.need(B.pin(args.handoff_sha256), 'independent handoff pin required')
            handoff = B.decode(B.read(Path(args.output)/'handoff.json', args.handoff_sha256))
            start = handoff['started_monotonic']
            B.need(handoff['schema'] == VERSION and handoff['packet_sha256'] == args.packet_sha256
                and type(start) in (int, float) and math.isfinite(start) and 0 <= args.started-start < STARTUP,
                'original startup clock required')
            args.started = start
            signal.setitimer(signal.ITIMER_REAL, max(.001, start+STARTUP-time.monotonic()))
        else:
            B.need(args.handoff_sha256 is None, 'parent handoff forbidden')
        out = Path(args.output)/'worker' if args.worker else Path(args.output)
        out.mkdir(mode=0o700, exist_ok=False)
        claimed = out
        B.save(out/'start.json', dict(schema=VERSION, packet_sha256=args.packet_sha256,
            source_commit=args.source_commit, resources=RESOURCES, started_utc=datetime.now(timezone.utc).isoformat(),
            started_monotonic=args.started, deadlines=dict(startup=args.started+STARTUP, work=args.started+WORK,
                cleanup=args.started+CLEANUP, hard=args.started+HARD), **FLAGS))
        worker(args, ctx, handoff) if args.worker else parent(args, ctx)
    except BaseException as exc:
        if claimed is not None:
            B.save(claimed/'failure.json', dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(), **FLAGS))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, handler in previous.items():
            signal.signal(s, handler)


if __name__ == '__main__':
    main()

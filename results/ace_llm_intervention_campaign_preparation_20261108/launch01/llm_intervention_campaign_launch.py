"""Stable campaign command bound to a later authenticated runtime stage.

The trusted GPU supervisor stages the pinned archive, writes one exclusive
binding and supplies its byte pin in ACE_CAMPAIGN_STAGE_BINDING_SHA256. The
campaign command and contract are fixed before CPU arm execution. This shim
never accepts an arbitrary command from the binding and never loads a model.
It is an execution consistency check, not a hostile same-user sandbox.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

VERSION = 'ace-campaign-staged-launch-v1'
BINDING_ENV = 'ACE_CAMPAIGN_STAGE_BINDING_SHA256'
MAX_BYTES = 2 * 1024 * 1024


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _keys(value, names):
    _need(type(value) is dict and set(value) == set(names), 'exact launch fields required')


def _raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _decode(raw):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            _need(name not in result, 'duplicate launch key')
            result[name] = value
        return result
    result = json.loads(raw, object_pairs_hook=unique)
    _raw(result)
    return result


def _absolute(value):
    return (type(value) is str and value.startswith('/') and '\x00' not in value
            and str(Path(value)) == value and '..' not in Path(value).parts)


def read_pinned(path, pin):
    _need(_pin(pin), 'independent byte pin required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        _need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_BYTES,
              'bounded regular launch input required')
        raw = handle.read(MAX_BYTES+1)
        after = os.fstat(handle.fileno())
    _need(len(raw) <= MAX_BYTES and _sha(raw) == pin and
          (before.st_ino, before.st_size, before.st_mtime_ns) ==
          (after.st_ino, after.st_size, after.st_mtime_ns), 'launch input pin or stability differs')
    return raw


def validate_contract(contract):
    _keys(contract, ('schema', 'policy_id', 'entry', 'entry_sha256', 'freeze',
        'freeze_sha256', 'worker_output', 'launch_output', 'binding_path',
        'stage_path', 'archive_sha256', 'manifest_sha256', 'bootstrap', 'bootstrap_sha256'))
    _need(contract['schema'] == VERSION and contract['policy_id'] in
          ('frozen_llm', 'sft_llm', 'dpo_llm'), 'unsupported campaign service')
    for key in ('entry', 'freeze', 'worker_output', 'launch_output', 'binding_path', 'stage_path'):
        _need(_absolute(contract[key]), 'normalized absolute launch path required: '+key)
    for key in ('entry_sha256', 'freeze_sha256', 'archive_sha256', 'manifest_sha256', 'bootstrap_sha256'):
        _need(_pin(contract[key]), 'invalid launch pin: '+key)
    _need(Path(contract['entry']).name == 'llm_intervention_model_service.py'
          and Path(contract['entry']).parent == Path(contract['freeze']).parent,
          'captured service entry and freeze must be adjacent')
    _need(type(contract['bootstrap']) is str and 0 < len(contract['bootstrap']) <= 32768
          and _sha(contract['bootstrap'].encode()) == contract['bootstrap_sha256'],
          'independently pinned bootstrap required')
    paths = [Path(contract[k]) for k in ('worker_output', 'launch_output')]
    _need(paths[0] != paths[1] and not any(a in b.parents for a, b in (paths, paths[::-1])),
          'worker and launch evidence roots must be separate')
    _need(all(not any(p == Path(contract[k]) or p in Path(contract[k]).parents
                         for p in paths)
              for k in ('entry', 'freeze', 'binding_path', 'stage_path')),
          'inputs must be outside new output roots')
    return _decode(_raw(contract))


def derive_command(contract, *, contract_sha256, binding, stage):
    """Rebuild exact argv from frozen paths and a separately pinned stage binding.

    The supervisor authenticates the actual archive extraction. Receipt text
    alone does not prove execution or byte integrity of the staged runtime.
    """
    c = validate_contract(contract)
    _need(_pin(contract_sha256), 'contract byte pin required')
    _keys(binding, ('schema', 'job_id', 'stage_path', 'stage_sha256', 'staged_prefix', 'contracts'))
    _need(binding['schema'] == VERSION and type(binding['job_id']) is str
          and re.fullmatch('[1-9][0-9]*', binding['job_id']) is not None,
          'actual scheduler identity required')
    _need(_absolute(binding['staged_prefix']) and binding['stage_path'] == c['stage_path']
          and _pin(binding['stage_sha256']), 'stage binding path or pin differs')
    _keys(binding['contracts'], ('frozen_llm', 'sft_llm', 'dpo_llm'))
    _need(all(_pin(p) for p in binding['contracts'].values())
          and binding['contracts'][c['policy_id']] == contract_sha256,
          'service contract is outside fixed stage binding')
    _need(type(stage) is dict and stage.get('version') == 'ace-runtime-archive-v1'
          and stage.get('archive_sha256') == c['archive_sha256']
          and stage.get('manifest_sha256') == c['manifest_sha256']
          and stage.get('staged_prefix') == binding['staged_prefix']
          and stage.get('objects') == 49966 and stage.get('file_bytes') == 8397575073
          and stage.get('model_loaded') is False and stage.get('runtime_qualified') is False,
          'authenticated archive staging receipt differs')
    prefix = Path(binding['staged_prefix'])
    _need(prefix != Path('/') and prefix not in Path(c['entry']).parents,
          'staged runtime must be separate from project source')
    return [str(prefix/'bin/python'), '-I', '-B', '-c', c['bootstrap'],
        c['entry'], c['entry_sha256'], '--freeze', c['freeze'],
        '--freeze-sha256', c['freeze_sha256'], '--output', c['worker_output'],
        '--runtime-stage', c['stage_path'], '--runtime-stage-sha256', binding['stage_sha256']]


def save(path, value):
    with path.open('xb') as handle:
        handle.write(_raw(value))
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True)
    parser.add_argument('--contract-sha256', required=True)
    args = parser.parse_args()
    contract = validate_contract(_decode(read_pinned(args.contract, args.contract_sha256)))
    binding_pin = os.environ.get(BINDING_ENV)
    binding = _decode(read_pinned(contract['binding_path'], binding_pin))
    stage = _decode(read_pinned(contract['stage_path'], binding['stage_sha256']))
    command = derive_command(contract, contract_sha256=args.contract_sha256, binding=binding, stage=stage)
    _need(os.environ.get('SLURM_JOB_ID') == binding['job_id'], 'actual Slurm job differs from stage binding')
    _need(os.environ.get('CUDA_VISIBLE_DEVICES') and os.environ.get('MKL_THREADING_LAYER') == 'GNU'
          and all(not os.environ.get(k) for k in ('MKL_SERVICE_FORCE_INTEL', 'PYTHONHOME', 'PYTHONPATH', 'LD_PRELOAD')),
          'explicit GPU child environment required')
    read_pinned(contract['entry'], contract['entry_sha256'])
    read_pinned(contract['freeze'], contract['freeze_sha256'])
    output = Path(contract['launch_output'])
    output.mkdir(exist_ok=False)
    # The immutable campaign records the shim argv. This record supplies the
    # exact derived exec argv without changing that campaign or its digest.
    save(output/'execution.json', {'schema': VERSION, 'utc': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid(), 'job_id': binding['job_id'], 'contract_sha256': args.contract_sha256,
        'stage_binding_sha256': binding_pin, 'stage_sha256': binding['stage_sha256'],
        'command': command, 'archive_execution_independently_required': True,
        'scientific_qualified': False})
    try:
        os.execv(command[0], command)
    except BaseException as exc:
        save(output/'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'retry_allowed': False})
        raise


if __name__ == '__main__':
    main()

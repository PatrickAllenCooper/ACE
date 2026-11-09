"""Read-only admission of independently accepted scientific training checkpoints.

The caller pins the actual independent acceptance and supplies its separately
frozen expected lineage. This module hashes opaque adapter bytes, never loads a
tensor, and never creates acceptance. Source/runtime/scheduler/custody review
must already have established the acceptance record. Service inference recipe,
development inputs and campaign execution require their own admission.
"""
from __future__ import annotations

import hashlib
import json
import re

VERSION = 'ace-scientific-campaign-training-lineage-v1'
EXPECTED_PINS = frozenset(('prepared_sha256', 'training_plan_sha256', 'exposure_sha256',
    'scientific_freeze_sha256', 'collection_acceptance_sha256',
    'cpu_preparation_acceptance_sha256', 'model_files_sha256', 'runtime_contract_sha256'))
WORLD_IDS = tuple(range(202611070000, 202611070032))
ROSTER = tuple(f'policy_train/{wid}/snapshot-{i:03d}' for wid in WORLD_IDS for i in range(5))


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _sha(raw):
    _need(type(raw) is bytes, 'captured bytes required')
    return hashlib.sha256(raw).hexdigest()


def _decode(raw):
    _need(type(raw) is bytes and len(raw) <= 256 * 1024**2, 'bounded captured JSON required')
    def unique(items):
        value = {}
        for key, item in items:
            _need(key not in value, 'duplicate lineage field')
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=unique)
    json.dumps(value, allow_nan=False)
    _need(type(value) is dict, 'lineage object required')
    return value


def _semantic(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def admit_training(*, acceptance_raw, acceptance_sha256, expected, prepared_raw,
                   engine_terminal_raw, sft_checkpoint_raw, sft_adapter_raw,
                   final_checkpoint_raw, final_adapter_raw):
    """Return SFT/DPO service roles from pinned, accepted policy_train artifacts.

    `expected` is supplied by the external coordinator from the frozen scientific
    plan, accepted CPU preparation and reconciled submission. It is not derived
    from the acceptance being checked. The independent audit must emit schema
    ace-scientific-training-gpu-acceptance-v1 with the fields checked below.
    An exit code, worker summary or an artificial acceptance is insufficient.
    """
    _need(_pin(acceptance_sha256) and _sha(acceptance_raw) == acceptance_sha256,
          'independent scientific training acceptance pin differs')
    _need(type(expected) is dict and set(expected) == EXPECTED_PINS | {'job_id', 'source_commit'}
          and all(_pin(expected[k]) for k in EXPECTED_PINS)
          and type(expected['job_id']) is str and re.fullmatch('[1-9][0-9]*', expected['job_id'])
          and type(expected['source_commit']) is str and re.fullmatch('[0-9a-f]{40}', expected['source_commit']),
          'complete independently supplied training lineage required')
    a = _decode(acceptance_raw)
    _need(a.get('schema') == 'ace-scientific-training-gpu-acceptance-v1'
          and a.get('status') == 'scientific_training_gpu_accepted'
          and a.get('scope') == 'policy_train'
          and type(a.get('required_findings')) is int and a['required_findings'] == 0
          and a.get('scientific_utility_qualified') is False
          and all(a.get(k) is True for k in ('training_gpu_qualified',
              'checkpoint_reload_inference_qualified', 'complete_custody_verified',
              'runtime_qualified', 'owned_process_cleanup_verified', 'scheduler_verified'))
          and all(a.get(k) == v for k, v in expected.items()),
          'independent complete scientific training acceptance required')
    _need(_sha(prepared_raw) == expected['prepared_sha256']
          and _pin(a.get('engine_terminal_sha256'))
          and _sha(engine_terminal_raw) == a['engine_terminal_sha256'], 'accepted input or terminal differs')
    p, terminal = _decode(prepared_raw), _decode(engine_terminal_raw)
    plan, exposure = p['training_plan'], p['exposure']
    _need(p.get('version') == 'ace-complete-collection-training-plan-v1'
          and p['training_plan_sha256'] == _semantic(plan) == expected['training_plan_sha256']
          and p['exposure_sha256'] == _semantic(exposure) == expected['exposure_sha256']
          and plan['scope'] == exposure['scope'] == 'policy_train'
          and exposure['record_roster'] == list(ROSTER)
          and [r['record_id'] for r in plan['records']] == list(ROSTER),
          'accepted complete scientific batch or exposure differs')
    d = exposure['counts']['dpo_steps']
    _need(type(d) is int and 1 <= d <= 128 and exposure['has_both_training_stages'] is True
          and plan['execution_kind'] == 'qwen_screen_singletons'
          and type(plan['sft_steps']) is int and plan['sft_steps'] == 160
          and type(plan['dpo_steps']) is int and plan['dpo_steps'] == d
          and plan['checkpoint_steps'] == [159, 159+d], 'fixed training schedule differs')
    counts = {'completed_steps':160+d, 'sft_steps':160, 'dpo_steps':d, 'environment_responses':0}
    _need(terminal.get('status') == 'complete'
          and terminal.get('execution_kind') == 'qwen_screen_singletons'
          and terminal.get('plan_sha256') == expected['training_plan_sha256']
          and all(type(terminal.get(k)) is int and terminal[k] == v for k, v in counts.items())
          and all(type(a.get(k)) is int and a[k] == v for k, v in counts.items()),
          'accepted complete schedule or zero-response GPU stage differs')
    roles = {}
    for role, stage, treatment, count, checkpoint_raw, adapter_raw in (
            ('sft_llm', 'sft', 'sft', 160, sft_checkpoint_raw, sft_adapter_raw),
            ('dpo_llm', 'final', 'dpo', 160+d, final_checkpoint_raw, final_adapter_raw)):
        cp_pin, adapter_pin = _sha(checkpoint_raw), _sha(adapter_raw)
        _need(_pin(a.get(stage+'_checkpoint_sha256')) and _pin(a.get(stage+'_adapter_sha256'))
              and cp_pin == a[stage+'_checkpoint_sha256']
              and adapter_pin == a[stage+'_adapter_sha256'], 'accepted checkpoint or adapter differs')
        cp = _decode(checkpoint_raw)
        _need(cp['files']['adapter'] == adapter_pin
              and type(cp['completed_steps']) is int and cp['completed_steps'] == count
              and type(cp['attempt']['index']) is int and cp['attempt']['index'] == count-1
              and cp['attempt']['stage'] == treatment
              and cp['attempt']['optimizer_step_started'] is True
              and cp['attempt']['optimizer_step_returned'] is True,
              'checkpoint role or completed optimizer step differs')
        roles[role] = {'treatment':treatment, 'training_scope':'policy_train',
            'adapter_sha256':adapter_pin, 'checkpoint_sha256':cp_pin, 'completed_steps':count,
            'accepted_training_job_id':expected['job_id']}
    return {'schema':VERSION, 'acceptance_sha256':acceptance_sha256, **expected,
        'engine_terminal_sha256':a['engine_terminal_sha256'], 'roles':roles,
        'new_training_updates':0, 'environment_responses':0, 'tensor_files_deserialized':False,
        'development_execution_qualified':False, 'scientific_utility_qualified':False}

"""Read-only role admission for the accepted artificial GPU03 checkpoints.

Authenticate the independent acceptance pin, its exact engine terminal and the
same prepared batch before constructing service roles. Checkpoint receipt pins
are distinct from outer acceptance pins. No tensors or model code are loaded.
The original accepted artifacts and their false inner provenance flags remain
unchanged. This narrow admission does not qualify a future campaign execution.
"""
from __future__ import annotations

import hashlib
import json

VERSION = 'ace-artificial-campaign-training-lineage-v1'
ACCEPTANCE_SHA256 = '4f49ed2e09da27f87b02ca3e20fc835ae5a9b5eeeeb26ae1039421375ed0a3d7'
PREPARED_SHA256 = 'ceb5847336cf664b215a97d0c8f56f86f2f696e02e44f684ec5c9277e364653e'
JOB_ID = '35379434'
CHECKPOINT_PINS = {
    'sft': 'feebd8800d9405a1cf0ccbb58a38fe301a3f58fc5b6741e777824008ce067680',
    'final': 'd3727084d31ff0ee785283d7432acca0f2fac89f66dd8288c28493737180e56c'}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    _need(type(raw) is bytes, 'captured bytes required')
    return hashlib.sha256(raw).hexdigest()


def _decode(raw):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            _need(k not in result, 'duplicate lineage field')
            result[k] = v
        return result
    result = json.loads(raw, object_pairs_hook=unique)
    json.dumps(result, allow_nan=False)
    _need(type(result) is dict, 'lineage object required')
    return result


def _semantic(value):
    return _sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def admit_training(*, acceptance_raw, prepared_raw, engine_terminal_raw,
                   sft_checkpoint_raw, sft_adapter_raw, final_checkpoint_raw, final_adapter_raw):
    """Return two exact artificial service adapter-role bindings and batch pins.

    Pin constants identify the already accepted finite execution. A future
    scientific source must have its own reviewed admission, never relabel this.
    Input file reading/capture and source/runtime execution belong to the outer
    CPU driver. Adapter bytes are hashed opaquely, never deserialized here.
    """
    _need(_sha(acceptance_raw) == ACCEPTANCE_SHA256, 'independent GPU03 acceptance pin differs')
    _need(_sha(prepared_raw) == PREPARED_SHA256, 'accepted prepared batch differs')
    a, prepared = _decode(acceptance_raw), _decode(prepared_raw)
    _need(a.get('status') == 'artificial_integrated_training_gpu_accepted'
          and a.get('job_id') == JOB_ID and a.get('training_gpu_qualified') is True
          and a.get('checkpoint_reload_inference_qualified') is True
          and a.get('scientific_qualified') is False and a.get('required_remaining') == 0,
          'accepted artificial training scope differs')
    _need(_sha(engine_terminal_raw) == a['engine_terminal_sha256'], 'accepted engine terminal differs')
    terminal = _decode(engine_terminal_raw)
    _need(terminal.get('status') == 'complete' and terminal.get('completed_steps') == 6
          and terminal.get('sft_steps') == 4 and terminal.get('dpo_steps') == 2
          and terminal.get('environment_responses') == 0
          and terminal.get('plan_sha256') == a['training_plan_sha256'],
          'accepted completed training schedule differs')
    _need(prepared['training_plan_sha256'] == a['training_plan_sha256']
          and _semantic(prepared['training_plan']) == a['training_plan_sha256']
          and _semantic(prepared['exposure']) == a['exposure_sha256']
          and prepared['training_plan']['scope'] == 'artificial', 'complete batch or exposure lineage differs')
    roles = {}
    for role, stage, count, checkpoint_raw, adapter_raw in (
            ('sft_llm','sft',4,sft_checkpoint_raw,sft_adapter_raw),
            ('dpo_llm','final',6,final_checkpoint_raw,final_adapter_raw)):
        _need(_sha(checkpoint_raw) == CHECKPOINT_PINS[stage], 'original checkpoint receipt pin differs')
        checkpoint = _decode(checkpoint_raw)
        adapter_pin = _sha(adapter_raw)
        _need(adapter_pin == a[stage+'_adapter_sha256']
              and checkpoint['files']['adapter'] == adapter_pin
              and checkpoint['completed_steps'] == count
              and checkpoint['attempt']['index'] == count-1
              and checkpoint['attempt']['stage'] == ('sft' if stage == 'sft' else 'dpo')
              and checkpoint['attempt']['optimizer_step_started'] is True
              and checkpoint['attempt']['optimizer_step_returned'] is True,
              'checkpoint role, update count or adapter identity differs')
        # These exact receipt pins come from accepted GPU03 custody and reloads.
        roles[role] = {'treatment': 'artificial_adapter', 'training_scope': 'artificial',
            'adapter_sha256': adapter_pin, 'checkpoint_sha256': _sha(checkpoint_raw),
            'completed_steps': count, 'accepted_training_job_id': JOB_ID}
    return {'schema': VERSION, 'acceptance_sha256': ACCEPTANCE_SHA256,
        'prepared_sha256': PREPARED_SHA256, 'engine_terminal_sha256': a['engine_terminal_sha256'],
        'training_plan_sha256': a['training_plan_sha256'], 'exposure_sha256': a['exposure_sha256'],
        'roles': roles, 'new_training_updates': 0, 'environment_responses': 0,
        'scientific_qualified': False}

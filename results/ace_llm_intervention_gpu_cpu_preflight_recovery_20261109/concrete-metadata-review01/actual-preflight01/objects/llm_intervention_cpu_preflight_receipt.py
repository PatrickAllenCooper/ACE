"""Pure pinned CPU-preflight metadata consistency. No I/O, clock reads or imports of producers.

Caller independently authenticates both SHA arguments. This does not establish
actual file verification, continuous immutability, GPU work or final acceptance.
"""
import datetime
import hashlib
import json
from pathlib import PurePosixPath
import re

MAX_JSON_BYTES = 128 * 1024**2
READ_SET_MAX_BYTES = 16 * 1024**2
EXCLUDED = 'worker/artifacts/campaign'
ROOT_KEYS = {'cpu_output', 'campaign'}
LINEAGE_KEYS = {'cpu_acceptance_sha256', 'failed_gpu01_acceptance_sha256',
                'recovery_authority_sha256', 'source_sha256', 'checker_sha256',
                'namespace_sha256', 'original_namespace_sha256'}
PROOF_KEYS = {'custody_sha256', 'before_observation_sha256',
              'after_observation_sha256', 'read_set_artifact_sha256'}
CONTEXT_KEYS = {'schema', 'control_root', 'gpu_output', 'failed_gpu01_root',
                'cpu_roots', 'lineage', 'baseline', 'proof_pins', 'source_files_sha256',
                'model_files_sha256', 'resources_sha256', 'limits_sha256',
                'cpu_inventory_sha256', 'consumed_read_set_sha256'}
EXPECTED_KEYS = {'schema', 'execution_context', 'execution_context_sha256',
                 'cpu_inventories', 'consumed_read_set', 'receipt_not_after_utc'}
RECEIPT_KEYS = {'schema', 'phase', 'execution_context_sha256', 'inventory_before',
                'inventory_after', 'consumed_read_set', 'absent_paths',
                'started_utc', 'finished_utc'}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _keys(value, keys, role):
    _need(type(value) is dict and set(value) == keys, 'exact '+role+' keys required')


def _pin(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _decode(data, expected, role):
    _need(type(data) is bytes and 0 < len(data) <= MAX_JSON_BYTES, 'bounded exact '+role+' bytes required')
    _need(_pin(expected) and hashlib.sha256(data).hexdigest() == expected, 'independent '+role+' byte pin differs')
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite JSON number forbidden')
    # Decode as UTF-8 explicitly: no alternative encodings or mutable object API.
    return json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=nonfinite)


def _path(value, absolute=False):
    _need(type(value) is str and value and '\\' not in value
          and not any(ord(c) < 32 or ord(c) == 127 for c in value), 'ordinary path spelling required')
    path = PurePosixPath(value)
    _need(str(path) == value and path.is_absolute() is absolute and value != '/'
          and all(p not in ('.', '..', '') for p in value.split('/')[1 if absolute else 0:]),
          'canonical absolute/relative path required')
    return path


def _utc(value):
    _need(type(value) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z', value),
          'explicit finite UTC timestamp ending Z required')
    return datetime.datetime.fromisoformat(value[:-1] + '+00:00')


def _file_info(info):
    _keys(info, {'bytes', 'sha256'}, 'file information')
    _need(type(info['bytes']) is int and info['bytes'] >= 0 and _pin(info['sha256']), 'exact nonnegative file bytes/SHA required')


def _inventory(value):
    _keys(value, {'files', 'directories'}, 'inventory')
    files, dirs = value['files'], value['directories']
    _need(type(files) is dict and type(dirs) is list and all(type(n) is str for n in dirs), 'original maps/list required')
    directory_set = set(dirs)
    _need(dirs == sorted(directory_set) and not set(files).intersection(directory_set), 'unique sorted noncolliding directory roster required')
    for name in list(files) + dirs:
        _path(name)
        _need(all('/'.join(name.split('/')[:i]) in directory_set for i in range(1, len(name.split('/')))), 'missing parent directory')
    for info in files.values():
        _file_info(info)


def _read_names():
    names = {'campaign.json', 'source-plan-pins.json', 'start.json'}
    for policy in range(5):
        prefix = f'policy-{policy:02d}/'
        names.update(prefix+n for n in ('attempt.json', 'attempt.candidate.json'))
        names.update(prefix+f'world-{world:02d}/arm-receipt.json' for world in range(24))
    return names


def _summary(value, files, directory_count=None):
    result = {'canonical_sha256': _digest(value), 'file_count': len(files),
              'total_bytes': sum(info['bytes'] for info in files.values())}
    if directory_count is not None:
        result['directory_count'] = directory_count
    return result


def _matched_summary(actual, wanted, role):
    _keys(actual, set(wanted), role)
    for key, value in wanted.items():
        _need(type(actual[key]) is type(value) and actual[key] == value, 'complete '+role+' digest/count differs')


def validate_receipt(receipt_bytes, receipt_sha256, expected_bytes, expected_sha256):
    """Return detached scoped evidence from TWO independently pinned byte strings.

    Context/receipt exist before packet freeze. The packet later binds their pins;
    neither object contains a future packet SHA. No current clock or filesystem is
    consulted; receipt_not_after_utc is an independently anchored source-context time.
    """
    receipt = _decode(receipt_bytes, receipt_sha256, 'receipt')
    expected = _decode(expected_bytes, expected_sha256, 'expected context')
    _keys(receipt, RECEIPT_KEYS, 'receipt'); _keys(expected, EXPECTED_KEYS, 'expected')
    _need(receipt['schema'] == 'ace-gpu-cpu-preflight-receipt-v1'
          and receipt['phase'] == 'CPU-prelaunch-complete'
          and expected['schema'] == 'ace-gpu-cpu-preflight-expected-v1', 'preflight schema/phase differs')
    context = expected['execution_context']; _keys(context, CONTEXT_KEYS, 'execution context')
    _need(context['schema'] == 'ace-gpu-cpu-preflight-context-v1', 'execution context schema differs')
    roots = context['cpu_roots']; _keys(roots, ROOT_KEYS, 'CPU roots')
    for name in roots:
        _path(roots[name], True)
    _need(roots['campaign'] == roots['cpu_output']+'/'+EXCLUDED, 'original CPU output/campaign relationship differs')
    control = _path(context['control_root'], True); old = _path(context['failed_gpu01_root'], True)
    _path(context['gpu_output'], True)
    _need(context['gpu_output'] == str(old/'gpu'), 'frozen absent GPU01/gpu output required')
    for root in (old, PurePosixPath(roots['cpu_output'])):
        _need(control != root and control not in root.parents and root not in control.parents, 'new control root overlaps original roots')
    _keys(context['lineage'], LINEAGE_KEYS, 'independent lineage')
    _keys(context['proof_pins'], PROOF_KEYS, 'accepted sealed preparation evidence')
    _need(all(_pin(h) for h in [*context['lineage'].values(), *context['proof_pins'].values()]), 'independent source/lineage/proof pins required')
    baseline = context['baseline']; _keys(baseline, {'sha256', 'terminal_jobs', 'submission_journals'}, 'baseline')
    _need(_pin(baseline['sha256']) and type(baseline['terminal_jobs']) is int and baseline['terminal_jobs'] == 47
          and type(baseline['submission_journals']) is int and baseline['submission_journals'] == 96, 'exact independently pinned47/96 baseline required')
    for key in ('source_files_sha256', 'model_files_sha256', 'resources_sha256', 'limits_sha256',
                'cpu_inventory_sha256', 'consumed_read_set_sha256'):
        _need(_pin(context[key]), 'independent execution contract digest required')
    inventories = expected['cpu_inventories']; _keys(inventories, ROOT_KEYS, 'complete CPU inventories')
    for inventory in inventories.values():
        _inventory(inventory)
    for name in list(inventories['cpu_output']['files']) + inventories['cpu_output']['directories']:
        _need(name != EXCLUDED and not name.startswith(EXCLUDED+'/'), 'campaign must be separately inventoried')
    campaign = inventories['campaign']; members = list(campaign['files']) + campaign['directories']
    _need({n.split('/')[0] for n in members} == {'campaign.json', 'source-plan-pins.json', 'start.json'} | {f'policy-{i:02d}' for i in range(5)}, 'original five-CPU-policy campaign roster required')
    for policy in range(5):
        service = f'policy-{policy:02d}/service'
        _need(all(n != service and not n.startswith(service+'/') for n in members), 'CPU persistent service evidence forbidden')
    reads = expected['consumed_read_set']
    _need(type(reads) is dict and set(reads) == _read_names(), 'exact133 campaign admission read-set members required')
    for name, info in reads.items():
        _file_info(info)
        _need(campaign['files'].get(name) == info, 'read-set pin must descend from independently supplied full campaign map')
    _need(sum(v['bytes'] for v in reads.values()) <= READ_SET_MAX_BYTES, 'complete consumed read set exceeds16MiB')
    full_value = {'cpu_roots': roots, 'cpu_output_excluded_directory': EXCLUDED, 'inventories': inventories}
    full_files = {root+'/'+name:info for root, inv in inventories.items() for name, info in inv['files'].items()}
    full_summary = _summary(full_value, full_files, sum(len(inv['directories']) for inv in inventories.values()))
    read_summary = _summary(reads, reads)
    _need(context['cpu_inventory_sha256'] == full_summary['canonical_sha256']
          and context['consumed_read_set_sha256'] == read_summary['canonical_sha256'], 'context full-map/read-set digest differs')
    context_pin = _digest(context)
    _need(_pin(expected['execution_context_sha256']) and _pin(receipt['execution_context_sha256'])
          and context_pin == expected['execution_context_sha256'] == receipt['execution_context_sha256'], 'independent execution context binding differs')
    _matched_summary(receipt['inventory_before'], full_summary, 'before inventory')
    _matched_summary(receipt['inventory_after'], full_summary, 'after inventory')
    _matched_summary(receipt['consumed_read_set'], read_summary, 'consumed read set')
    absent = sorted([context['gpu_output']] + [roots['campaign']+f'/policy-{i:02d}/service' for i in range(5)])
    _need(type(receipt['absent_paths']) is list and receipt['absent_paths'] == absent, 'exact recorded GPU-output/CPU-service absence scope required')
    started, finished, upper = (_utc(receipt['started_utc']), _utc(receipt['finished_utc']), _utc(expected['receipt_not_after_utc']))
    _need(started <= finished <= upper, 'receipt UTC order exceeds independent source-context bound')
    result = {'schema':'ace-gpu-cpu-preflight-scoped-evidence-v1', 'phase':receipt['phase'],
              'receipt_sha256':receipt_sha256, 'expected_sha256':expected_sha256,
              'execution_context_sha256':context_pin, 'cpu_inventory':full_summary,
              'consumed_read_set':read_summary, 'finished_utc':receipt['finished_utc'],
              'receipt_not_after_utc':expected['receipt_not_after_utc'], 'absent_paths':absent,
              'scope':'independently_pinned_prelaunch_metadata_consistency_only',
              'full_external_postcustody_required':True, 'continuous_immutability_proven':False,
              'gpu_execution_qualified':False, 'acceptance':False}
    return json.loads(_canonical(result))

"""Supplied-screen collection glue; no realization, tokenizer, model or scheduler.

Trusted sequential caller API. The caller supplies independent packet/admission
pins and owns the live CPU supervisor, namespace lock and scientific freeze.
Admission is a caller attestation, not something this worker can self-qualify.
Only collect_training_batch performs responses; its complete original evidence
stays in collection/. A failed or interrupted output is never reusable.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.util
import itertools
import json
import os
from pathlib import Path
import stat
import sys
import time

VERSION = 'ace-scientific-training-collection-cpu-v1'
ADMISSION_VERSION = VERSION + '/admission'
ENTRY = 'scripts/research/collect_llm_intervention_scientific_training_cpu.py'
PROTOCOL_SOURCES = frozenset('scripts/research/llm_intervention_' + n + '.py' for n in (
    'contract', 'episode', 'numerical_policy', 'replicated_branches', 'replicated_feedback',
    'scm', 'screen_protocol', 'supervision', 'target_risk', 'task', 'training_dataset', 'world'))
COLLECTION_SOURCES = PROTOCOL_SOURCES | {'scripts/research/llm_intervention_training_collection.py'}
SOURCE_PATHS = COLLECTION_SOURCES | {ENTRY}
WORLD_IDS = tuple(range(202611070000, 202611070032))
INPUT_NAMES = frozenset({'proposal.json', 'realization-terminal.json'} |
                        {f'descriptor-{i:03d}.json' for i in range(32)})
MAX_INPUT_BYTES = 8 * 1024**2
ACCOUNT_FIELDS = ('total_reserved_cost_units', 'reserved_response_attempts', 'known_returns',
                  'executor_accepted_responses', 'unknown_return_attempts')


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(_canonical(value).encode())


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _json(raw):
    _need(type(raw) is bytes and 0 < len(raw) <= MAX_INPUT_BYTES, 'bounded JSON bytes required')
    def pairs(rows):
        result = {}
        for key, value in rows:
            _need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    _canonical(value)
    return value


def _read(path, maximum=MAX_INPUT_BYTES):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as f:
        before = os.fstat(f.fileno())
        _need(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, 'bounded regular file required')
        raw = f.read(maximum+1)
        after = os.fstat(f.fileno())
    _need(len(raw) <= maximum and (before.st_ino, before.st_size, before.st_mtime_ns) ==
          (after.st_ino, after.st_size, after.st_mtime_ns), 'input changed during capture')
    return raw


def _write(path, raw):
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return _sha(raw)


def _save(path, value):
    return _write(path, (_canonical(value)+'\n').encode())


class _Captured(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, source):
        self.source = {Path(n).stem: raw for n, raw in source.items() if n != ENTRY}
        self.root = Path(__file__).absolute().parent
        self.loaded = {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.source:
            return importlib.util.spec_from_loader(fullname, self,
                origin=str(self.root/(fullname+'.py')))
        if fullname.startswith(('llm_intervention_', 'collect_llm_intervention_')):
            raise ImportError('outside admitted collection source: ' + fullname)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(self.root/(module.__name__+'.py'))
        self.loaded[module.__name__] = module
        exec(compile(self.source[module.__name__], module.__file__, 'exec'), vars(module))


@contextmanager
def _captured(source):
    _need(not any(n.startswith('llm_intervention_') for n in sys.modules),
          'project helpers imported before source authentication')
    loader = _Captured(source)
    sys.meta_path.insert(0, loader)
    try:
        yield
    finally:
        sys.meta_path.remove(loader)
        for name, module in loader.loaded.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]


def _authenticate(packet_raw, packet_sha256, inputs, admission_raw, admission_sha256, output):
    for raw, pin in ((packet_raw, packet_sha256), (admission_raw, admission_sha256)):
        _need(_pin(pin) and _sha(raw) == pin, 'independent packet/admission pin differs')
    p, a = _json(packet_raw), _json(admission_raw)
    fields = {'schema', 'scope', 'files', 'source_pins', 'proposal_sha256', 'world_roster',
              'roster_sha256', 'world_pins', 'configuration_pins', 'freeze_sha256', 'namespace_admission_sha256'}
    _need(type(p) is dict and set(p) == fields and p['schema'] == VERSION
          and p['scope'] == 'policy_train' and p['world_roster'] == list(WORLD_IDS)
          and all(type(w) is int for w in p['world_roster'])
          and p['roster_sha256'] == _digest(WORLD_IDS), 'exact scientific collection packet required')
    for field, names in (('files', INPUT_NAMES), ('source_pins', SOURCE_PATHS),
                         ('world_pins', set(map(str, WORLD_IDS))),
                         ('configuration_pins', set(map(str, WORLD_IDS)))):
        _need(type(p[field]) is dict and set(p[field]) == names and all(_pin(v) for v in p[field].values()),
              'exact independent pin map required: ' + field)
    _need(all(_pin(p[k]) for k in ('proposal_sha256', 'freeze_sha256', 'namespace_admission_sha256')),
          'proposal/freeze/namespace pins required')
    expected_admission = {'schema': ADMISSION_VERSION, 'stage': 'policy_train_collection',
        'status': 'admitted', 'packet_sha256': packet_sha256, 'output': str(output),
        'freeze_sha256': p['freeze_sha256'], 'namespace_admission_sha256': p['namespace_admission_sha256'],
        'realization_terminal_sha256': p['files']['realization-terminal.json'],
        'source_runtime_scheduler_qualified': True, 'namespace_owned': True, 'accounting_admitted': True,
        'max_response_attempts': 9216, 'max_reserved_cost_units': 10240, 'gpus': 0}
    _need(_canonical(a) == _canonical(expected_admission), 'exact caller stage/namespace/accounting admission required')
    _need(set(inputs) == INPUT_NAMES and all(_sha(inputs[n]) == pin for n, pin in p['files'].items()),
          'complete independently pinned training-only input membership required')
    source = {n: _read(Path(__file__).absolute().parent/Path(n).name) for n in SOURCE_PATHS}
    _need(all(_sha(source[n]) == pin for n, pin in p['source_pins'].items()), 'admitted source bytes differ')
    return p, source


def _bindings(p, inputs):
    from llm_intervention_screen_protocol import validate_proposal, assembly_recipe
    from llm_intervention_training_collection import configuration_payload
    from llm_intervention_world import WorldDescriptor
    proposal = _json(inputs['proposal.json'])
    protocol_pins = {n: p['source_pins'][n] for n in PROTOCOL_SOURCES}
    _need(validate_proposal(proposal, expected_source_pins=protocol_pins) == p['proposal_sha256'],
          'independent original proposal differs')
    _need(tuple(r['world_id'] for r in proposal['training_worlds']) == WORLD_IDS, 'training roster differs')
    terminal = _json(inputs['realization-terminal.json'])
    _need(terminal.get('version') == 'ace-first-screen-descriptor-realization-v1'
          and terminal.get('status') == 'complete' and type(terminal.get('worlds_generated')) is int
          and terminal['worlds_generated'] == 56 and terminal.get('retry_allowed') is False
          and type(terminal.get('environment_responses')) is int and terminal['environment_responses'] == 0
          and type(terminal.get('policy_updates')) is int and terminal['policy_updates'] == 0
          and terminal.get('scientific_execution_qualified') is False
          and terminal.get('proposal_sha256') == p['proposal_sha256']
          and terminal.get('source_pins') == protocol_pins
          and terminal.get('freeze_sha256') == p['freeze_sha256']
          and terminal.get('namespace_admission_sha256') == p['namespace_admission_sha256'],
          'complete admitted realization bindings required')
    entries = terminal.get('entries')
    rows = proposal['training_worlds'] + proposal['development_worlds']
    _need(type(entries) is list and len(entries) == 56, 'complete original realization roster required')
    for i, (entry, row) in enumerate(zip(entries, rows)):
        _need(type(entry) is dict and set(entry) == {'scope', 'world_id', 'path', 'status', 'descriptor_sha256', 'file_sha256'}
              and type(entry['world_id']) is int and entry['world_id'] == row['world_id']
              and entry['scope'] == row['scope'] and entry['path'] == f'world-{i:03d}.json'
              and entry['status'] == 'complete' and _pin(entry['descriptor_sha256']) and _pin(entry['file_sha256']),
              'realization entry order/identity/status differs')
    worlds, configs, planned = {}, {}, []
    for i, wid in enumerate(WORLD_IDS):
        name, entry = f'descriptor-{i:03d}.json', entries[i]
        _need(entry['descriptor_sha256'] == p['world_pins'][str(wid)]
              and entry['file_sha256'] == p['files'][name], 'independent realization descriptor pins differ')
        world = WorldDescriptor.from_private_payload(_json(inputs[name]), expected_sha256=p['world_pins'][str(wid)])
        config = assembly_recipe(proposal, world_id=wid, expected_source_pins=protocol_pins)
        ap, lp = config['assembly_parameters'], config['learner_parameters']
        _need(world.scope == 'policy_train' and type(world.world_id) is int and world.world_id == wid
              and world.graph == ap['graph']
              and {m.node: m.noise_variance for m in world.mechanisms} == lp['noise_variances']
              and _digest(configuration_payload(config)) == p['configuration_pins'][str(wid)],
              'supplied descriptor identity/public law/configuration differs')
        worlds[wid], configs[wid] = world, config
        for si, snapshot in enumerate(ap['snapshots']):
            planned.append({'record_id': f'policy_train/{wid}/snapshot-{si:03d}', 'scope': 'policy_train',
                'world_id': wid, 'snapshot_index': si, 'path': f'world-{i:03d}/snapshot-{si:03d}',
                'candidate_ids': list(snapshot.candidates),
                'pairs': [list(pair) for pair in itertools.combinations(snapshot.candidates, 2)]})
    _need(len(planned) == 160 and sum(len(r['pairs']) for r in planned) == 3584,
          'full 160-record/3584-pair schedule required')
    return proposal, worlds, configs, planned


def _invoke_collection(**kwargs):
    from llm_intervention_training_collection import collect_training_batch
    return collect_training_batch(**kwargs)


def _verify_collection(root, returned, planned, worlds, configs, source_pins):
    # Reuse the collector's saved-evidence reconstruction, not another parser
    # for reservations or matched supervision. No environment/learner calls.
    import llm_intervention_training_collection as collection
    terminal_raw, inventory_raw, plan_raw = (_read(root/n, 64*1024**2) for n in ('terminal.json', 'inventory.json', 'plan.json'))
    terminal, inventory, plan = map(_json, (terminal_raw, inventory_raw, plan_raw))
    _need(_canonical(returned) == _canonical(terminal) and terminal['status'] == 'complete'
          and terminal['scope'] == inventory['scope'] == 'policy_train'
          and terminal['world_roster'] == plan['world_roster'] == list(WORLD_IDS)
          and terminal['planned_records'] == plan['records'] == planned
          and terminal['record_roster'] == inventory['record_roster'] == [r['record_id'] for r in planned]
          and terminal['inventory_sha256'] == _sha(inventory_raw)
          and inventory['collection_plan_sha256'] == _sha(plan_raw)
          and plan['source_file_sha256'] == source_pins
          and plan['maximum_response_attempts'] == 9216 and plan['maximum_reserved_cost_units'] == 10240,
          'saved complete collection bindings/counts differ')
    _need(terminal['exposure_ready'] is terminal['accounting_complete'] is True
          and all(terminal[k] is False for k in ('retry_allowed', 'partial_records_authorized_for_training',
            'complete_case_filtering_allowed', 'scientific_evaluation_performed', 'policy_training_performed',
            'accounting_totals_are_lower_bounds')), 'failed or partial collection cannot be handed off')
    observed = collection._inventory(root, planned, worlds)
    _need(inventory['records'] == observed and all(r['status'] == 'complete' for r in observed)
          and inventory['record_pins'] == {r['record_id']: r['record_sha256'] for r in observed}
          and inventory['supervision_receipt_pins'] == {r['record_id']: r['supervision_receipt_sha256'] for r in observed},
          'complete original record/receipt/pair inventory differs')
    totals = dict.fromkeys(ACCOUNT_FIELDS, 0)
    _need(len(terminal['worlds']) == 32, 'complete world accounting required')
    for i, (wid, saved) in enumerate(zip(WORLD_IDS, terminal['worlds'])):
        journal_plan = collection._journal_plan(configs[wid])
        _need(saved['world_id'] == wid and saved['path'] == f'world-{i:03d}'
              and saved['status'] == 'complete' and saved['journal_plan'] == journal_plan, 'world accounting identity differs')
        account = collection._world_accounting(root/saved['path'], journal_plan)
        _need(account == saved['accounting'] and account['accounting_complete'] is True
              and account['accounting_errors'] == [] and account['counts_are_lower_bounds'] is False,
              'original journal accounting differs')
        for key in ACCOUNT_FIELDS:
            _need(type(account[key]) is int and account[key] >= 0, 'integer charged accounting required')
            totals[key] += account[key]
        menu_cost = sum(a.cost_units for a in configs[wid]['assembly_parameters']['menu'])
        for kind, count, cost in (('branch', 128, 16*menu_cost), ('scoring', 128, 128),
                                  ('trajectory', account['trajectory']['reserved_response_attempts'], 32)):
            row = account[kind]
            _need(type(count) is int and (20 <= count <= 32 if kind == 'trajectory' else True)
                  and row['reserved_response_attempts'] == row['known_returns'] == row['executor_accepted_responses'] == count
                  and row['reserved_cost_units'] == cost and row['unknown_return_attempts'] == 0,
                  'complete charged trajectory/branch/scoring counts differ')
        initial = [r for r in account['reservations'] if r['kind'] == 'trajectory' and r['request_id'] < 8]
        _need([r['request_id'] for r in initial] == list(range(8))
              and all(r['action_id'] == 'observe_all' and r['cost_units'] == 1 for r in initial),
              'eight separately charged initial responses required')
    _need(terminal['accounting'] == totals and totals['total_reserved_cost_units'] == 10240
          and totals['reserved_response_attempts'] == totals['known_returns'] == totals['executor_accepted_responses']
          and totals['reserved_response_attempts'] <= 9216 and totals['unknown_return_attempts'] == 0,
          'complete aggregate actual charges differ')
    return {'collection_terminal_sha256': _sha(terminal_raw), 'collection_inventory_sha256': _sha(inventory_raw),
        'collection_plan_sha256': _sha(plan_raw), 'record_count': 160, 'retained_pair_count': 3584, 'accounting': totals}


def run_collection(*, packet_raw, packet_sha256, inputs, admission_raw, admission_sha256, output):
    """Run one admitted supplied batch in a fresh absolute output directory.

    packet: exact schema in _authenticate, including fourteen source byte pins,
    proposal semantic pin, 32 descriptor semantic/raw pins and configuration pins.
    inputs: exactly proposal.json, realization-terminal.json, descriptor-000..031.json
    as immutable bytes. Admission: exact caller record in _authenticate, pinned
    independently, declaring current runtime/scheduler, namespace and accounting
    admission for this packet and output. These assertions must be externally
    established; this API neither enforces CPU deadlines nor acquires locks.

    A fresh isolated process with no previously imported project helpers is
    required. Parent must authenticate this entrypoint before invoking it.
    Returned wrapper terminal binds the untouched collector output in collection/;
    it is not independent acceptance or source/runtime/scheduler qualification.
    """
    output = Path(output)
    _need(output.is_absolute() and output == output.resolve() and not output.is_symlink(),
          'absolute resolved exclusive output required')
    output.mkdir(mode=0o700, exist_ok=False)
    start, cpu, invoked = time.monotonic(), time.process_time(), False
    stage = 'capture'
    try:
        _save(output/'intent.json', {'schema': VERSION, 'packet_sha256': packet_sha256,
            'admission_sha256': admission_sha256, 'world_roster': WORLD_IDS,
            'record_count': 160, 'retained_pair_count': 3584, 'retry_allowed': False,
            'started_utc': datetime.now(timezone.utc).isoformat()})
        _need(type(packet_raw) is bytes and type(admission_raw) is bytes
              and len(packet_raw)+len(admission_raw) <= MAX_INPUT_BYTES, 'bounded packet/admission required')
        _write(output/'packet.json', packet_raw)
        _write(output/'admission.json', admission_raw)
        _need(type(inputs) is dict and len(inputs) <= 64 and all(type(k) is str and type(v) is bytes for k,v in inputs.items())
              and sum(map(len, inputs.values())) <= MAX_INPUT_BYTES, 'bounded captured input bytes required')
        inputs = dict(inputs)
        (output/'inputs').mkdir(mode=0o700)
        index = []
        for i, (name, raw) in enumerate(sorted(inputs.items())):
            path = f'inputs/{i:03d}.bin'
            index.append({'name': name, 'path': path, 'sha256': _write(output/path, raw)})
        _save(output/'input-index.json', index)
        stage = 'admission'
        p, source = _authenticate(packet_raw, packet_sha256, inputs, admission_raw, admission_sha256, output)
        (output/'source').mkdir(mode=0o700)
        for name, raw in sorted(source.items()):
            _write(output/'source'/Path(name).name, raw)
        with _captured(source):
            stage = 'supplied_bindings'
            proposal, worlds, configs, planned = _bindings(p, inputs)
            _save(output/'planned-records.json', planned)
            stage, invoked = 'collection', True
            sources = {n: p['source_pins'][n] for n in COLLECTION_SOURCES}
            returned = _invoke_collection(scope='policy_train', world_roster=WORLD_IDS,
                expected_roster_sha256=p['roster_sha256'], worlds=worlds,
                world_pins={wid: p['world_pins'][str(wid)] for wid in WORLD_IDS},
                configurations=configs, configuration_pins={wid: p['configuration_pins'][str(wid)] for wid in WORLD_IDS},
                expected_source_pins=sources, output=output/'collection', proposal=proposal,
                expected_proposal_sha256=p['proposal_sha256'])
            stage = 'saved_evidence'
            checked = _verify_collection(output/'collection', returned, planned, worlds, configs, sources)
        result = {'schema': VERSION, 'status': 'collection_complete', 'packet_sha256': packet_sha256,
            'admission_sha256': admission_sha256, 'collection_path': 'collection', **checked,
            'elapsed_seconds': time.monotonic()-start, 'process_cpu_seconds': time.process_time()-cpu,
            'worlds_generated': 0, 'training_updates': 0, 'tokenizer_calls': 0, 'model_loads': 0,
            'development_evaluated': False, 'retry_allowed': False, 'independent_acceptance_required': True,
            'source_runtime_scheduler_qualified': False}
        _save(output/'terminal-candidate.json', result)
        os.link(output/'terminal-candidate.json', output/'terminal.json')
        return result
    except BaseException as exc:
        # Never reinterpret partial charges as zero; the collector's complete
        # tree and original failure remain authority even if unreadable.
        failure = {'schema': VERSION, 'status': 'failed', 'stage': stage,
            'exception_type': type(exc).__name__, 'message': str(exc), 'collection_invoked': invoked,
            'accounting_authority': 'collection/ (original journals and terminal/failure)' if invoked else None,
            'responses_before_collection': 0, 'retry_allowed': False, 'exposure_ready': False,
            'world_roster': WORLD_IDS, 'planned_records': 160, 'planned_pairs': 3584,
            'elapsed_seconds': time.monotonic()-start, 'process_cpu_seconds': time.process_time()-cpu}
        try:
            if os.path.lexists(output/'terminal.json'):
                failure['status'] = 'publication_interrupted_reconciliation_required'
                _save(output/'publication-interruption.json', failure)
            else:
                _save(output/'failure.json', failure)
        except BaseException:
            pass  # A secondary publication failure must not mask the first error.
        raise

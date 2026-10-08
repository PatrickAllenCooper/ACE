"""Assemble a private, relative Stage B plan after original full acceptance.

No execution, model loading, response generation, installs or network. Callers
must supply an independently pinned prepared core contract, never a default
custody location. This tool may decode accepted outcomes ONLY after the original
gate and complete composite custody checks; preparation on partial custody fails.

Composite inventory input: schema delivery-prospective-composite-v1, completed
true, conflict_policy 'reject', conflicts [], missing_indices [], cells_verified
640, registration_sha256, matrix_sha256, and files [{path, sha256, sources}].
Paths are package-relative: B/... for original study files, source/runner/... for
the 19 learners, and B/project/... for the original source-commit receipt and
its committed files. Every claimed source is verified, including unused/private
logs and failures. Include all retained custody files; completeness outside this
explicit inventory is not a claim this planner can authenticate.

Frozen source inventory is the unchanged archive_delivery_worker_provenance
format, rooted at source_root. Its B closure is exactly 20 worker/generator
files. Optional sources on those entries are also all checked. Original workers,
receipts, inventories and core contract remain in exclusive private snapshots.
Structured projections are newly authored files, added to the inherited plan as
identity entries. Their original/derived provenance lives in replay_contract.json
and the private derivation record, not a fabricated original receipt digest.
The future replay caller can be supplied explicitly with replay; no implicit
lookup or read of replay_delivery_prospective_release.py is performed.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import prepare_delivery_prospective_release_core as core
import delivery_prospective_source_contract as source_contracts
from build_delivery_release import write
from verify_delivery_release import identifying_bytes, relative, sha

WORLDS = [f'{size}:{i:02d}' for size in (5, 30) for i in range(20)]
HISTORIES = {'balanced_varied_value': 'balanced', 'matched_random': 'random'}
ARMS = [('delivery', 0), ('online', 0), ('simpler', 0), ('ablation', 0),
        ('delivery', 1), ('simpler', 1), ('delivery', 2), ('simpler', 2)]
CELL_KEYS = ('index', 'case', 'graph_size', 'history', 'arm', 'init', 'binding')
SUPERVISOR_KEYS = ('status', 'exit_code', 'registration_sha256', 'acceptance_sha256',
                   'peak_tree_rss_bytes', 'elapsed_seconds')
COMPLETE_KEYS = ('n_fits', 'n_primary_cells', 'charged_responses', 'registration_sha256',
                 'fit_seal_sha256', 'scores_sha256', 'fit_cpu_core_hours')
RESOURCE_KEYS = ('world_wall_seconds', 'qualification_wall_seconds', 'collection_wall_seconds',
                 'evaluation_wall_seconds', 'audit_wall_seconds', 'historical_failed_pilot_cpu_seconds',
                 'pilot_reserved_seconds', 'total_requested_cpu_core_hours', 'threads', 'rss_bytes',
                 'max_simultaneous_worlds')
REGISTRATION_KEYS = ('at', 'stage', 'source_revision', 'worlds', 'histories', 'cells',
                     'dependencies', 'source_hashes', 'worker_hashes', 'generator_hashes',
                     'matrix_fits', 'primary_cells', 'new_response_ceiling', 'training_responses',
                     'shared_evaluation_responses', 'estimand', 'primary_targets', 'metric',
                     'calibration', 'init', 'contrasts', 'superiority', 'masking', 'scope',
                     'stop', 'initial_response_count')
SCORE_KEYS = ('cells', 'primary_analysis', 'primary_rows', 'scope', 'new_training_responses',
              'new_shared_evaluation_responses', 'fit_seal_sha256', 'evaluation_cpu_seconds',
              'evaluation_wall_seconds')
PROTOCOL_FILES = {
    'descriptor_manifest.json': 'descriptor_manifest_sha256',
    'attribution_gate.json': 'gate_sha256', 'pilot_acceptance.json': 'pilot_acceptance_sha256',
    'pilot_projection.json': 'pilot_projection_sha256',
    'historical_pilot_failure.json': 'historical_pilot_failure_sha256',
    'descriptor_runtime_parity.json': 'descriptor_runtime_parity_sha256'}
MAX_JSON_BYTES = 32 * 1024 * 1024


def snapshot(path, expected=None):
    """Hash and decode one bounded byte snapshot; reject duplicate/NaN JSON."""
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('metadata exceeds planner bound')
    raw = path.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError('metadata exceeds planner bound')
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('metadata snapshot differs')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid), digest, raw


def project(value, keys):
    return {key: value[key] for key in keys}


def require(value, message):
    if not value:
        raise ValueError(message)


def checked_copies(sources, expected):
    require(core.digest(expected) and isinstance(sources, list) and sources,
            'digest and explicit custody copies required')
    for source in sources:
        require(isinstance(source, str) and Path(source).is_absolute(), 'absolute private custody path required')
        p = Path(source)
        require(not any(v.is_symlink() for v in (p, *p.parents)), 'symlink in private custody')
        require(p.is_file() and sha(p) == expected, 'missing/changed/conflicting custody copy')
    return Path(sources[0])


def number(value, label):
    require(core.finite_nonnegative(value), 'invalid ' + label)


def execution(value, reg, wall_key, account=True):
    require(value['status'] == 'complete' and type(value['exit_code']) is int and value['exit_code'] == 0,
            'incomplete execution')
    if account:
        require(value['account'] == 'ucb736_asc1', 'original execution account differs')
    require(value['registration_sha256'] == reg['_sha'], 'execution registration differs')
    number(value['peak_tree_rss_bytes'], 'RSS'); number(value['elapsed_seconds'], 'elapsed time')
    require(type(value['peak_tree_rss_bytes']) is int and
            value['peak_tree_rss_bytes'] <= reg['resources']['rss_bytes'] and
            value['elapsed_seconds'] <= reg['resources'][wall_key], 'execution resource bounds differ')


def extend(plan_file, prospective, inventory, source_inventory, source_root,
           core_file, core_contract, expected_core_contract_sha256, private_dir, out,
           replay=None, expected_registration=core.REGISTRATION_SHA,
           expected_source_contract_sha256=source_contracts.PREDECESSOR_SHA):
    # This MUST be the first operation. No plan/inventory/core/score reads precede it.
    gate = core.accepted_before_scores(prospective, expected_registration)
    prospective = Path(prospective)
    reg, _, _ = snapshot(prospective/'registration.json', gate['registration_sha256'])
    require(reg['worlds'] == WORLDS and reg['histories'] == HISTORIES and
            reg['cells'] == [list(a) for a in ARMS] and len(reg['source_hashes']) == 19,
            'exact registered 640 matrix and 19 learners required')
    reg['_sha'] = gate['registration_sha256']
    inv, inv_hash, inv_raw = snapshot(inventory)
    require(inv['schema'] == 'delivery-prospective-composite-v1' and inv['completed'] is True and
            inv['conflict_policy'] == 'reject' and inv['conflicts'] == [] and inv['missing_indices'] == [] and
            type(inv['cells_verified']) is int and inv['cells_verified'] == 640 and
            inv['registration_sha256'] == reg['_sha'], 'complete conflict-rejecting 640 inventory required')
    entries = {}
    for entry in inv['files']:
        name = entry['path']; relative(Path('/'), name)
        require(name not in entries and name.startswith(('B/', 'source/runner/')),
                'duplicate or unregistered inventory path')
        entries[name] = {**entry, '_source': checked_copies(entry['sources'], entry['sha256'])}
    require(not any(n.endswith(('/stop_new_fits.json', '/failure.json')) for n in entries),
            'retained active failure blocks planning')
    def bound(name, expected=None):
        require(name in entries, 'missing custody artifact: ' + name)
        e = entries[name]
        require(expected is None or e['sha256'] == expected, 'original digest binding differs: ' + name)
        return e
    def metadata(name, expected=None):
        e = bound(name, expected)
        return snapshot(e['_source'], e['sha256'])[0]
    for name, expected in [('registration.json', reg['_sha']),
                           ('acceptance.json', gate['acceptance_sha256']),
                           ('audit_execution.json', gate['audit_execution_sha256']),
                           ('scores.json', gate['scores_sha256'])]:
        bound('B/'+name, expected)
    matrix = metadata('B/matrix.json', inv['matrix_sha256'])
    seal = metadata('B/fit_seal.json')
    complete = metadata('B/complete.json', gate['custody_bound_acceptance_projection']['metadata']['complete_sha256'])
    require(seal['registration_sha256'] == reg['_sha'] and
            seal['matrix_sha256'] == inv['matrix_sha256'] and
            matrix['registration_sha256'] == reg['_sha'] and
            complete['fit_seal_sha256'] == bound('B/fit_seal.json')['sha256'], 'matrix/seal/complete differs')
    for name, key in PROTOCOL_FILES.items():
        bound('B/'+name, reg[key])
    src_inv, src_inv_hash, src_inv_raw = snapshot(source_inventory)
    require(src_inv['protocol_hashes']['B'] == reg['_sha'], 'source inventory registration differs')
    expected_sources = {'B/scripts/research/'+n: h for n, h in reg['worker_hashes'].items()}
    expected_sources.update({'B/'+n: h for n, h in reg['generator_hashes'].items()})
    b_sources = [e for e in src_inv['files'] if e['stage'] == 'B']
    require(len(b_sources) == 20 and len(expected_sources) == 20 and
            {e['path'] for e in b_sources} == set(expected_sources), 'exact frozen source closure required')
    frozen = {}
    for e in b_sources:
        require(e['source_revision'] == core.REVISION and e['transformation'] == 'identity' and
                e['sha256'] == expected_sources[e['path']], 'frozen source provenance differs')
        p = relative(source_root, e['path'])
        sources = [str(p.absolute()), *e.get('sources', [])]
        frozen[e['path']] = {'path': e['path'], 'sha256': e['sha256'], 'sources': sources,
                             '_source': checked_copies(sources, e['sha256'])}
    committed = metadata('B/project/source_commit_receipt.json', reg['source_commit_receipt_sha256'])
    require(committed['source_revision'] == core.REVISION, 'committed revision differs')
    require({'scripts/research/'+n for n in reg['worker_hashes']} | set(reg['generator_hashes'])
            <= set(committed['files']), 'committed closure incomplete')
    for name, h in committed['files'].items():
        bound('B/project/'+name, h)
    for name, h in reg['source_hashes'].items():
        bound('source/runner/'+name, h)
    require(core.digest(expected_core_contract_sha256), 'external core contract SHA256 required')
    cc, cc_hash, cc_raw = snapshot(core_contract, expected_core_contract_sha256)
    require(cc['registration_sha256'] == reg['_sha'] and cc['original_revision'] == core.REVISION and
            cc['target_dependencies'] == reg['dependencies'] and cc['matrix_fits'] == 640 and
            cc['primary_cells'] == 240 and cc['frozen_source_inventory_sha256'] == src_inv_hash and
            cc['frozen_source_bindings_checked'] == 20 and cc['source_guards_bypassed'] is False and
            cc['B_outcomes_opened'] is False and cc['new_fits'] == 0 and cc['new_responses'] == 0 and
            cc['original_auditor_sha256'] == reg['worker_hashes']['audit_delivery_prospective_results.py'] and
            cc['original_utility_sha256'] == reg['worker_hashes']['runner_delivery_confirmation.py'] and
            cc['preparation_and_gate_tool_sha256'] == sha(core.__file__), 'prepared core contract differs')
    checked_copies([str(Path(core_file).absolute())], cc['derived_core_sha256'])
    # Re-extract but never execute the pinned original source closure.
    code, segments = core.extracted_core(frozen['B/scripts/research/audit_delivery_prospective_results.py']['_source'],
        cc['original_auditor_sha256'], frozen['B/scripts/research/runner_delivery_confirmation.py']['_source'],
        cc['original_utility_sha256'])
    require(hashlib.sha256(code.encode()).hexdigest() == cc['derived_core_sha256'] and
            segments == cc['exact_extracted_segments'], 'exact derived core provenance differs')
    accepted = gate['custody_bound_acceptance_projection']['metadata']
    training = {case+'/'+h: 'B/training/'+case.replace(':', '-')+'/'+h for case in WORLDS for h in HISTORIES}
    evaluation = {case: 'B/evaluation/'+case.replace(':', '-') for case in WORLDS}
    descriptors = {case: {n: 'B/descriptors/'+case.replace(':', '-')+'/'+n+'.json'
                         for n in ('world', 'actions')} for case in WORLDS}
    collection = metadata('B/collection_complete.json')
    require(collection['registration_sha256'] == reg['_sha'] and collection['charged_responses'] == 32000 and
            collection['matrix_sha256'] == inv['matrix_sha256'] and
            collection['artifacts'] == accepted['training_receipt_hashes'], 'collection receipt differs')
    for folders, hashes in [(training, accepted['training_receipt_hashes']),
                             (evaluation, accepted['evaluation_receipt_hashes'])]:
        for key, folder in folders.items():
            for name, h in hashes[key].items():
                bound(folder+'/'+name, h)
    dm = metadata('B/descriptor_manifest.json')
    require(set(dm['worlds']) == set(WORLDS), 'descriptor membership differs')
    for case, paths in descriptors.items():
        for n, path in paths.items():
            bound(path, dm['worlds'][case][n+'_sha256'])
    required_fits = {f'B/fits/{case.replace(":", "-")}/{history}/{arm}-i{init}/{name}'
                     for case in WORLDS for history in HISTORIES for arm, init in ARMS
                     for name in ('receipt.json', 'models.pt')}
    require({n for n in entries if n.startswith('B/fits/') and n.endswith(('/receipt.json', '/models.pt'))}
            == required_fits, 'exact 640 fit custody membership required')
    for case in WORLDS:
        for index in range(WORLDS.index(case)*16, (WORLDS.index(case)+1)*16):
            bound(evaluation[case]+'/cell-'+str(index)+'-predictions.npz')
    # Construct canonical paths from the original registration for comparison
    # only. They are never edited, executed, or published.
    base = Path(reg['output'])
    require(base.is_absolute() and '..' not in base.parts, 'invalid canonical output')
    canonical = []; cells = []; latest = None
    for case in WORLDS:
        for history in HISTORIES:
            for arm, init in ARMS:
                index = len(cells); folder = 'B/fits/'+case.replace(':', '-')+'/'+history+'/'+arm+'-i'+str(init)
                binding = {'protocol_sha256': reg['_sha'],
                           'input_sha256': accepted['training_receipt_hashes'][case+'/'+history]['input.json'],
                           'kernel_sha256': reg['worker_hashes']['delivery_prospective_models.py'],
                           'source_hashes': reg['source_hashes'], 'dependencies': reg['dependencies']}
                cell = dict(index=index, case=case, graph_size=int(case.split(':')[0]),
                            history=history, arm=arm, init=init, binding=binding)
                original = {**cell, 'input_dir': str(base/training[case+'/'+history][2:]),
                            'out': str(base/folder[2:])}
                canonical.append(original)
                require(original['out'] in seal['artifacts'], 'fit seal membership missing')
                hashes = seal['artifacts'][original['out']]
                bound(folder+'/models.pt', hashes['model_sha256'])
                receipt = metadata(folder+'/receipt.json', hashes['receipt_sha256'])
                require(receipt['complete'] is True and receipt['binding'] == binding and receipt['arm'] == arm and
                        receipt['init'] == init and receipt['model_sha256'] == hashes['model_sha256'] and
                        receipt['new_queries_in_fit'] == 0 and receipt['evaluation_responses_read'] == 0,
                        'fit receipt identity/boundary differs')
                finished = datetime.fromisoformat(receipt['finished_at'])
                latest = max(latest, finished) if latest else finished
                cells.append({**cell, 'folder': folder, **hashes})
    require(matrix['cells'] == canonical and set(seal['artifacts']) == {c['out'] for c in canonical},
            'exact original full matrix/seal membership differs')
    barrier = project(metadata('B/evaluation_started.json'), ('at', 'fit_seal_sha256'))
    require(barrier['fit_seal_sha256'] == bound('B/fit_seal.json')['sha256'] and
            datetime.fromisoformat(barrier['at']) >= latest, 'evaluation barrier differs')
    safe_execution = {}
    supervisor = metadata('B/audit_execution.json', gate['audit_execution_sha256'])
    execution(supervisor, reg, 'audit_wall_seconds')
    for phase, wall in [('qualification', 'qualification'), ('collect', 'collection'), ('evaluate', 'evaluation')]:
        name = 'B/'+phase+'_execution.json'
        value = metadata(name, accepted['phase_execution_hashes'][phase+'_execution.json'])
        execution(value, reg, wall+'_wall_seconds')
        safe_execution[name] = project(value, tuple(k for k in SUPERVISOR_KEYS if k != 'acceptance_sha256'))
    qualification = metadata('B/qualification_complete.json')
    require(qualification['registration_sha256'] == reg['_sha'] and qualification['dependencies'] == reg['dependencies'] and
            qualification['confirmation_responses_evaluated'] == 0 and
            qualification['execution_sha256'] == bound('B/qualification_execution.json')['sha256'], 'qualification differs')
    bound('B/qualification.log', qualification['log_sha256'])
    world_execution = {c: 'B/world_execution/'+c.replace(':', '-') for c in WORLDS}
    for case, folder in world_execution.items():
        done = metadata(folder+'/complete.json'); attempts = metadata(folder+'/execution.json')['attempts']
        require(done['n_fits'] == 16 and done['protocol_sha256'] == reg['_sha'] and
                [a['cell_index'] for a in attempts] == [c['index'] for c in cells if c['case'] == case],
                'world completion/attempt membership differs')
        safe_attempts = []
        for a in attempts:
            require(type(a['cell_index']) is int and a['status'] == 'complete' and
                    type(a['exit_code']) is int and a['exit_code'] == 0, 'world attempt failed')
            number(a['peak_tree_rss_bytes'], 'world RSS')
            require(type(a['peak_tree_rss_bytes']) is int and
                    a['peak_tree_rss_bytes'] <= reg['resources']['rss_bytes'], 'world RSS exceeds bound')
            keys = ['cell_index', 'status', 'exit_code', 'peak_tree_rss_bytes']
            # Legacy supervise() omitted elapsed time. Preserve measured values
            # when present and report their absence; never fabricate telemetry.
            if 'elapsed_seconds' in a:
                number(a['elapsed_seconds'], 'attempt elapsed'); keys.append('elapsed_seconds')
            if 'finished_at' in a:
                keys.append('finished_at')
            safe_attempts.append(project(a, keys))
        safe_execution[folder+'/complete.json'] = project(done, ('at', 'n_fits', 'protocol_sha256'))
        safe_execution[folder+'/execution.json'] = {'attempts': safe_attempts}
    # Inventory, source/core contracts, every copy and every membership above
    # are verified before this first score decode. Use one captured snapshot.
    plan, plan_hash, plan_raw = snapshot(plan_file)
    plan = copy.deepcopy(plan); indexed = {e['path']: e for e in plan['files']}
    require(len(indexed) == len(plan['files']), 'duplicate inherited plan entry')
    prepared_sources = source_contracts.preflight(plan, snapshot, expected_source_contract_sha256)
    require(prepared_sources is None or replay is not None,
            'inherited source contract requires an explicit B replay interface')
    source_contracts.validate_before_scores(plan, prepared_sources, frozen, cc, checked_copies)
    replay_entry = None
    if replay is not None:
        replay_entry = {'path': 'replay_delivery_prospective_release.py',
                        'sources': [str(Path(replay).absolute())], 'sha256': sha(replay)}
        checked_copies(replay_entry['sources'], replay_entry['sha256'])
    scores = metadata('B/scores.json', gate['scores_sha256'])
    require(set(scores) == set(SCORE_KEYS) and set(scores['cells']) == {str(i) for i in range(640)} and
            scores['fit_seal_sha256'] == bound('B/fit_seal.json')['sha256'] and scores['scope'] == reg['scope'] and
            scores['new_training_responses'] == 32000 and scores['new_shared_evaluation_responses'] == 16000 and
            len(scores['primary_rows']) == 240, 'accepted score structure differs')
    predictions = {}
    for c, original in zip(cells, canonical):
        saved = scores['cells'][str(c['index'])]
        require(set(saved) == {'cell', 'metric', 'evaluation_input_sha256', 'predictions_sha256'} and
                saved['cell'] == original and saved['evaluation_input_sha256'] ==
                accepted['evaluation_receipt_hashes'][c['case']]['input.json'], 'score cell binding differs')
        name = evaluation[c['case']]+'/cell-'+str(c['index'])+'-predictions.npz'
        bound(name, saved['predictions_sha256']); predictions[str(c['index'])] = name
        saved['cell'] = project(c, CELL_KEYS)
    acceptance = metadata('B/acceptance.json', gate['acceptance_sha256'])
    require(set(acceptance) == core.ACCEPTANCE_FIELDS and
            project(acceptance, accepted.keys()) == accepted, 'captured acceptance metadata differs')
    # All output paths and collision checks precede the first output write.
    for name in indexed:
        relative(Path('/'), name)
    require(not any(n.startswith('B/') or n == 'delivery_prospective_replay_core.py' or
                    n == 'replay_delivery_prospective_release.py' for n in indexed), 'B already in inherited plan')
    private_dir, out = Path(private_dir).absolute(), Path(out).absolute()
    require(not private_dir.exists() and not out.exists() and not out.is_relative_to(private_dir),
            'exclusive separate private directory and plan output required')
    inputs = [Path(p).absolute() for p in (prospective, source_root, plan_file, inventory, source_inventory,
                                          core_file, core_contract)]
    inputs += [Path(s) for e in [*entries.values(), *frozen.values()] for s in e['sources']]
    for p in inputs:
        require(not private_dir.is_relative_to(p) and not p.is_relative_to(private_dir) and
                out != p and not out.is_relative_to(p), 'output overlaps original custody')
    for name, h in reg['source_hashes'].items():
        path = 'source/runner/'+name
        if path in indexed:
            e = indexed[path]
            require(e.get('transform', {'kind': 'identity'}) == {'kind': 'identity'} and e['original_sha256'] == h,
                    'inherited learner conflict')
            checked_copies(e['sources'], h)
    private_dir.mkdir(parents=True, exist_ok=False)
    (private_dir/'original').mkdir(); (private_dir/'derived').mkdir()
    for name, raw in [('input_plan.json', plan_raw), ('composite_inventory.json', inv_raw),
                      ('source_inventory.json', src_inv_raw), ('prepared_core_contract.json', cc_raw)]:
        with (private_dir/name).open('xb') as f:
            f.write(raw)
    preserved = {}
    def preserve(e, prefix='original'):
        target = relative(private_dir/prefix, e['path']); target.parent.mkdir(parents=True, exist_ok=True)
        with e['_source'].open('rb') as source, target.open('xb') as dest:
            shutil.copyfileobj(source, dest, 1024*1024)
        require(sha(target) == e['sha256'], 'custody changed during private snapshot')
        return target
    for name, e in entries.items():
        preserved[name] = preserve(e)
    for e in frozen.values():
        preserve(e, 'frozen_source')
    checked_copies([str(Path(core_file).absolute())], cc['derived_core_sha256'])
    core_target = private_dir/'prepared_core.py'
    with Path(core_file).open('rb') as src, core_target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    require(sha(core_target) == cc['derived_core_sha256'], 'core changed during snapshot')
    provenance = {}; public = {}
    def add_identity(name, path, h, role='prospective-original-artifact'):
        if name in indexed:
            require(indexed[name]['original_sha256'] == h, 'inherited identity conflict')
        else:
            require(not identifying_bytes(path), 'identifier requires explicit disposition: '+name)
            entry = {'path': name, 'sources': [str(path)], 'role': role, 'original_sha256': h,
                     'transform': {'kind': 'identity'}}
            plan['files'].append(entry); indexed[name] = entry
        public[name] = h
    def derive(name, value, original_name):
        target = relative(private_dir/'derived', name); target.parent.mkdir(parents=True, exist_ok=True)
        write(target, value); h = sha(target)
        add_identity(name, target, h, 'prospective-derived-metadata')
        provenance[name] = {'original_path': original_name,
                            'original_sha256': bound(original_name)['sha256'], 'derived_sha256': h,
                            'method': 'explicit structured projection; original private bytes preserved'}
    # Numerical artifacts, fit receipts, action menus and charge journals retain
    # identity. Descriptor source is machine custody metadata, projected out.
    for c in cells:
        for n in ('receipt.json', 'models.pt'):
            name = c['folder']+'/'+n; add_identity(name, preserved[name], bound(name)['sha256'])
    for folder in [*training.values(), *evaluation.values()]:
        for n in ('input.json', 'queries.ndjson', 'receipt.json'):
            name = folder+'/'+n; add_identity(name, preserved[name], bound(name)['sha256'])
    for name in predictions.values():
        add_identity(name, preserved[name], bound(name)['sha256'])
    for paths in descriptors.values():
        world = metadata(paths['world'])
        allowed = {'size', 'seed', 'family', 'order', 'parents', 'coefficients', 'noise_sd', 'source'}
        if world['size'] == 30:
            allowed.add('nonlinear_nodes')
        require(set(world) == allowed, 'descriptor projection fields differ')
        derive(paths['world'], {k: world[k] for k in sorted(allowed - {'source'})}, paths['world'])
        name = paths['actions']; add_identity(name, preserved[name], bound(name)['sha256'])
    for name, h in reg['source_hashes'].items():
        path = 'source/runner/'+name; add_identity(path, preserved[path], h, 'original-learner-source')
    helper = 'B/scripts/research/delivery_prospective_design.py'
    helper_path = private_dir/'frozen_source'/helper
    add_identity('delivery_prospective_design.py', helper_path, frozen[helper]['sha256'], 'original-safe-helper')
    add_identity('delivery_prospective_replay_core.py', core_target, cc['derived_core_sha256'], 'derived-exact-numerical-core')
    contract_target = private_dir/'prepared_core_contract.json'
    add_identity('B/core_contract.json', contract_target, cc_hash, 'prepared-core-provenance')
    if replay_entry:
        add_identity(replay_entry['path'], Path(replay_entry['sources'][0]), replay_entry['sha256'], 'prospective-replay-adapter')
    safe_reg = project(reg, REGISTRATION_KEYS); safe_reg['resources'] = project(reg['resources'], RESOURCE_KEYS)
    derive('B/registration.json', safe_reg, 'B/registration.json')
    derive('B/scores.json', scores, 'B/scores.json')
    derive('B/acceptance.json', project(acceptance, sorted(core.ACCEPTANCE_FIELDS)), 'B/acceptance.json')
    derive('B/complete.json', project(complete, COMPLETE_KEYS), 'B/complete.json')
    derive('B/audit_execution.json', project(supervisor, SUPERVISOR_KEYS), 'B/audit_execution.json')
    derive('B/evaluation_started.json', barrier, 'B/evaluation_started.json')
    derive('B/matrix.json', {'registration_sha256': reg['_sha'], 'cells': [project(c, CELL_KEYS) for c in cells]}, 'B/matrix.json')
    derive('B/fit_seal.json', {'registration_sha256': reg['_sha'], 'matrix_sha256': inv['matrix_sha256'],
                            'artifacts': {c['folder']: project(c, ('model_sha256', 'receipt_sha256')) for c in cells}}, 'B/fit_seal.json')
    for name, value in safe_execution.items():
        derive(name, value, name)
    failure = metadata('B/historical_pilot_failure.json')
    derive('B/historical_pilot_failure.json', project(failure, ('state', 'allocated_cpu_seconds')),
           'B/historical_pilot_failure.json')
    original_artifacts = {n: {'original_sha256': e['sha256'],
                              'relative_artifact': n if n in public else None,
                              'derived_sha256': public.get(n)} for n, e in entries.items()}
    contract = {'schema': 'delivery-prospective-replay-v1', 'registration_sha256': reg['_sha'],
                'original_revision': core.REVISION, 'dependencies': reg['dependencies'], 'scope': reg['scope'],
                'matrix_fits': 640, 'primary_cells': 240,
                'original_acceptance_sha256': gate['acceptance_sha256'],
                'original_supervisor_sha256': gate['audit_execution_sha256'],
                'original_scores_sha256': gate['scores_sha256'],
                'original_complete_sha256': accepted['complete_sha256'],
                'core_sha256': cc['derived_core_sha256'], 'core_contract_sha256': cc_hash,
                'learner_hashes': reg['source_hashes'],
                'helper_hashes': {'delivery_prospective_design.py': frozen[helper]['sha256']},
                'core_path': 'delivery_prospective_replay_core.py', 'core_contract_path': 'B/core_contract.json',
                'learner_root': 'source/runner', 'registration': safe_reg, 'resources': safe_reg['resources'],
                'upstream_gate': gate,
                'supervisor': project(supervisor, SUPERVISOR_KEYS), 'complete': project(complete, COMPLETE_KEYS),
                'fit_seal_sha256': bound('B/fit_seal.json')['sha256'], 'barrier': barrier,
                'cells': cells, 'training': training, 'evaluation': evaluation,
                'descriptors': descriptors, 'world_execution': world_execution, 'predictions': predictions,
                'metadata_projections': provenance, 'original_artifacts': original_artifacts,
                'composite_inventory_sha256': inv_hash, 'source_inventory_sha256': src_inv_hash,
                'preparation_only': True, 'new_fits': 0, 'new_responses': 0,
                'world_attempt_elapsed_seconds_unknown': [a['cell_index']
                    for case in WORLDS for a in safe_execution[world_execution[case]+'/execution.json']['attempts']
                    if 'elapsed_seconds' not in a],
                'limitations': ['Target-runtime checkpoint/statistical replay remains required.',
                    'Private unprojected workers, logs and custody receipts are not public artifacts.',
                    'Inventory completeness outside explicitly claimed copies is not authenticated.',
                    'Legacy world journals may omit elapsed time; missing measurements are reported as unknown.',
                    'Full composite inventory construction from the fit-only reconciler remains a separate preparation step.',
                    'Historical freeze, all-sprint accounting and human anonymity/redistribution approval remain open.']}
    target = private_dir/'derived/B/replay_contract.json'; write(target, contract)
    add_identity('B/replay_contract.json', target, sha(target), 'prospective-relative-replay-contract')
    # Manifest bindings pin derived files. Original digests are explicit
    # provenance values, never rebound to newly authored metadata digests.
    def bind(pointer, path):
        plan['bindings'].append({'record': 'B/replay_contract.json', 'pointer': pointer,
                                 'artifact': path, 'digest': 'sha256'})
    def escape(name):
        return name.replace('~', '~0').replace('/', '~1')
    for n, h in public.items():
        if n == 'B/replay_contract.json':
            continue
        if n in provenance:
            bind('/metadata_projections/'+escape(n)+'/derived_sha256', n)
        elif n in original_artifacts:
            bind('/original_artifacts/'+escape(n)+'/derived_sha256', n)
    bind('/core_sha256', 'delivery_prospective_replay_core.py')
    bind('/core_contract_sha256', 'B/core_contract.json')
    bind('/helper_hashes/delivery_prospective_design.py', 'delivery_prospective_design.py')
    for n in reg['source_hashes']:
        bind('/learner_hashes/'+escape(n), 'source/runner/'+n)
    source_transition = source_contracts.transition(plan, indexed, prepared_sources, private_dir,
        plan_hash, frozen, cc, contract, sha(target), replay_entry)
    plan['status'] = 'private accepted B relative-artifact preparation; target-runtime replay and public approval pending'
    write(private_dir/'derivation.json', {'inherited_plan_sha256': plan_hash,
          'composite_inventory_sha256': inv_hash, 'source_inventory_sha256': src_inv_hash,
          'core_contract_sha256': cc_hash, 'original_artifacts': {n: {'sources': e['sources'],
           'sha256': e['sha256'], 'private_snapshot': str(preserved[n])} for n, e in entries.items()},
          'metadata_projections': provenance, 'source_contract_transition': source_transition,
          'B_outcomes_decoded_after_full_acceptance': True,
          'new_fits': 0, 'new_responses': 0})
    write(out, plan)
    return {'plan_sha256': sha(out), 'replay_contract_sha256': sha(target),
            'files': len(plan['files']), 'fits': 640, 'worlds': 40, 'training_bundles': 80,
            'evaluation_bundles': 40, 'predictions': 640, 'preparation_only': True,
            'replay_adapter_included': replay is not None, 'source_contract_transition': source_transition,
            'new_fits': 0, 'new_responses': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan-file', 'prospective', 'inventory', 'source-inventory', 'source-root',
                 'core-file', 'core-contract', 'private-dir', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-core-contract-sha256', required=True)
    parser.add_argument('--expected-source-contract-sha256', default=source_contracts.PREDECESSOR_SHA)
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    print(json.dumps(extend(**vars(args)), indent=2))

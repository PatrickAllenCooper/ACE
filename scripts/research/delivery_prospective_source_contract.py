"""Supersede an inherited A/C/F source contract during accepted B planning.

No outcome, checkpoint, response or original-worker execution is performed here.
The caller must first pass the original upstream audit and complete custody gates.
This records included helper/core/interface roles; exact-runtime numerical replay
and human anonymity/submission approval remain distinct gates.
"""
import copy
import ast
import hashlib
from pathlib import Path

from build_delivery_release import write, reconcile
from extend_delivery_source_contract import ROLES, INTERFACES, snapshot_entry

ACTIVE = 'reproduction/source_contract.json'
PREDECESSOR = 'reproduction/source_contract_A_C_F.json'
PREDECESSOR_SHA = '9f4766216adbc1bfc217994c4c1122ccc874fa927880114ba311395c600ef3c2'
DESIGN = 'B/scripts/research/delivery_prospective_design.py'
AUDITOR = 'B/scripts/research/audit_delivery_prospective_results.py'
GUARD = 'B/scripts/research/runner_delivery_confirmation.py'
REPORTING = 'prepare_delivery_prospective_supplement.py'
VERIFIER = 'verify_delivery_release.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preflight(plan, snapshot, expected=PREDECESSOR_SHA):
    """Run before outputs or accepted score decoding; no contract is a legacy plan.

    Legacy plans cannot acquire a fabricated predecessor/disposition claim. The
    production candidate16 contains ACTIVE and must match the independent pin.
    """
    indexed = {f['path']: f for f in plan['files']}
    require(len(indexed) == len(plan['files']), 'duplicate source-contract input path')
    if ACTIVE not in indexed:
        return None
    require(PREDECESSOR not in indexed, 'source contract already transitioned')
    entry = indexed[ACTIVE]
    require(entry['original_sha256'] == expected and
            entry.get('transform', {'kind': 'identity'}) == {'kind': 'identity'} and
            entry['role'] == 'derived-source-disposition-contract', 'unsupported inherited source contract')
    source = reconcile(entry['sources'], expected)
    value, digest, raw = snapshot(source, expected)
    require(value['schema'] == 'delivery-source-dispositions-v1' and
            value['B_included'] is False and value['training_reproduction_available'] is False and
            value['public_release_approved'] is False and value['anonymity_review_complete'] is False and
            value['runner_grant_authority_resolved'] is True, 'invalid A/C/F predecessor state')
    rows = value['bindings']
    require(len(rows) == 24 and {r['original_location'] for r in rows} == set(ROLES),
            'exact predecessor worker edges required')
    require(all(r['disposition'] == 'private-original-only' and
                r['released_original_path'] is None and r['released_original_sha256'] is None and
                r['anonymous_execution_qualified'] is False for r in rows),
            'predecessor source disposition changed')
    require(len({r['binding_id'] for r in rows}) == 24, 'duplicate predecessor binding identity')
    return {'value': value, 'sha256': digest, 'raw': raw, 'entry': copy.deepcopy(entry)}


def validate_before_scores(plan, prepared, frozen, cc, checked_copies):
    """Pure validation against the planned package bytes, including projections."""
    if prepared is None:
        return
    indexed = {e['path']: e for e in plan['files']}
    old = prepared['value']
    rows = {r['original_location']: r for r in old['bindings']}
    for name, row in rows.items():
        require(row['stage'] == name[0], 'predecessor stage differs')
        if name.startswith('B/'):
            require(row['original_sha256'] == frozen[name]['sha256'],
                    'source-contract original B identity differs')
    require(cc['original_auditor_sha256'] == rows[AUDITOR]['original_sha256'] and
            cc['original_utility_sha256'] == rows[GUARD]['original_sha256'],
            'core original source edges differ')

    def planned_digest(name, identity=False):
        e = indexed[name]
        if identity:
            require(e.get('transform', {'kind': 'identity'}) == {'kind': 'identity'},
                    'included source/interface must have identity bytes')
        checked_copies(e['sources'], e['original_sha256'])
        # The pinned predecessor describes packaged bytes. Runtime projections
        # are separately derived and need not equal original input digests.
        return hashlib.sha256(snapshot_entry(e)).hexdigest()

    interfaces = old['interfaces']
    require(len(interfaces) == 5 and {i['path'] for i in interfaces} == set(INTERFACES),
            'exact predecessor interface set required')
    for interface in interfaces:
        name = interface['path']
        require(planned_digest(name, True) == interface['sha256'], 'inherited interface differs')
        runtimes = interface['runtime_records']
        require(len(runtimes) == len(INTERFACES[name][2]) and
                {r['path'] for r in runtimes} == set(INTERFACES[name][2]),
                'inherited runtime set differs')
        for runtime in runtimes:
            require(planned_digest(runtime['path']) == runtime['sha256'], 'inherited runtime differs')
    for field, path in [('runner_notice_sha256', 'notices/ACE_RUNNER_MIT.txt'),
                        ('ACE_source_notice_sha256', 'notices/ACE_APACHE_2_0.txt')]:
        require(planned_digest(path, True) == old[field], 'inherited source notice differs')
    if 'delivery_prospective_design.py' in indexed:
        require(planned_digest('delivery_prospective_design.py', True) == rows[DESIGN]['original_sha256'],
                'included helper source differs')


def import_description(raw):
    """Describe supported static import statements without executing source."""
    try:
        tree = ast.parse(raw)
    except (SyntaxError, UnicodeError) as error:
        raise ValueError('invalid captured interface source') from error
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append((type(node).__name__, getattr(node, 'module', None),
                            getattr(node, 'level', 0), tuple((a.name, a.asname) for a in node.names)))
    return tree, imports


def validate_reporting_before_scores(plan, reporting_entry, checked_copies, replay_entry):
    """Reporting imports existing package modules; it never imports originals."""
    indexed = {e['path']: e for e in plan['files']}
    name = reporting_entry['path']
    require(not any(p == name or p.startswith(name+'/') or name.startswith(p+'/') for p in indexed),
            'reporting interface collision in inherited plan')
    require(VERIFIER in indexed, 'reporting requires packaged verifier import')
    checked_copies(indexed[ACTIVE]['sources'], indexed[ACTIVE]['original_sha256'])
    verifier = indexed[VERIFIER]
    require(verifier.get('transform', {'kind': 'identity'}) == {'kind': 'identity'},
            'reporting verifier import requires identity bytes')
    checked_copies(verifier['sources'], verifier['original_sha256'])
    reporting_entry['verifier_sha256'] = verifier['original_sha256']
    standard = [('Import', None, 0, ((name, None),)) for name in
                ('argparse', 'csv', 'hashlib', 'json', 'math', 're')]
    standard += [('ImportFrom', 'datetime', 0, (('datetime', None), ('timezone', None))),
                 ('ImportFrom', 'pathlib', 0, (('Path', None),))]
    symbols = ('ARMS', 'DEPENDENCIES', 'HISTORIES', 'WORLDS', 'byte_integrity', 'gate')
    _, imports = import_description(reporting_entry['_raw'])
    expected = standard + [('ImportFrom', 'replay_delivery_prospective_release', 0,
                            tuple((n, None) for n in symbols)),
                           ('ImportFrom', 'verify_delivery_release', 0, (('relative', None),))]
    require(sorted(map(repr, imports)) == sorted(map(repr, expected)),
            'unsupported captured reporting import closure')
    replay_tree, replay_imports = import_description(replay_entry['_raw'])
    replay_expected = [('Import', None, 0, ((n, None),)) for n in
                       ('argparse', 'hashlib', 'importlib.metadata', 'importlib.abc',
                        'importlib.util', 'json', 'math', 're', 'sys', 'time', 'torch')]
    replay_expected += [('Import', None, 0, (('numpy', 'np'),))] * 2
    replay_expected += [('ImportFrom', 'datetime', 0, (('datetime', None),)),
                        ('ImportFrom', 'io', 0, (('BytesIO', None),)),
                        ('ImportFrom', 'pathlib', 0, (('Path', None),)),
                        ('ImportFrom', 'verify_delivery_release', 0,
                         (('relative', None), ('verify', None), ('sha', None))),
                        ('ImportFrom', 'ace.oracle', 0, (('MLPSurrogate', None),))]
    require(sorted(map(repr, replay_imports)) == sorted(map(repr, replay_expected)),
            'unsupported captured reporting replay import closure')
    exports = {n.name for n in replay_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    exports.update(t.id for n in replay_tree.body if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name))
    require(set(symbols) <= exports, 'captured replay reporting exports missing')
    reporting_entry['validated_import_records'] = [
        {'path': replay_entry['path'], 'sha256': replay_entry['sha256'], 'symbols': list(symbols)},
        {'path': VERIFIER, 'sha256': verifier['original_sha256'], 'symbols': ['relative']}]
    reporting_entry['validated_standard_library_imports'] = sorted({
        module if kind == 'ImportFrom' else names[0][0] for kind, module, _, names in standard})
    reporting_entry['validated_replay_imports'] = [
        {'kind': kind, 'module': module, 'level': level,
         'names': [{'name': name, 'alias': alias} for name, alias in names]}
        for kind, module, level, names in replay_imports]


def transition(plan, indexed, prepared, private_dir, input_plan_sha, frozen, cc,
               replay_contract, replay_contract_sha, replay_entry, reporting_entry=None):
    require(reporting_entry is None or prepared is not None and replay_entry is not None,
            'reporting requires explicit replay and inherited source predecessor')
    if prepared is None:
        return {'status': 'legacy input has no source-disposition contract; full disposition remains required'}
    require(replay_entry is not None, 'source-contract transition requires explicit B replay adapter')
    old = prepared['value']; new = copy.deepcopy(old)
    # All B original edges must retain the independently checked source closure.
    for row in new['bindings']:
        if row['stage'] == 'B':
            require(row['original_sha256'] == frozen[row['original_location']]['sha256'],
                    'source-contract original B identity differs')
    design = next(r for r in new['bindings'] if r['original_location'] == DESIGN)
    audit = next(r for r in new['bindings'] if r['original_location'] == AUDITOR)
    guard = next(r for r in new['bindings'] if r['original_location'] == GUARD)
    require(cc['original_auditor_sha256'] == audit['original_sha256'] and
            cc['original_utility_sha256'] == guard['original_sha256'], 'core original source edges differ')
    helper_name = 'delivery_prospective_design.py'
    def identity(name):
        e = indexed[name]
        require(e.get('transform', {'kind': 'identity'}) == {'kind': 'identity'},
                'included source/interface must have identity bytes')
        return e['original_sha256']
    require(identity(helper_name) == design['original_sha256'], 'included helper source differs')
    require(identity('delivery_prospective_replay_core.py') == cc['derived_core_sha256'] and
            identity('B/replay_contract.json') == replay_contract_sha and
            identity(replay_entry['path']) == replay_entry['sha256'], 'B core/runtime/interface identity differs')
    design.update(disposition='identity-source-included-as-replay-helper',
                  released_original_path=helper_name, released_original_sha256=identity(helper_name),
                  next_gate='exact-runtime inference/import qualification and human anonymity review; no collection/training qualification')
    new['schema'] = 'delivery-source-dispositions-v2'
    new['contract_scope'] = 'A/C/F and accepted B saved-artifact replay preparation'
    new['prepared_against_plan_sha256'] = input_plan_sha
    new['predecessor'] = {'path': PREDECESSOR, 'sha256': prepared['sha256'],
                         'scope': 'historical A/C/F-only preparation; superseded current-status assertions'}
    new['B_included'] = True
    new['B_numerical_replay_qualified'] = False
    new['B_upstream'] = {'original_acceptance_sha256': replay_contract['original_acceptance_sha256'],
                        'original_supervisor_sha256': replay_contract['original_supervisor_sha256'],
                        'runtime_record': 'B/replay_contract.json', 'runtime_record_sha256': replay_contract_sha,
                        'exact_dependencies': replay_contract['dependencies']}
    new['derived_sources'] = [{
        'path': 'delivery_prospective_replay_core.py', 'sha256': cc['derived_core_sha256'],
        'role': 'derived exact numerical functions/constants/utilities; excludes original full auditor/launcher',
        'source_bindings': [AUDITOR, GUARD],
        'original_auditor_sha256': audit['original_sha256'],
        'original_utility_sha256': guard['original_sha256'],
        'segment_contract': 'B/core_contract.json', 'segment_contract_sha256': identity('B/core_contract.json'),
    }]
    new['interfaces'].append({
        'path': replay_entry['path'], 'sha256': replay_entry['sha256'],
        'role': 'prospective-replay-adapter', 'scope': '640 saved B checkpoints and registered/descriptive statistics; numerical qualification pending',
        'implementation': 'separately authored release adapter',
        'command': ['python', replay_entry['path'], '--root', '.', '--expected-manifest-sha256', '<independently supplied manifest digest>'],
        'additional_full_replay_arguments': [],
        'runtime_records': [{'path': 'B/replay_contract.json', 'sha256': replay_contract_sha}],
        'does_not_reproduce': ['response collection', 'optimizer trajectories', 'Slurm scheduling', 'original full-audit worker'],
    })
    if reporting_entry is not None:
        require(identity(reporting_entry['path']) == reporting_entry['sha256'] and
                identity(VERIFIER) == reporting_entry['verifier_sha256'],
                'reporting source/import identity differs')
        new['B_descriptive_reporting_qualified'] = False
        new['interfaces'].append({
            'path': reporting_entry['path'], 'sha256': reporting_entry['sha256'],
            'role': 'prospective-descriptive-reporting-adapter',
            'scope': 'complete-matrix descriptive reporting after independently pinned full supplemental replay; preparation only',
            'implementation': 'separately authored release adapter; identity source snapshot included',
            'command': ['python', reporting_entry['path'], '--root', '.', '--manifest-sha256',
                        '<independently supplied manifest digest>', '--replay-receipt',
                        '<external full supplemental replay receipt>', '--replay-sha256',
                        '<independently supplied replay receipt digest>', '--destination',
                        '<exclusive output outside package>'],
            'additional_full_replay_arguments': [],
            'runtime_records': [{'path': 'B/replay_contract.json', 'sha256': replay_contract_sha}],
            'import_records': reporting_entry['validated_import_records'],
            'standard_library_imports': reporting_entry['validated_standard_library_imports'],
            'captured_replay_imports': reporting_entry['validated_replay_imports'],
            'transitive_runtime_imports': {name: replay_contract['dependencies'][name] for name in ('numpy', 'torch')},
            'transitive_learner_import': {'path': 'source/runner/ace/oracle.py',
                'sha256': replay_contract['learner_hashes']['ace/oracle.py'], 'symbols': ['MLPSurrogate']},
            'anonymous_execution_qualified': False, 'import_qualified': False,
            'reporting_execution_qualified': False, 'generated_report_qualified': False,
            'B_numerical_replay_qualified': False,
            'does_not_reproduce': ['response collection', 'optimizer trajectories', 'Slurm scheduling',
                                  'original full-audit worker', 'numerical supplemental replay',
                                  'additional significance tests'],
            'next_gate': 'pinned full target-runtime supplemental replay, reporting/import qualification and human anonymity review',
        })
    new['future_B_extension'] = 'B metadata source transition recorded; exact target-runtime full numerical replay and human approval remain required'
    # Preserve and bind the old bytes under an explicitly historical location.
    target = Path(private_dir)/'source_contract_predecessor.json'
    with target.open('xb') as f:
        f.write(prepared['raw'])
    require(hashlib.sha256(target.read_bytes()).hexdigest() == prepared['sha256'], 'predecessor snapshot differs')
    previous = copy.deepcopy(prepared['entry'])
    previous.update(path=PREDECESSOR, sources=[str(target)], role='historical-source-disposition-predecessor')
    plan['files'].remove(indexed[ACTIVE]); plan['files'].append(previous)
    del indexed[ACTIVE]; indexed[PREDECESSOR] = previous
    for edge in plan['bindings']:
        if edge['record'] == ACTIVE:
            edge['record'] = PREDECESSOR
        if edge['artifact'] == ACTIVE:
            edge['artifact'] = PREDECESSOR
    active_target = Path(private_dir)/'source_contract_B.json'; write(active_target, new)
    active = {'path': ACTIVE, 'sources': [str(active_target)],
              'original_sha256': hashlib.sha256(active_target.read_bytes()).hexdigest(),
              'role': 'derived-source-disposition-contract', 'transform': {'kind': 'identity'}}
    plan['files'].append(active); indexed[ACTIVE] = active
    def bind(pointer, artifact):
        plan['bindings'].append({'record': ACTIVE, 'pointer': pointer, 'artifact': artifact, 'digest': 'sha256'})
    bind('/predecessor/sha256', PREDECESSOR)
    for i, interface in enumerate(new['interfaces']):
        bind(f'/interfaces/{i}/sha256', interface['path'])
        for j, runtime in enumerate(interface['runtime_records']):
            bind(f'/interfaces/{i}/runtime_records/{j}/sha256', runtime['path'])
        for j, imported in enumerate(interface.get('import_records', [])):
            bind(f'/interfaces/{i}/import_records/{j}/sha256', imported['path'])
        if 'transitive_learner_import' in interface:
            imported = interface['transitive_learner_import']
            bind(f'/interfaces/{i}/transitive_learner_import/sha256', imported['path'])
    for field, path in [('runner_notice_sha256', 'notices/ACE_RUNNER_MIT.txt'),
                        ('ACE_source_notice_sha256', 'notices/ACE_APACHE_2_0.txt')]:
        require(identity(path) == new[field], 'inherited source notice differs')
        bind('/'+field, path)
    design_index = new['bindings'].index(design)
    bind(f'/bindings/{design_index}/released_original_sha256', helper_name)
    bind('/B_upstream/runtime_record_sha256', 'B/replay_contract.json')
    bind('/derived_sources/0/sha256', 'delivery_prospective_replay_core.py')
    bind('/derived_sources/0/segment_contract_sha256', 'B/core_contract.json')
    require(new['training_reproduction_available'] is False and new['public_release_approved'] is False and
            new['anonymity_review_complete'] is False and all(r['anonymous_execution_qualified'] is False for r in new['bindings']),
            'source transition cannot qualify execution or public readiness')
    result = {'status': 'source contract superseded; numerical target replay remains unqualified',
            'predecessor_sha256': prepared['sha256'], 'active_contract_sha256': active['original_sha256'],
            'original_edges': 24, 'included_original_helpers': 1, 'interfaces': len(new['interfaces']),
            'B_numerical_replay_qualified': False, 'new_fits': 0, 'new_responses': 0}
    if reporting_entry is not None:
        result['B_descriptive_reporting_qualified'] = False
        result['generated_report_qualified'] = False
    return result

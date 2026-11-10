"""Pure saved-metadata supplement to the closed CPU aid; never acceptance.

Only the pinned stdlib audit aid is loaded. Production sources are read as AST
oracles, never imported or executed. Actual use requires a completed, pinned aid
report for the same independently pinned config and terminal custody.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import posixpath
import re
import resource
import stat
import time
import traceback
import types

AID_SHA = '6d56672582f8c41bdd01c64b446c1d1607316650c6357bec95a889c357ab7eb4'
ORACLE_PINS = {
    'prepare_llm_intervention_scientific_campaign_cpu.py': 'ba439d819f49f6832c97181e4ff727c4813f29031e7b67f8085670f845b0f893',
    'prepare_llm_intervention_campaign_cpu.py': '850ca3d4d7ac4799f68d74bd2d0b1196d10809751a4bdf878633122fc6a9c2c4',
    'qualify_llm_intervention_service_cpu.py': 'd658bedd192d5c3e92f8fc22a052c67d6524df296623ef74e2c5f38bf02fbeb5',
    'llm_intervention_service_runtime.py': '117dc22bd56889906612bff0c975095918e062974fa537757b3e93c981dbfbc0',
    'qualify_llm_intervention_runtime_metadata.py': '31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329',
    'llm_intervention_scientific_campaign.py': 'e3072b8a4c18ef459154fc71ebec8874992d4cee7c829e55b4203cf5111565d8',
}
ENTRY = 'prepare_llm_intervention_scientific_campaign_cpu.py'
DIAGNOSTIC = 'qualify_llm_intervention_runtime_metadata.py'
CPU = 'launch/cpu/'
ART = CPU+'worker/artifacts/'
PHASE = ART+'cpu-phase/'
CAMPAIGN = ART+'campaign/'
POLICIES = ('uniform_random', 'menu_coverage', 'local_information', 'target_risk',
            'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm')
SCREEN_SOURCES = {'llm_intervention_'+n+'.py' for n in
    'contract episode numerical_policy replicated_branches replicated_feedback scm screen_policy screen_protocol supervision target_risk task training_dataset world'.split()}
WORLD_IDS = list(range(202611070100, 202611070124))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return sha(canonical(value).encode())


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def remote_path(value):
    """Normalize saved absolute POSIX paths, never inspect the audit host tree."""
    need(type(value) is str and value.startswith('/') and '\0' not in value, 'absolute saved runtime path required')
    return PurePosixPath(posixpath.normpath(value))


def inside(value, parent):
    p, base = remote_path(value), remote_path(str(parent))
    return p == base or base in p.parents


def constant(source, key):
    """Only literal containers and referenced literal assignments; no source execution."""
    assignments = {}
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    need(target.id not in assignments, 'duplicate static oracle assignment')
                    assignments[target.id] = node.value
    active = set()
    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            need(node.id not in active and node.id in assignments, 'unknown/cyclic static oracle reference')
            active.add(node.id)
            result = value(assignments[node.id]); active.remove(node.id); return result
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            vals = [value(x) for x in node.elts]
            return tuple(vals) if isinstance(node, ast.Tuple) else set(vals) if isinstance(node, ast.Set) else vals
        if isinstance(node, ast.Dict):
            need(all(k is not None for k in node.keys), 'static oracle unpacking forbidden')
            pairs = [(value(k), value(v)) for k, v in zip(node.keys, node.values)]
            need(len(dict(pairs)) == len(pairs), 'duplicate static oracle key')
            return dict(pairs)
        if isinstance(node, ast.Call):
            need(isinstance(node.func, ast.Name) and node.func.id == 'dict' and not node.args,
                 'only keyword-only dict literal construction allowed')
            names = [kw.arg for kw in node.keywords]
            need(all(type(name) is str for name in names) and len(set(names)) == len(names),
                 'dict literal unpacking or duplicate keyword forbidden')
            need(all(isinstance(kw.value, ast.Constant) for kw in node.keywords),
                 'dict construction requires literal keyword values')
            # Interpret only the frozen builtin spelling, never call source code.
            for binding in ast.walk(tree):
                need(not ((isinstance(binding, ast.Name) and isinstance(binding.ctx, ast.Store) and binding.id == 'dict')
                          or (isinstance(binding, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and binding.name == 'dict')
                          or (isinstance(binding, ast.alias) and (binding.asname or binding.name.split('.')[0]) in ('dict', '*'))),
                     'shadowed or wildcard-imported dict constructor forbidden')
            return {kw.arg: value(kw.value) for kw in node.keywords}
        raise ValueError('nonliteral static oracle node')
    return value(ast.Name(id=key, ctx=ast.Load()))


def load_oracles(directory, aid):
    sources = {n: aid.file_bytes(Path(directory).resolve()/n, h, maximum=1024**2)
               for n, h in ORACLE_PINS.items()}
    diagnostic = sources[DIAGNOSTIC]
    return dict(source_pins=ORACLE_PINS, generated_module=constant(diagnostic, 'GENERATED_MODULE'),
        cache_pairs=constant(diagnostic, 'CACHE_PAIRS'), provenance_sources=constant(diagnostic, 'PROVENANCE_SOURCES'),
        bootstrap=constant(diagnostic, 'BOOTSTRAP'), original_runtime=constant(diagnostic, 'RUNTIME_SOURCE'),
        extra_packages=constant(sources['llm_intervention_service_runtime.py'], 'EXTRA_PACKAGES'),
        proposal_sha256=constant(sources[ENTRY], 'PROPOSAL_SHA'),
        proposal_byte_sha256=constant(sources[ENTRY], 'PROPOSAL_BYTE_SHA'))


def check_prefix(prefix, snapshot, generated_module, baseline=None):
    need(remote_path(snapshot['prefix']) == prefix and remote_path(snapshot['base_prefix']) == prefix
         and inside(snapshot['executable'], prefix), 'copied interpreter prefix differs')
    proof = snapshot.get('generated_provenance', {})
    if proof.get('present'):
        need(proof.get('verified') is True, 'generated module provenance rejected')
        need(baseline is not None and snapshot['sys_path'] == baseline+[proof['directory']],
             'unexpected generated search-path delta')
        need(snapshot.get('mixed_modules') == {generated_module: proof['path']}
             and snapshot['mixed_module_paths'] == [proof['path']], 'unexpected generated module exception')
    else:
        need(all(type(p) is str and p.startswith('/') and inside(p, prefix) for p in snapshot['sys_path']),
             'mixed interpreter search paths')
        need(baseline is None or snapshot['sys_path'] == baseline, 'unexpected interpreter search-path delta')
        need(not snapshot['mixed_module_paths'], 'mixed loaded module paths')
    need(not snapshot['mixed_native_paths'], 'mixed loaded native paths')
    need(not any(p.get('version_error') for p in snapshot['packages'].values()), 'package version observation rejected')


def audit_runtime(view, oracle):
    p = view.p; contract = p['runtime']; stage = view.j(CPU+'stage.json'); stage_pin = view.h(CPU+'stage.json')
    bound = view.j(ART+'runtime-stage-binding.json'); prefix = remote_path(stage['staged_prefix'])
    need(not inside(str(prefix), oracle['original_runtime']), 'original environment cannot be the stage')
    need(bound['stage_sha256'] == stage_pin and bound['contract'] == contract and bound['stage'] == stage
         and type(bound.get('gpu_mapping_enabled')) is bool and bound['gpu_mapping_enabled'] is False,
         'runtime stage receipt/device policy differs')
    need(stage.get('version') == 'ace-runtime-archive-v1' and stage.get('archive_sha256') == contract['archive_sha256']
         and stage.get('manifest_sha256') == contract['manifest_sha256'] and stage.get('staged_prefix') == str(prefix)
         and stage.get('objects') == 49966 and stage.get('file_bytes') == 8397575073
         and stage.get('model_loaded') is False and stage.get('runtime_qualified') is False
         and type(stage.get('cache_source_sha256')) is dict, 'archive stage binding differs')
    need(set(stage['cache_source_sha256']) == {x for pair in oracle['cache_pairs'] for x in pair}, 'exact source/cache stage roster differs')
    before = view.j(ART+'runtime-before-imports.json')
    check_prefix(prefix, before, oracle['generated_module'])
    need(before['python_version'] == '3.10.19' and not before['generated_provenance']['present'], 'initial runtime baseline differs')
    metadata = view.j(ART+'runtime-metadata.json')
    need(sorted(r['version'] for r in metadata['numpy_entries']) == ['2.2.5', '2.2.6'], 'both original NumPy metadata versions required')
    need(all(inside(r['metadata_path'], prefix) and inside(r['distribution_root'], prefix) for r in metadata['distributions'])
         and all(inside(r['metadata_path'], prefix) for r in metadata['resolved_defaults'].values()), 'mixed distribution metadata paths')
    candidates = {}
    for row in metadata['distributions']:
        key = re.sub(r'[-_.]+', '-', row['name']).lower()
        candidates.setdefault(key, []).append(row['version'])
    need(all(sorted(candidates.get(k, [])) == v for k, v in contract['metadata_candidates'].items()), 'captured metadata candidates differ')
    provenance = view.j(CPU+'independent-provenance-sources.json')
    need(view.j(ART+'runtime-provenance-sources.json') == provenance, 'independent provenance source capture differs')
    need(sha(provenance['expected_source'].encode('utf-8')) == provenance['expected_source_sha256'], 'generated expected-source text/hash differs')
    need(set(provenance['sources']) == set(oracle['provenance_sources']), 'independent provenance source roster differs')
    for name, (relative, expected) in oracle['provenance_sources'].items():
        row = provenance['sources'][name]
        need(row['path'] == str(prefix/'lib/python3.10/site-packages'/relative) and row['sha256'] == expected,
             'independently pinned provenance source path/hash differs')
    cache = view.j(ART+'cache-reconciliation.json')
    need(cache['no_cache_modified'] is True and len(cache['pairs']) == len(oracle['cache_pairs']), 'cache reconciliation missing/modified')
    for row, (cache_name, source_name) in zip(cache['pairs'], oracle['cache_pairs']):
        need(row['cache_path'] == str(prefix/cache_name) and row['source_path'] == str(prefix/source_name)
             and row['cache_sha256'] == stage['cache_source_sha256'][cache_name]
             and row['source_sha256'] == stage['cache_source_sha256'][source_name]
             and row['matches_stage_pins'] is True and row['admissible_for_diagnosis'] is True
             and row['cache_executed_by_comparison'] is False
             and (row.get('cpython_timestamp_reject_expected') is True or row.get('code_equivalent') is True),
             'cache/source reconciliation differs')
    observed_pins = {}
    for filename in ('runtime-after-service-imports.json', 'runtime-after-cpu-phase.json'):
        observed = view.j(ART+filename)
        check_prefix(prefix, observed, oracle['generated_module'], before['sys_path'])
        need(observed['python_version'] == '3.10.19' and set(observed['packages']) == {'numpy', 'scipy', 'torch', 'transformers'}
             and set(observed['additional_package_versions']) == set(oracle['extra_packages']), 'separate runtime package rosters differ')
        versions = {k: v['version'] for k, v in observed['packages'].items()}; versions.update(observed['additional_package_versions'])
        proof = observed['generated_provenance']
        need(versions == contract['imported_versions'] and proof['verified'] is True
             and proof['temporary_root'] == bound['temporary_root']
             and proof['expected_source_sha256'] == provenance['expected_source_sha256']
             and proof['source_sha256'] == provenance['expected_source_sha256']
             and proof['source_content'] == provenance['expected_source'], 'actual runtime/generated source differs')
        # Reconcile the concrete provenance fields emitted by the closed live checker.
        folder = remote_path(proof['directory']); root = remote_path(bound['temporary_root'])
        path = folder/(oracle['generated_module']+'.py')
        need(folder.parent == root and folder.name.startswith('tmp') and not inside(str(root), prefix)
             and proof['path'] == str(path) and proof['module_file'] == proof['spec_origin'] == str(path)
             and proof['owner_type'] == 'tempfile.TemporaryDirectory', 'generated path/owner/origin differs')
        need(proof['source_pins'] == {n: r['sha256'] for n, r in provenance['sources'].items() if n.startswith('torch.')}, 'generated module source pins differ')
        need(all(proof['temporary_root_stat'][k] == bound['temporary_root_stat'][k] for k in ('dev', 'ino', 'uid', 'gid', 'mode')),
             'generated temporary root identity differs')
        uid = bound['temporary_root_stat']['uid']
        for record in (proof['temporary_root_stat'], proof['directory_stat']):
            need(record['uid'] == uid and stat.S_ISDIR(record['mode']) and stat.S_IMODE(record['mode']) == 0o700,
                 'saved generated directory ownership/mode differs')
        fs = proof['file_stat']
        need(fs['uid'] == uid and stat.S_ISREG(fs['mode']) and fs['nlink'] == 1 and not fs['mode'] & 0o022
             and fs['size'] == len(provenance['expected_source'].encode('utf-8')), 'saved generated file ownership/size differs')
        need(proof['function_origins'] == {n: dict(module=oracle['generated_module'], code_filename=str(path))
             for n in ('forward', 'forward_async', '_remote_forward')}, 'generated function origins differ')
        observed_pins[filename] = view.h(ART+filename)
    return dict(saved_runtime_predicates_passed=True, stage_sha256=stage_pin, observed_runtime_sha256=observed_pins,
        cache_pairs=3, imported_versions=contract['imported_versions'], generated_source_sha256=provenance['expected_source_sha256'],
        runtime_live_inspected=False, generated_source_executed=False, cache_bytecode_executed=False,
        filesystem_semantics='Saved absolute POSIX paths; no Mac resolution of remote paths. Original live ownership/loader observations remain pinned evidence.')


def expected_prefixes(view, oracle):
    p = view.p; prefix = view.j(CPU+'stage.json')['staged_prefix']; stage_pin = view.h(CPU+'stage.json')
    root = remote_path(p['cpu_output']).parent/'inputs'; deadline = view.j(CPU+'worker.started.json')['deadline_monotonic']
    need(type(deadline) in (float, int) and math.isfinite(deadline), 'finite original work deadline required')
    return {key: {kind: [str(remote_path(prefix)/'bin/python'), '-I', '-B', '-c', oracle['bootstrap'],
        str(root/ENTRY), p['files'][ENTRY], '--packet', str(root/'packet.json'), '--packet-sha256', view.packet_pin,
        '--source-commit', p['source_commit'], '--numeric-entry', kind, '--world-id', key,
        '--runtime-stage', str(remote_path(p['cpu_output'])/'stage.json'), '--runtime-stage-sha256', stage_pin,
        '--worker-deadline', str(deadline), '--'] for kind in ('screen', 'ranker')} for key in p['world_pins']}


def policy_specs(p, bindings, prefixes, export, ranker_pin):
    rows = {(r['world_id'], r['policy_id']): r for r in bindings}
    need(len(rows) == len(bindings) == 96, 'exact96 original numerical bindings required')
    expected = {(wid, policy) for wid in WORLD_IDS for policy in POLICIES[:4]}
    need(set(rows) == expected, 'numerical binding world/policy roster differs')
    result = {}; provenance_pin = digest({n: p['files'][n] for n in SCREEN_SOURCES})
    for key in p['world_pins']:
        wid = int(key); specs = []
        for pid in POLICIES[:4]:
            row = rows[wid, pid]
            need(row['binding_sha256'] == digest(row['binding']), 'numerical binding semantic pin differs')
            specs.append(dict(policy_id=pid, kind='task_process', command=prefixes[key]['screen']+[
                '--binding-json', canonical(row['binding']), '--binding-sha256', row['binding_sha256']],
                timeout_seconds=p['numeric_timeout_seconds'], identity_sha256=None, policy_provenance_sha256=provenance_pin))
        specs.append(dict(policy_id='learned_ranker', kind='task_process', command=prefixes[key]['ranker']+[
            '--method', 'learned_ranker', '--seed', '109', '--draws', '32', '--checkpoint', export['path'],
            '--checkpoint-sha256', export['sha256'], '--training-receipt-sha256', export['training_receipt_sha256']],
            timeout_seconds=p['numeric_timeout_seconds'], identity_sha256=None, policy_provenance_sha256=ranker_pin))
        specs.extend(p['persistent_specs'][pid] for pid in POLICIES[5:]); result[wid] = specs
    return result


def audit_bundle(view, oracle):
    p = view.p; bundle = view.j(PHASE+'campaign-bundle.json'); c = bundle['campaign']
    proposal = view.j('launch/inputs/proposal.json'); rows = proposal['development_worlds']
    need(view.h('launch/inputs/proposal.json') == oracle['proposal_byte_sha256']
         and digest(proposal) == oracle['proposal_sha256'], 'original proposal byte/semantic pin differs')
    need([r['world_id'] for r in rows] == WORLD_IDS and set(p['world_pins']) == {str(w) for w in WORLD_IDS}, 'exact24 development identities differ')
    prefixes = expected_prefixes(view, oracle)
    need(view.j(ART+'numeric-prefixes.json') == prefixes, 'original numeric prefixes/deadline differ')
    ranker = view.j(PHASE+'ranker/terminal.json'); final = ranker['exports']['dpo']
    need(final['path'] == 'dpo.ranker.pt', 'final-only learned ranker checkpoint role differs')
    view.h(PHASE+'ranker/'+final['path'], final['sha256'])
    view.h(PHASE+'ranker/training_complete.json', final['training_receipt_sha256'])
    export = dict(path=str(remote_path(p['cpu_output'])/'worker/artifacts/cpu-phase/ranker'/final['path']),
        sha256=final['sha256'], training_receipt_sha256=final['training_receipt_sha256'])
    bindings = view.j('launch/inputs/numeric-bindings.json')
    specs = policy_specs(p, bindings, prefixes, export, view.h(PHASE+'ranker/terminal.json'))
    need(bundle['campaign_sha256'] == digest(c) and c['scope'] == bundle['scope'] == 'development'
         and len(c['sources']) == 24 and c['policy_ids'] == list(POLICIES)
         and c['policy_phases'] == {pid: 'cpu' if pid in POLICIES[:5] else 'gpu' for pid in POLICIES}
         and c['services'] == p['services'] and bundle['proposal_sha256'] == oracle['proposal_sha256']
         and bundle['numeric_policy_bindings'] == bindings and bundle['service_identity_pins'] == p['service_identity_pins']
         and bundle['source_pins'] == p['protocol_source_pins']
         and bundle['world_pins'] == [dict(world_id=r['world_id'], sha256=p['world_pins'][str(r['world_id'])]) for r in rows]
         and bundle['worlds_generated'] == bundle['responses_generated'] == 0
         and all(bundle[k] is False for k in ('execution_authorized', 'scientific_protocol_frozen', 'scientific_provenance_qualified',
             'source_qualified', 'runtime_qualified', 'scheduler_qualified')), 'complete campaign declaration differs')
    need(c['strata_fields'] == ['regime', 'graph_stratum', 'cost_stratum']
         and c['scientific_provenance_qualified'] is c['retry_allowed'] is False, 'whole campaign strata/scope differs')
    arms = list(POLICIES[:4])+['numerical_ranker', 'llm_frozen', 'llm_sft', 'llm_dpo']
    need(bundle['proposal_arm_mapping'] == dict(zip(POLICIES, arms)), 'original arm aliases differ')
    plan_pins = []; descriptor_pins = []
    for i, (source, row) in enumerate(zip(c['sources'], rows)):
        plan = source['plan']; public = row['public']; wid = row['world_id']
        # No simulator, numerical interpretation or display: canonical identities only.
        original_world = view.j('launch/inputs/'+f'descriptor-{i:03d}.json')
        descriptor_pin = digest(original_world)
        need(digest(plan['world']) == descriptor_pin == p['world_pins'][str(wid)]
             and plan['world']['world_id'] == wid, 'supplied descriptor canonical identity differs')
        plan_pin = digest(plan)
        binding = lambda v: {k: v[k] for k in ('request_namespace', 'replicate_id')}
        need(source['plan_sha256'] == plan_pin and plan['policy_specs'] == specs[wid]
             and plan['initial_actions'] == public['initial_actions'] and plan['acquisition_cap_units'] == 24
             and plan['initial_cost_units'] == public['initial_cost_units'] and plan['world_sha256'] == p['world_pins'][str(wid)]
             and plan['task'] == public['task'] and plan['learner'] == dict(graph=public['graph'],
                 noise_variances=public['learner']['noise_variances'], prior_variance=public['learner']['prior_variance'])
             and [e['arm'] for e in row['development_episodes']] == arms
             and plan['initial_binding'] == binding(row['initial_binding'])
             and plan['measurement_bindings'] == {pid: binding(e['measurement_binding']) for pid, e in zip(POLICIES, row['development_episodes'])}
             and source['strata'] == {k: row[k] for k in ('regime', 'graph_stratum', 'cost_stratum')}
             and plan['objective'] == public['objective'] and plan['domain_notes'] == public['domain_notes']
             and plan['scientific_provenance_qualified'] is plan['retry_allowed'] is False
             and plan['menu'] == public['menu'] and plan['outer_provenance_receipt_sha256'] == p['outer_provenance_receipt_sha256'],
             'original development plan/design/namespaces/policies differ')
        need(plan['initial_actions'] == ['observe_all']*8 and plan['initial_cost_units'] == 8
             and len({canonical(v) for v in plan['measurement_bindings'].values()}) == 8, 'shared8/arm-specific namespaces differ')
        plan_pins.append(dict(world_id=wid, sha256=plan_pin)); descriptor_pins.append(dict(world_id=wid, sha256=descriptor_pin))
    need(bundle['source_plan_pins'] == plan_pins, 'all24 original source plan pins differ')
    need(digest(view.j(CAMPAIGN+'campaign.json')) == digest(c)
         and view.j(CAMPAIGN+'source-plan-pins.json') == plan_pins, 'published campaign/source-plan binding differs')
    # Close the argv bridge left by the base aid: completed decisions must use these exact specs.
    for pi, pid in enumerate(POLICIES[:5]):
        base = CAMPAIGN+f'policy-{pi:02d}/'; attempt = view.j(base+'attempt.json')
        need(attempt['campaign_sha256'] == bundle['campaign_sha256'] and attempt['source_plan_pins'] == plan_pins,
             'attempt whole campaign pin differs')
        for wi, wid in enumerate(WORLD_IDS):
            prefix = base+f'world-{wi:02d}/'
            for n in view.full:
                if n.startswith(prefix) and re.search(r'/decision-\d{4}\.process\.json$', n):
                    need(view.j(n)['command'] == specs[wid][pi]['command'], 'executed decision argv differs from frozen plan')
    need(view.j(PHASE+'terminal.json')['campaign_sha256'] == bundle['campaign_sha256'], 'phase whole campaign pin differs')
    return dict(saved_bundle_predicates_passed=True, campaign_sha256=bundle['campaign_sha256'],
        source_plan_pins=plan_pins, descriptor_canonical_pins=descriptor_pins, worlds=24, policies=list(POLICIES),
        declared_cells=192, numeric_policy_bindings=96, ranker_checkpoint_sha256=export['sha256'],
        descriptor_values_emitted=False, private_outcomes_read=False, simulator_or_model_calls=0)


def open_completed_aid(aid, cfg, cfg_pin, report):
    need(report['machine_checks_complete'] is True and report['errors'] == []
         and report['acceptance'] is False and report['scientific_qualified'] is False
         and report['audit_source_sha256'] == AID_SHA and report['config_sha256'] == cfg_pin
         and report['completed_checks'] == ['custody', 'packet', 'ancestor', 'accounting', 'runtime', 'ranker', 'attempts', 'numeric', 'terminal'],
         'completed original aid report for this config required')
    view = aid.Audit(cfg)
    receipt = aid.decode(view.artifact('remote_custody'))
    need(receipt['full_backup_verified'] is True and report['facts']['custody']['archive_sha256'] == receipt['backup_archive_sha256'],
         'same authenticated archive required')
    view.full = receipt['full_inventory']['files']; view.local = receipt['local_inventory']['files']
    view.root = Path(cfg['capsules'][0]).resolve()/'remote'
    view.p = aid.decode(view.artifact('inner')); view.packet_pin = cfg['artifacts']['inner']['sha256']
    need(report['facts']['packet']['inner_sha256'] == view.packet_pin, 'original aid packet differs')
    return view


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('aid-source', 'config', 'config-sha256', 'aid-report', 'aid-report-sha256', 'oracles', 'output'):
        ap.add_argument('--'+key, required=True)
    args = ap.parse_args(); out = Path(args.output).absolute(); out.mkdir(mode=0o700, exist_ok=False)
    result = dict(acceptance=False, scientific_qualified=False, supplement_checks_complete=False, facts={}, findings=[],
        scope='Only previously open saved runtime and campaign-bundle predicates', aid_source_sha256=AID_SHA,
        config_sha256=args.config_sha256, aid_report_sha256=args.aid_report_sha256, oracle_source_pins=ORACLE_PINS)
    started = time.monotonic(); cpu = time.process_time(); view = None
    try:
        source = Path(args.aid_source).read_bytes(); need(sha(source) == AID_SHA, 'closed independent aid source pin differs')
        aid = types.ModuleType('closed_saved_metadata_aid'); aid.__file__ = str(Path(args.aid_source).resolve())
        exec(compile(source, aid.__file__, 'exec'), aid.__dict__)  # Pinned stdlib aid only; its main is not called.
        cfg_raw = aid.file_bytes(Path(args.config).resolve(), args.config_sha256)
        report_raw = aid.file_bytes(Path(args.aid_report).resolve(), args.aid_report_sha256, maximum=128*1024**2)
        cfg = aid.decode(cfg_raw); report = aid.decode(report_raw)
        view = open_completed_aid(aid, cfg, args.config_sha256, report)
        oracle = load_oracles(args.oracles, aid)
        for n, h in ORACLE_PINS.items():
            need(view.p['files'][n] == h, 'actual input source oracle pin differs')
        (out/'input-contract.json').write_bytes(cfg_raw)
        (out/'prior-aid-report.json').write_bytes(report_raw)
        (out/'supplement-source.py').write_bytes(Path(__file__).read_bytes())
        result['facts']['runtime'] = audit_runtime(view, oracle)
        result['facts']['bundle'] = audit_bundle(view, oracle)
        result['supplement_checks_complete'] = True
    except BaseException as exc:
        result['findings'].append(dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    finally:
        result.update(elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
            peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            supplement_source_sha256=sha(Path(__file__).read_bytes()), remote_actions=0, scientific_execution=False,
            necessary_exact_fetches={} if view is None else view.missing,
            remaining_acceptance_scope='Independent actual source/commit/namespace/Slurm/history/cleanup review and retained failures; exact16-key acceptance authored separately.')
        with (out/'report.json').open('xb') as f:
            f.write(raw(result))
    return 0 if result['supplement_checks_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

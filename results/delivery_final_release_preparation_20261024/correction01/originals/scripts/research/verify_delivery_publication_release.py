"""Offline successor checks; private execution evidence is referenced, not proved.

Authenticate this file and the generic verifier independently before execution.
No model, statistical recomputation, network or private custody access.
"""
import argparse
import copy
import json
from pathlib import Path
import re

import delivery_prospective_publication as reports
from verify_delivery_release import relative, pointer, verify

require, digest, decode, pin = reports.require, reports.digest, reports.decode, reports.pin
CANDIDATE_SHA = 'b30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9'
PLAN_SHA = '595fd0a87250322fcb30247276f2b820f4e971d880afd2f8b5a74ba670284a93'
SOURCE_CONTRACT_SHA = '269bbc3f0f8d614177a3e181beb3ebd6e2eaa3fcc40026dba20b0048b998fe65'
REPLAY_ACCEPTANCE_SHA = '57da98e2bbce82faad642442daa491b2cf3722df7524bfd4c881a188a36e3232'
OLD_FILES, OLD_BINDINGS = 3893, 5068
# Terminal failures can never be presented as successful report executions.
# A future positive identity must be bound by independent private acceptance;
# neither a numeric job identifier nor this exclusion list supplies acceptance.
FAILED_REPORT_JOBS = frozenset({'34864950'})
CONTRACT = 'publication/source_contract.json'
PREDECESSOR = 'publication/replay_input_manifest.json'
CHECKER = 'verify_delivery_publication_release.py'
PAPER = 'paper/aistats_ace_2027/'
BASE_COMPANIONS = ('tmlr.sty', 'fancyhdr.sty', 'tmlr.bst', 'delivery_references.bib',
    'delivery_claims.tex', 'delivery_attribution_table.tex', 'delivery_history_table.tex',
    'delivery_physical_table.tex', 'delivery_theory.tex', 'delivery_method_diagram.tex',
    'delivery_evaluation_design.tex')
COMPANIONS = BASE_COMPANIONS + reports.COMPANIONS
STYLES = ('tmlr.sty', 'fancyhdr.sty', 'tmlr.bst', 'TMLR_STYLE_LICENSE',
          'archive/tmlr_official_template.tex')
BASE_PAPER = ('paper.tex', *BASE_COMPANIONS, 'claim_index.json',
              'tmlr_style_provenance.json', 'TMLR_STYLE_LICENSE', 'archive/tmlr_official_template.tex')
REPORT_FILES = (*reports.COMPANIONS, *('delivery_prospective_'+n for n in reports.DATA),
                reports.PRIMARY_INDEX, reports.DESCRIPTIVE_INDEX)
SOURCES = {
    'scripts/research/sync_delivery_manuscript.py': 'source/publication/sync_delivery_manuscript.py',
    'scripts/research/generate_delivery_claims.py': 'source/publication/generate_delivery_claims.py',
    'scripts/research/verify_delivery_table_data.py': 'source/publication/verify_delivery_table_data.py',
    'docs/development/guidance/delivery_reviewer_commands_2026-10-08.md': 'publication/reviewer_commands.md',
    'scripts/research/delivery_prospective_publication.py': 'delivery_prospective_publication.py',
    'scripts/research/export_delivery_prospective_claims.py': 'source/publication/export_delivery_prospective_claims.py',
    'scripts/research/prepare_delivery_prospective_supplement.py': 'prepare_delivery_prospective_supplement.py',
}
SNAPSHOT_MAP = {**{PAPER+n: PAPER+n for n in (*BASE_PAPER, *REPORT_FILES)}, **SOURCES,
    PAPER+reports.PROOFS['manifest']: PREDECESSOR,
    PAPER+reports.PROOFS['contract']: 'B/replay_contract.json',
    **{PAPER+n: None for n in (reports.PROOFS['freeze'], reports.PROOFS['execution'],
                              reports.PROOFS['replay'], reports.BUNDLE)}}
ORIGINAL_INPUTS = ('registration.json', 'acceptance.json', 'scores.json',
                   'complete.json', 'fit_seal.json', 'audit_execution.json')
COUNTS = {'cells': 640, 'fixed_init0': 320, 'system_log_ratios': 80,
          'history_specific_ratios': 8, 'short_fit_ablation': 2, 'primary_floor_counts': 4}
REPORT_OUTPUTS = {**{'primary/'+n: PAPER+n for n in reports.PRIMARY},
    'primary/claim_index.json': PAPER+reports.PRIMARY_INDEX,
    **{'descriptive/'+n: PAPER+n for n in reports.DESCRIPTIVE},
    **{'descriptive/'+n: PAPER+'delivery_prospective_'+n for n in reports.DATA},
    'descriptive/output_index.json': PAPER+reports.DESCRIPTIVE_INDEX}
PRIVATE_KEYS = ('request', 'planner', 'plan', 'bundle', 'snapshot_inventory', 'replay_acceptance',
                'report_acceptance', 'report_custody')
FALSE_FLAGS = ('training_reproduction_available', 'anonymous_execution_qualified',
               'anonymity_review_complete', 'public_release_approved', 'submission_approved')
SOURCE_USAGE = {target: 'provenance-only; not an original-worker execution interface' for target in SOURCES.values()}
SOURCE_USAGE['delivery_prospective_publication.py'] = 'publication-checker dependency; metadata consistency only'
SOURCE_USAGE['prepare_delivery_prospective_supplement.py'] = 'descriptive reporting; independently accepted matching replay required'


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def index(rows):
    require(type(rows) is list, 'file list required')
    result = {r['path']: r for r in rows}
    require(len(result) == len(rows), 'duplicate artifact location')
    return result


def inheritance(old, new):
    """Exact metadata preservation, including original/projected semantics."""
    before, after = index(old['files']), index(new['files'])
    require(old['schema'] == new['schema'] == 'delivery-anonymous-derived-v1', 'manifest schema differs')
    require(len(before) == OLD_FILES and len(old['bindings']) == OLD_BINDINGS, '3893/5068 predecessor required')
    require(all(after.get(n) == e for n, e in before.items()), 'inherited manifest entry changed')
    require(new['bindings'][:OLD_BINDINGS] == old['bindings'], 'inherited bindings changed')
    return before, after


# Only these selectors cross the public boundary. Hash maps have constrained
# basename keys; runtime dictionaries and commands are never copied wholesale.
# The two scientific trees must match the authenticated acceptance tree's exact
# structure/strings and frozen numerical tolerance in _report_and_writing.
SPECS = {
    'replay_freeze': {'/manifest_sha256': 'hash', '/package_files': 'int',
        '/package_bindings': 'int', '/base_python_sha256': 'hash', '/new_fits': 'int', '/new_responses': 'int'},
    'replay_execution': {'/status': 'complete', '/stage': 'qualified', '/exit_code': 'int',
        '/manifest_sha256': 'hash', '/freeze_sha256': 'hash', '/replay_receipt_sha256': 'hash',
        '/new_fits': 'int', '/new_responses': 'int', '/child_execution/status': 'complete',
        '/child_execution/exit_code': 'int', '/child_execution/cleanup/complete': 'true',
        '/authenticated_runtime/environment_objects_verified': 'int'},
    'replay': {'/full_supplemental_replay': 'true', '/integrity/expected_digest_supplied': 'true',
        '/integrity/manifest_sha256': 'hash', '/integrity/files_verified': 'int',
        '/integrity/bindings_verified': 'int', '/checkpoints_replayed': 'int',
        '/primary_cells_recomputed': 'int', '/cached_responses_checked': 'int',
        '/new_optimizer_updates': 'int', '/new_responses': 'int',
        '/primary_analysis': 'analysis', '/secondary_descriptive_analysis': 'analysis'},
    'report_freeze': {'/manifest_sha256': 'hash', '/source_pins': 'sources',
        '/metadata_pins': 'metadata', '/original_input_pins': 'originals',
        '/new_fits': 'int', '/new_optimizer_updates': 'int', '/new_responses': 'int'},
    'report_execution': {'/status': 'complete', '/job_id': 'report-job', '/freeze_sha256': 'hash',
        '/report_receipt_sha256': 'hash', '/child/status': 'complete', '/child/exit_code': 'int',
        '/child/cleanup/complete': 'true', '/new_fits': 'int', '/new_responses': 'int'},
    'report_receipt': {'/status': 'complete', '/freeze_sha256': 'hash', '/source_pins': 'sources',
        '/metadata_pins': 'metadata', '/original_input_pins': 'originals', '/outputs': 'outputs',
        '/new_fits': 'int', '/new_optimizer_updates': 'int', '/new_responses': 'int'},
    'report_runtime': {'/inventory_sha256': 'hash', '/environment_objects': 'int'},
}
for _name in reports.DEPENDENCIES:
    SPECS['replay_freeze']['/exact_dependencies/'+_name] = 'version'
    SPECS['replay']['/runtime/'+_name] = 'version'
for _name in ('runtime_qualification_sha256', 'runtime_inventory_sha256',
              'pre_checkpoint_runtime_sha256', 'python_sha256'):
    SPECS['replay_execution']['/authenticated_runtime/'+_name] = 'hash'
for _name in COUNTS:
    SPECS['report_receipt']['/counts/'+_name] = 'int'
for _kind, _prefix in (('report_receipt', '/statistics_dependencies/'), ('report_runtime', '/dependencies/')):
    for _name in ('numpy', 'scipy'):
        SPECS[_kind][_prefix+_name+'/version'] = 'version'
        SPECS[_kind][_prefix+_name+'/sha256'] = 'hash'


def field(value, kind):
    if kind == 'hash':
        pin(value)
    elif kind == 'analysis':
        require(type(value) is dict, 'scientific analysis object required')
    elif kind == 'int':
        require(type(value) is int and value >= 0, 'nonnegative integer required')
    elif kind == 'true':
        require(value is True, 'true evidence field required')
    elif kind == 'version':
        require(type(value) is str and re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', value), 'version required')
    elif kind == 'report-job':
        require(type(value) is str and re.fullmatch(r'[1-9][0-9]*', value), 'canonical report job identity required')
        require(value not in FAILED_REPORT_JOBS, 'failed report attempt cannot supply success evidence')
    elif kind in ('sources', 'metadata', 'originals', 'outputs'):
        require(type(value) is dict and value, 'nonempty digest map required')
        if kind == 'originals':
            require(set(value) == set(ORIGINAL_INPUTS), 'six original inputs required')
        elif kind == 'outputs':
            require(set(value) == set(REPORT_OUTPUTS), '13 report outputs required')
        else:
            suffix = 'py' if kind == 'sources' else 'json'
            require(all(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*\.'+suffix, n) for n in value),
                    'only source/metadata basenames allowed')
        for v in value.values():
            pin(v)
    else:
        require(value == kind, 'completion field differs')


def project(kind, raw):
    original = decode(raw)
    values = {}
    for selector, rule in SPECS[kind].items():
        value = pointer(original, selector)
        field(value, rule)
        values[selector] = copy.deepcopy(value)
    return {'schema': 'delivery-publication-evidence-projection-v1', 'kind': kind,
            'original_sha256': digest(raw), 'rules': copy.deepcopy(SPECS[kind]), 'values': values,
            'scope': 'derived public fields; complete original evidence remains private'}


def projection(value, kind):
    require(set(value) == {'schema', 'kind', 'original_sha256', 'rules', 'values', 'scope'} and
            value['schema'] == 'delivery-publication-evidence-projection-v1' and value['kind'] == kind,
            'projection schema differs')
    pin(value['original_sha256'])
    require(value['rules'] == SPECS[kind] and set(value['values']) == set(SPECS[kind]),
            'projection selectors differ')
    for selector, rule in SPECS[kind].items():
        field(value['values'][selector], rule)
    require(value['scope'] == 'derived public fields; complete original evidence remains private',
            'projection scope differs')
    return value['values']


def evidence(projections, original_contract, expected_manifest):
    p = {k: projection(projections[k], k) for k in SPECS}
    f, e, r, rf, re_, rr, rt = (p[k] for k in SPECS)
    h = {k: projections[k]['original_sha256'] for k in SPECS}
    require(f['/manifest_sha256'] == e['/manifest_sha256'] == r['/integrity/manifest_sha256'] ==
            rf['/manifest_sha256'] == expected_manifest, 'replay/report manifest lineage differs')
    require(e['/freeze_sha256'] == h['replay_freeze'] and e['/replay_receipt_sha256'] == h['replay'],
            'replay execution lineage differs')
    require(re_['/freeze_sha256'] == rr['/freeze_sha256'] == h['report_freeze'] and
            re_['/report_receipt_sha256'] == h['report_receipt'], 'report execution lineage differs')
    for values in p.values():
        for key, value in values.items():
            if key.endswith(('/exit_code', '/new_fits', '/new_responses', '/new_optimizer_updates')):
                require(type(value) is int and value == 0, 'nonzero work or exit')
    for key, expected in {'/package_files': OLD_FILES, '/package_bindings': OLD_BINDINGS}.items():
        require(f[key] == expected, 'frozen package cardinality differs')
    for key, expected in {'files_verified': OLD_FILES, 'bindings_verified': OLD_BINDINGS}.items():
        require(r['/integrity/'+key] == expected, 'replay membership differs')
    for key, count in {'checkpoints_replayed': 640, 'primary_cells_recomputed': 240,
                       'cached_responses_checked': 48000}.items():
        require(r['/'+key] == count, 'replay coverage differs')
    require(e['/authenticated_runtime/python_sha256'] == f['/base_python_sha256'] and
            e['/authenticated_runtime/environment_objects_verified'] > 0, 'replay runtime differs')
    for name, version in reports.DEPENDENCIES.items():
        require(f['/exact_dependencies/'+name] == r['/runtime/'+name] == version, 'runtime version differs')
    for key in ('source_pins', 'metadata_pins', 'original_input_pins'):
        require(rf['/'+key] == rr['/'+key], 'report frozen input/source mismatch')
    meta = rf['/metadata_pins']
    require(meta['replay_receipt.json'] == h['replay'] and
            meta['replay_acceptance.json'] == REPLAY_ACCEPTANCE_SHA and
            meta['runtime_inventory.json'] == rt['/inventory_sha256'] ==
            e['/authenticated_runtime/runtime_inventory_sha256'], 'report accepted replay/runtime differs')
    require(rt['/environment_objects'] == e['/authenticated_runtime/environment_objects_verified'],
            'report environment membership differs')
    upstream = original_contract['upstream_gate']
    originals = {'registration.json': original_contract['registration_sha256'],
        'acceptance.json': upstream['acceptance_sha256'], 'scores.json': upstream['scores_sha256'],
        'complete.json': original_contract['original_complete_sha256'],
        'fit_seal.json': original_contract['fit_seal_sha256'], 'audit_execution.json': upstream['audit_execution_sha256']}
    require(rf['/original_input_pins'] == originals, 'six historical/projected input edges differ')
    for name in ('numpy', 'scipy'):
        require(rr['/statistics_dependencies/'+name+'/version'] == rt['/dependencies/'+name+'/version'] ==
                reports.DEPENDENCIES[name], 'report statistics versions differ')
        require(rr['/statistics_dependencies/'+name+'/sha256'] == rt['/dependencies/'+name+'/sha256'],
                'report statistics backing hash differs')
    for name, count in COUNTS.items():
        require(rr['/counts/'+name] == count, 'report cardinality differs')
    return p


def _report_and_writing(raw, old, projections, p, primary, descriptive, supplement, contract):
    entries = index(old['files'])
    provenance = descriptive['provenance']
    require(set(primary['output_hashes']) == set(reports.PRIMARY) and
            set(descriptive['output_hashes']) == set(reports.DESCRIPTIVE+reports.DATA), 'report output set differs')
    for names, record in ((reports.PRIMARY, primary), (reports.DESCRIPTIVE+reports.DATA, descriptive)):
        for name in names:
            target = 'delivery_prospective_'+name if name in reports.DATA else name
            require(digest(raw[PAPER+target]) == pin(record['output_hashes'][name]), 'report output hash differs')
    for original, target in REPORT_OUTPUTS.items():
        require(p['report_receipt']['/outputs'][original] == digest(raw[target]), 'report execution output differs')
    require(primary['registration_sha256'] == provenance['original_registration_sha256'] == reports.REGISTRATION
            and primary['source_revision'] == reports.REVISION, 'registered study differs')
    for name in ('acceptance', 'scores'):
        path = 'B/'+name+'.json'; key = name+'_sha256'; edge = contract['metadata_projections'][path]
        require(edge['original_path'] == path and primary[key] == edge['original_sha256'] == contract['original_'+key]
                and provenance[key] == edge['derived_sha256'] == entries[path]['sha256'],
                'historical/projected reporting selector differs')
    for key, source in (('exporter_sha256', SOURCES['scripts/research/export_delivery_prospective_claims.py']),):
        require(primary[key] == digest(raw[source]), 'exporter source differs')
    require(provenance['tool_sha256'] == digest(raw['prepare_delivery_prospective_supplement.py']), 'reporter source differs')
    sources = p['report_receipt']['/source_pins']
    for name, expected in (('export_delivery_prospective_claims.py', primary['exporter_sha256']),
                           ('prepare_delivery_prospective_supplement.py', provenance['tool_sha256'])):
        require(sources[name] == expected, 'executed reporter source differs')
    require(provenance['manifest_sha256'] == p['replay_freeze']['/manifest_sha256'] and
            provenance['supplemental_replay_sha256'] == projections['replay']['original_sha256'], 'report replay lineage differs')
    require(primary['deployment_initialization'] == 0 and type(primary['deployment_initialization']) is int and
            primary['supervision_and_compute_matched'] is False and
            descriptive['actual_numerical_replay_performed_here'] is False, 'report scope differs')
    for name in ('primary_analysis', 'secondary_descriptive_analysis'):
        accepted = decode(raw['B/acceptance.json'])[name]
        reports.match(p['replay']['/'+name], accepted)
        reports.match(primary[name], p['replay']['/'+name])
        reports.match(primary[name], accepted)
    require(supplement['provenance'] == provenance, 'supplement provenance differs')
    for key, count in COUNTS.items():
        require(type(supplement[key]) is list and len(supplement[key]) == count, 'full descriptive data required')
    for key in ('new_fits', 'new_responses', 'additional_significance_tests'):
        reports.integer(supplement[key], 0, 'supplement scope differs')
    begin = '% BEGIN EDITOR COMPANION BUNDLE (generated; edit companion files)'
    end = '% END EDITOR COMPANION BUNDLE'
    text = raw[PAPER+'paper.tex'].decode(); chunks = [begin]
    require(text.count(begin) == text.count(end) == 1 and text.index(begin) < text.index(end), 'single companion bundle required')
    for name in COMPANIONS:
        content = raw[PAPER+name]; decoded = content.decode()
        require('\\end{filecontents*}' not in decoded, 'nested companion terminator')
        chunks += [f'% {name}: SHA256 {digest(content)}', f'\\begin{{filecontents*}}[overwrite]{{{name}}}',
                   decoded.rstrip('\n'), '\\end{filecontents*}']
    chunks.append(end)
    require(text[text.index(begin):text.index(end)+len(end)] == '\n'.join(chunks), '16 companion bytes differ')
    styles = decode(raw[PAPER+'tmlr_style_provenance.json'])['files']
    require(set(styles) == set(STYLES), 'style/notice membership differs')
    for name in STYLES:
        require(styles[name]['sha256'] == digest(raw[PAPER+name]), 'style/notice digest differs')
    require(decode(raw[PAPER+'claim_index.json'])['generator_sha256'] ==
            digest(raw[SOURCES['scripts/research/generate_delivery_claims.py']]), 'current claim source differs')
    require(raw[PAPER+'paper.pdf'].startswith(b'%PDF-'), 'compiled PDF required')


def publication_bindings(contract):
    def escape(name):
        return name.replace('~', '~0').replace('/', '~1')
    return [{'record': CONTRACT, 'pointer': '/artifacts/'+escape(n), 'artifact': n, 'digest': 'sha256'}
            for n in sorted(contract['artifacts'])] + [
        {'record': CONTRACT, 'pointer': '/predecessor/source_contract_sha256',
         'artifact': 'reproduction/source_contract.json', 'digest': 'sha256'}]


def validate_publication(manifest, old_raw, contract, raw):
    require(digest(old_raw) == CANDIDATE_SHA, 'independent candidate03 manifest pin differs')
    old = decode(old_raw); before, after = inheritance(old, manifest)
    require(set(contract) == {'schema', 'authoritative_path', 'predecessor', 'private_evidence',
        'execution_evidence', 'snapshot_members', 'projections', 'artifacts', 'original_source_bindings',
        'source_usage', *FALSE_FLAGS}, 'publication contract fields differ')
    require(contract['schema'] == 'delivery-publication-source-dispositions-v1' and
            contract['authoritative_path'] == CONTRACT, 'authoritative publication contract required')
    require(contract['predecessor'] == {'path': PREDECESSOR, 'sha256': CANDIDATE_SHA,
        'files': OLD_FILES, 'bindings': OLD_BINDINGS, 'source_contract': 'reproduction/source_contract.json',
        'source_contract_sha256': SOURCE_CONTRACT_SHA}, 'predecessor source lineage differs')
    require(before['reproduction/source_contract.json']['sha256'] == SOURCE_CONTRACT_SHA, 'v2 predecessor differs')
    require(all(contract[k] is False for k in FALSE_FLAGS), 'unsupported qualification or approval')
    require(contract['execution_evidence'] == 'accepted privately; not independently revalidated by this public checker',
            'private evidence distinction required')
    private = contract['private_evidence']
    require(set(private) == set(PRIVATE_KEYS), 'private evidence membership differs')
    for record in private.values():
        require(set(record) == {'sha256', 'availability'} and record['availability'] == 'private-custody', 'private evidence path forbidden')
        pin(record['sha256'])
    require(private['plan']['sha256'] == PLAN_SHA and private['replay_acceptance']['sha256'] == REPLAY_ACCEPTANCE_SHA,
            'private predecessor acceptance differs')
    snapshots = contract['snapshot_members']
    require(set(snapshots) == set(SNAPSHOT_MAP) and len(snapshots) == 42, 'complete 42-file disposition required')
    for name, target in SNAPSHOT_MAP.items():
        row = snapshots[name]
        require(set(row) == {'sha256', 'public_path', 'disposition'} and row['public_path'] == target and
                row['disposition'] == ('identity' if target else 'private-only'), 'snapshot disposition differs')
        pin(row['sha256'])
        if target:
            require(after[target]['sha256'] == row['sha256'], 'snapshot public byte binding differs')
    projections = {}
    require(set(contract['projections']) == set(SPECS), 'complete public projections required')
    for kind, row in contract['projections'].items():
        path = 'publication/evidence/'+kind+'.json'
        require(set(row) == {'path', 'sha256', 'original_sha256'} and row['path'] == path and
                row['sha256'] == digest(raw[path]), 'projection byte binding differs')
        projections[kind] = decode(raw[path])
        require(projections[kind]['original_sha256'] == row['original_sha256'], 'projection original binding differs')
    p = evidence(projections, decode(raw['B/replay_contract.json']), CANDIDATE_SHA)
    for name, kind in (('freeze', 'replay_freeze'), ('execution', 'replay_execution'), ('replay', 'replay')):
        require(snapshots[PAPER+reports.PROOFS[name]]['sha256'] == projections[kind]['original_sha256'],
                'private snapshot replay proof differs')
    require(snapshots[PAPER+reports.BUNDLE]['sha256'] == private['bundle']['sha256'], 'assembly snapshot differs')
    artifacts = contract['artifacts']
    required = (set(v for v in SNAPSHOT_MAP.values() if v) | {PAPER+'paper.pdf', CHECKER} |
                {r['path'] for r in contract['projections'].values()})
    require(set(artifacts) == required, 'complete publication artifact set required')
    for name, value in artifacts.items():
        require(after[name]['sha256'] == pin(value), 'publication artifact binding differs')
    require(set(after)-set(before) == required-set(before) | {CONTRACT}, 'unexpected publication addition')
    prior = decode(raw['reproduction/source_contract.json'])
    require(contract['original_source_bindings'] == prior['bindings'] and len(prior['bindings']) == 24,
            'original source relationships changed')
    require(contract['source_usage'] == SOURCE_USAGE, 'source use scope differs')
    require(manifest['bindings'][OLD_BINDINGS:] == publication_bindings(contract), 'publication bindings incomplete')
    exporter = next(r for r in prior['bindings'] if r['original_location'] ==
                    'B/scripts/research/export_delivery_prospective_claims.py')
    require(exporter['original_sha256'] == artifacts[SOURCES['scripts/research/export_delivery_prospective_claims.py']],
            'included original exporter source relationship differs')
    _report_and_writing(raw, old, projections, p, decode(raw[PAPER+reports.PRIMARY_INDEX]),
        decode(raw[PAPER+reports.DESCRIPTIVE_INDEX]), decode(raw[PAPER+'delivery_prospective_supplement.json']),
        decode(raw['B/replay_contract.json']))
    return {'inherited_files': OLD_FILES, 'inherited_bindings': OLD_BINDINGS, 'companions': 16,
        'report_outputs': 13, 'private_execution_revalidated': False, 'public_release_approved': False,
        'scope': 'offline byte, projection-schema and publication-lineage consistency only'}


def check(root, expected_manifest):
    pin(expected_manifest)
    integrity = verify(root, expected_manifest)
    manifest = decode(reports.read(relative(root, 'manifest.json')))
    contract = decode(reports.read(relative(root, CONTRACT)))
    names = set(contract['artifacts']) | {'B/acceptance.json', 'B/replay_contract.json', 'reproduction/source_contract.json'}
    raw = {n: reports.read(relative(root, n)) for n in names}
    require(digest(raw[CHECKER]) == digest(Path(__file__).read_bytes()), 'executing publication checker differs')
    require(digest(raw['delivery_prospective_publication.py']) == reports.SOURCE_SHA256, 'imported publication helper differs')
    result = validate_publication(manifest, raw[PREDECESSOR], contract, raw)
    return {**result, 'integrity': integrity}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.root, args.expected_manifest_sha256), indent=2))

"""Bind source dispositions and qualified replay interfaces into a NEW private plan.

This is documentation/byte-custody preparation, not original-worker relocation,
training reproduction, B acceptance, an anonymity certificate or public approval.
Original worker sources and all inherited artifacts remain unchanged.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from build_delivery_release import reconcile, write
from extend_delivery_notice_plan import captured, outputs
from verify_delivery_release import relative, sha

# Bind the bytes whose compiled implementation is actually executing, rather
# than hashing an independently replaceable pathname at the end of preparation.
_PREPARATION_BYTES = Path(__file__).read_bytes()
if compile(_PREPARATION_BYTES, __file__, 'exec', dont_inherit=True) != sys._getframe().f_code:
    raise ValueError('preparation source snapshot differs from executing implementation')
PREPARATION_SHA256 = hashlib.sha256(_PREPARATION_BYTES).hexdigest()

INVENTORY_SHA = '38b70084bad189fafbf2879aa54c056a1bbd9b9ce9bef21d73304bb773e52ccc'
BASE_PLAN_SHA = 'f2217b20a73d7ba2e47df881a3bfa26c8f1e4c27181c0e1c79929ca4eb6c2551'
REVIEW_SHA = 'cbbe51f73b23bdaadd83590be9300eb3be62ee745a56a80181e14cb6421d12ea'
RUNNER_LICENSE_SHA = '206c4b6a50d0291622cae04f1ec98c6bf64609b53836f727b0d470e00528491c'
ACE_LICENSE_SHA = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'
ROLES = {
    'A/scripts/research/delivery_attribution.py': 'archived input preparation, factorial fitting and evaluation',
    'A/scripts/research/runner_delivery_confirmation.py': 'A protocol guard, utilities and separate confirmation launcher',
    'C/scripts/research/chambers_delivery_validation.py': 'physical protocol freeze, fitting, sealing and evaluation',
    'C/scripts/research/runner_delivery_confirmation.py': 'C protocol guard, utilities and separate confirmation launcher',
    'B/scripts/research/delivery_prospective_batch.py': 'original B freeze, qualification, collection, fitting and barriers',
    'B/scripts/research/delivery_prospective_models.py': 'original online and retrospective fit kernels and inference',
    'B/scripts/research/delivery_prospective_io.py': 'charged response journals, heldout access guard and inference',
    'B/scripts/research/delivery_prospective_design.py': 'world descriptors and reserved action design',
    'B/scripts/research/delivery_prospective_analysis.py': 'registered paired system analysis and four-test Holm',
    'B/scripts/research/delivery_theory.py': 'analytical DAG and support examples',
    'B/scripts/research/runner_delivery_confirmation.py': 'B utilities, guards and separate confirmation launcher',
    'B/scripts/research/audit_delivery_prospective_pilot.py': 'development-only timing and resource qualification',
    'B/scripts/research/delivery_prospective_slurm.py': 'original CPU allocation and exclusive submission',
    'B/scripts/research/test_delivery_prospective_models.py': 'model and calibration development fixtures',
    'B/scripts/research/test_delivery_prospective_io.py': 'response charging and heldout guard development fixtures',
    'B/scripts/research/test_delivery_prospective_batch.py': 'batch membership, barrier and source development fixtures',
    'B/scripts/research/test_delivery_theory.py': 'analytical theory fixtures',
    'B/scripts/research/test_delivery_prospective_analysis.py': 'paired system and statistical fixtures',
    'B/scripts/research/audit_delivery_prospective_results.py': 'mandatory original full independent checkpoint and custody audit',
    'B/scripts/research/test_delivery_prospective_results_audit.py': 'original full audit development fixtures',
    'B/scripts/research/export_delivery_prospective_claims.py': 'original accepted B claim export',
    'B/scripts/research/test_export_delivery_prospective_claims.py': 'original claim export development fixtures',
    'B/baselines.py': 'five-node equation lineage; unrelated historical CLI capabilities',
    'B/experiments/large_scale_scm.py': 'thirty-node graph and coefficient generator lineage',
}
# The actual adapters independently qualify runtime/imports at execution. Merely
# listing them here does not execute, authenticate or extend those qualifications.
INTERFACES = {
    'verify_delivery_release.py': ('verification-tool', 'byte and metadata integrity only', []),
    'replay_delivery_attribution_release.py': ('attribution-replay-adapter', 'A saved model/control scores and mechanism diagnostics', ['A/protocol.json']),
    'replay_delivery_physical_release.py': ('physical-replay-adapter', 'C saved neural/linear predictions and conditional statistics', ['C/protocol.json']),
    'replay_delivery_confirmation_release.py': ('confirmation-replay-adapter', 'F saved online/delivery weights, paired statistics and journals', ['F/protocol.json', 'F/runtime.json']),
    'verify_delivery_claims_release.py': ('empirical-macro-analysis-adapter', '35 A/C/F empirical macros; excludes B and full prose/tables', []),
}


def snapshot_entry(entry):
    # Reconcile every claimed copy, then hash and parse one captured snapshot.
    path = reconcile(entry['sources'], entry['original_sha256'])
    raw = captured(path, entry['original_sha256'])
    transform = entry.get('transform', {'kind': 'identity'})
    if transform == {'kind': 'identity'}:
        return raw
    if entry['path'] not in {'A/protocol.json', 'C/protocol.json', 'F/protocol.json', 'F/runtime.json'} or transform['kind'] != 'project-json':
        raise ValueError('only recorded runtime/protocol JSON projections allowed')
    keys = transform['keys']
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('explicit unique projected keys required')
    original = json.loads(raw)
    return (json.dumps({k: original[k] for k in keys}, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def make_contract(plan, inventory, snapshots):
    files = {f['path']: f for f in plan['files']}
    if len(files) != len(plan['files']):
        raise ValueError('duplicate inherited artifact')
    if any(n.startswith('B/') or n in {'delivery_prospective_design.py', 'delivery_prospective_replay_core.py', 'replay_delivery_prospective_release.py'} for n in files):
        raise ValueError('B inclusion requires a successor source contract')
    if any(f['role'] in {'original-fit-worker', 'original-collection-worker', 'original-full-audit-worker'} for f in files.values()):
        raise ValueError('unsupported original execution role')
    entries = inventory['files']
    if len(entries) != 24 or {f['path'] for f in entries} != set(ROLES):
        raise ValueError('exact 24 original source bindings required')
    if any(f['transformation'] != 'identity' or f['public_release_approved'] is not False for f in entries):
        raise ValueError('original custody status changed')
    if inventory['worker_bindings'] != 24 or len({f['sha256'] for f in entries}) != 22:
        raise ValueError('original source cardinality changed')
    if any(f['original_sha256'] in {s['sha256'] for s in entries} for f in files.values()):
        raise ValueError('original worker unexpectedly included; new disposition review required')
    notice = json.loads(snapshots['notices/provenance.json'])
    if notice['runner_grant_authority_resolved'] is not True or notice['public_release_approved'] is not False:
        raise ValueError('resolved owner notice and private release status required')
    if hashlib.sha256(snapshots['notices/ACE_RUNNER_MIT.txt']).hexdigest() != RUNNER_LICENSE_SHA:
        raise ValueError('owner MIT notice differs')
    if hashlib.sha256(snapshots['notices/ACE_APACHE_2_0.txt']).hexdigest() != ACE_LICENSE_SHA:
        raise ValueError('ACE source notice differs')
    dispositions = []
    for n, f in enumerate(entries, 1):
        dispositions.append({
            'binding_id': f'{n:02d}', 'original_location': f['path'],
            'original_sha256': f['sha256'], 'source_revision': f['source_revision'],
            'protocol_field': f['protocol_field'], 'stage': f['stage'],
            'original_role': ROLES[f['path']], 'disposition': 'private-original-only',
            'source_notice': 'notices/ACE_APACHE_2_0.txt',
            'released_original_path': None, 'released_original_sha256': None,
            'original_transformation': 'identity',
            'known_identifier_screen_positive': f['known_identifier_screen_positive'],
            'anonymous_execution_qualified': False,
            'next_gate': 'separate derived location/import/guard interface or reviewed identity-source inclusion; no historical digest substitution',
        })
    interfaces = []
    for name, (role, scope, runtimes) in INTERFACES.items():
        if files[name]['role'] != role:
            raise ValueError('interface role mismatch')
        interfaces.append({
            'path': name, 'sha256': hashlib.sha256(snapshots[name]).hexdigest(),
            'role': role, 'scope': scope, 'implementation': 'separately authored release adapter',
            'command': ['python', name, '--root', '.', '--expected-manifest-sha256', '<independently supplied manifest digest>'],
            'additional_full_replay_arguments': ['--trust-original-classical-pickles'] if name == 'replay_delivery_attribution_release.py' else [],
            'runtime_records': [{'path': p, 'sha256': hashlib.sha256(snapshots[p]).hexdigest()} for p in runtimes],
            'does_not_reproduce': ['response collection', 'optimizer trajectories', 'Slurm scheduling', 'original full-audit worker'],
        })
    return {'schema': 'delivery-source-dispositions-v1', 'contract_scope': 'A/C/F private preparation; historical B custody disclosed, B release not included',
            'prepared_against_plan_sha256': BASE_PLAN_SHA, 'source_inventory_sha256': INVENTORY_SHA,
            'source_review_sha256': REVIEW_SHA, 'bindings': dispositions, 'interfaces': interfaces,
            'runner_grant_authority_resolved': True, 'runner_notice_sha256': RUNNER_LICENSE_SHA,
            'ACE_source_notice_sha256': ACE_LICENSE_SHA,
            'public_release_approved': False, 'anonymity_review_complete': False,
            'B_included': False, 'B_original_acceptance_required': True,
            'training_reproduction_available': False,
            'future_B_extension': 'supersede this contract with a new digest before adding B; update identity design-helper and derived core/interface dispositions against complete accepted custody and exact target-runtime replay',
            'new_fits': 0, 'new_responses': 0}


def extend(plan_file, expected_plan, inventory_root, review_file, private_dir, out):
    if expected_plan != BASE_PLAN_SHA:
        raise ValueError('only the reviewed candidate12 base plan is supported')
    raw = captured(plan_file, expected_plan); plan = copy.deepcopy(json.loads(raw))
    root = Path(inventory_root).resolve()
    inv_raw = captured(root/'inventory.json', INVENTORY_SHA)
    review_raw = captured(review_file, REVIEW_SHA)
    inventory = json.loads(inv_raw)
    indexed = {f['path']: f for f in plan['files']}
    names = [*INTERFACES, 'A/protocol.json', 'C/protocol.json', 'F/protocol.json', 'F/runtime.json',
             'notices/provenance.json', 'notices/ACE_RUNNER_MIT.txt', 'notices/ACE_APACHE_2_0.txt']
    snapshots = {name: snapshot_entry(indexed[name]) for name in names}
    # Never import/execute archived workers. Reading bytes authenticates custody.
    for entry in inventory['files']:
        captured(relative(root, entry['path']), entry['sha256'])
    contract = make_contract(plan, inventory, snapshots)
    if any(p.startswith('reproduction/') for p in indexed):
        raise ValueError('source contract already present')
    input_paths = [plan_file, root, review_file, *[s for f in plan['files'] for s in f['sources']]]
    private_dir, out = outputs(private_dir, out, input_paths)
    private_dir.mkdir(parents=True, exist_ok=False)
    (private_dir/'input_plan.json').write_bytes(raw)
    (private_dir/'inventory.json').write_bytes(inv_raw)
    (private_dir/'review.md').write_bytes(review_raw)
    target = private_dir/'source_contract.json'; write(target, contract)
    name = 'reproduction/source_contract.json'
    plan['files'].append({'path': name, 'sources': [str(target)], 'original_sha256': sha(target),
                          'role': 'derived-source-disposition-contract', 'transform': {'kind': 'identity'}})
    for i, interface in enumerate(contract['interfaces']):
        plan['bindings'].append({'record': name, 'pointer': f'/interfaces/{i}/sha256',
                                 'artifact': interface['path'], 'digest': 'sha256'})
        for j, runtime in enumerate(interface['runtime_records']):
            plan['bindings'].append({'record': name, 'pointer': f'/interfaces/{i}/runtime_records/{j}/sha256',
                                     'artifact': runtime['path'], 'digest': 'sha256'})
    plan['bindings'].append({'record': name, 'pointer': '/runner_notice_sha256',
                             'artifact': 'notices/ACE_RUNNER_MIT.txt', 'digest': 'sha256'})
    plan['bindings'].append({'record': name, 'pointer': '/ACE_source_notice_sha256',
                             'artifact': 'notices/ACE_APACHE_2_0.txt', 'digest': 'sha256'})
    plan['status'] = 'private source-contract preparation; B acceptance, anonymous original-worker relocation and human approval pending'
    write(out, plan)
    result = {'input_plan_sha256': expected_plan, 'output_plan_sha256': sha(out),
              'contract_sha256': sha(target), 'inventory_sha256': INVENTORY_SHA,
              'review_sha256': REVIEW_SHA, 'bindings': 24, 'interfaces': len(INTERFACES),
              'preparation_tool_sha256': PREPARATION_SHA256,
              'inherited_files': len(indexed), 'inherited_artifacts_modified': False,
              'original_workers_imported_or_modified': False, 'B_outcomes_opened': False,
              'runner_grant_authority_resolved': True, 'public_release_approved': False,
              'new_fits': 0, 'new_responses': 0}
    write(private_dir/'derivation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan-file', 'inventory-root', 'review-file', 'private-dir', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-plan', required=True)
    print(json.dumps(extend(**vars(parser.parse_args())), indent=2))

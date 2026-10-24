"""Prepare a NEW private publication plan; never build or publish a package.

Production use is deliberately blocked until the parent's independently accepted
report receipt schema is supplied and reviewed. No acceptance override is exposed.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re

import delivery_prospective_publication as reports
import prepare_delivery_publication_snapshot as snapshot
import verify_delivery_publication_release as public
from build_delivery_release import reconcile
from verify_delivery_release import relative, sha, project_toml_bytes, identifying_bytes, verify

require, digest, decode, pin, encoded = public.require, public.digest, public.decode, public.pin, public.encoded
INPUTS = {'plan', 'manifest', 'bundle', 'snapshot_inventory', 'replay_acceptance',
          'report_freeze', 'report_execution', 'report_receipt', 'report_runtime',
          'report_custody', 'report_acceptance', 'pdf', 'checker', 'planner'}
REPORT_ACCEPTANCE_SCHEMA = None


def capture(path, expected):
    raw = reports.read(path)
    require(digest(raw) == pin(expected), 'independently pinned input changed')
    return raw


def require_report_acceptance(raw, inputs):
    """Do not mistake a scheduler success or self-consistent packet for acceptance.

    Replace this closed gate only after the parent supplies its actual schema.
    Required validation: the actually successful successor job bound across
    acceptance/execution/custody/scheduler, cleanup, every evidence digest,
    source/runtime lineage, both output indexes and complete membership.
    Failed report01 is permanently ineligible; no report02 success is assumed.
    Tests of downstream pure functions do not implement this missing acceptance.
    """
    value = decode(raw)
    require(value.get('accepted') is True, 'independently accepted report required')
    raise ValueError('report acceptance schema pending parent terminal review; no final plan can be emitted')


def request_inputs(request_file, request_sha):
    request_raw = capture(request_file, request_sha); request = decode(request_raw)
    require(set(request) == {'schema', 'inputs'} and request['schema'] == 'delivery-final-release-inputs-v1',
            'explicit final-release input schema required')
    require(set(request['inputs']) == INPUTS, 'complete independently pinned input set required')
    for value in request['inputs'].values():
        require(set(value) == {'path', 'sha256'} and Path(value['path']).is_absolute(), 'explicit absolute input and pin required')
        pin(value['sha256'])
    inputs = request['inputs']
    require(inputs['manifest']['sha256'] == public.CANDIDATE_SHA and inputs['plan']['sha256'] == public.PLAN_SHA,
            'candidate03 and recovery plan pins required')
    require(inputs['replay_acceptance']['sha256'] == public.REPLAY_ACCEPTANCE_SHA, 'accepted replay12 pin required')
    # This precedes reading either report index, manuscript, package or outcomes.
    acceptance = capture(inputs['report_acceptance']['path'], inputs['report_acceptance']['sha256'])
    require_report_acceptance(acceptance, inputs)
    return request_raw, inputs


def inherited_plan(plan, manifest):
    """Validate exact planned transformation semantics without executing sources."""
    before, planned = public.index(manifest['files']), public.index(plan['files'])
    require(len(before) == public.OLD_FILES and set(planned) == set(before) and
            len(manifest['bindings']) == public.OLD_BINDINGS and plan['bindings'] == manifest['bindings'],
            'exact 3893/5068 inherited plan required')
    for name, entry in planned.items():
        expected = before[name]
        transform = entry.get('transform', {'kind': 'identity'})
        require(entry['original_sha256'] == expected['original_sha256'] and entry['role'] == expected['role']
                and transform == expected['transform'], 'inherited transformation semantics changed')
        source = reconcile(entry['sources'], expected['original_sha256'])
        if transform['kind'] == 'identity':
            require(sha(source) == expected['sha256'] and source.stat().st_size == expected['bytes'], 'inherited bytes changed')
        else:
            raw = reports.read(source)
            if transform['kind'] == 'project-json':
                value = decode(raw); keys = transform['keys']
                require(keys and len(keys) == len(set(keys)), 'unique projection keys required')
                derived = encoded({k: value[k] for k in keys})
            elif transform['kind'] == 'project-toml':
                derived = project_toml_bytes(raw, transform)
            else:
                raise ValueError('unsupported inherited transform')
            require(digest(derived) == expected['sha256'] and len(derived) == expected['bytes'],
                    'inherited projected bytes changed')


def capture_snapshot(root, inventory_raw, bundle_pin):
    inventory = decode(inventory_raw)
    require(inventory['schema'] == 'delivery-publication-source-snapshot-v1' and
            re.fullmatch('[0-9a-f]{40}', inventory['revision']) and
            inventory['prospective_bundle_sha256'] == bundle_pin and
            inventory['companions'] == list(public.COMPANIONS), 'integrated committed snapshot required')
    entries = public.index(inventory['files'])
    require(set(entries) == set(public.SNAPSHOT_MAP) and len(entries) == 42, 'exact prospective snapshot membership required')
    raw = {}
    for name, entry in entries.items():
        raw[name] = capture(relative(root, 'source/'+name), entry['sha256'])
        require(len(raw[name]) == entry['bytes'], 'snapshot size differs')
    snapshot.check_bundle(raw, bundle_pin)
    return raw


def omitted_selectors(value, selected, prefix=''):
    """Private record of omitted leaves; no private selectors enter the release."""
    if prefix in selected:
        return []
    if type(value) is dict:
        return [p for k, v in value.items() for p in omitted_selectors(v, selected,
                prefix+'/'+k.replace('~', '~0').replace('/', '~1'))]
    if type(value) is list:
        return [p for i, v in enumerate(value) for p in omitted_selectors(v, selected, prefix+'/'+str(i))]
    return [prefix]


def publication_payload(snapshot_raw, old_raw, old_contract_raw, evidence_raw, private_pins, pdf, checker):
    """Pure construction after the private acceptance gate; never attests execution.

    Kept separate so fabricated boundary tests need no real report/schema or I/O.
    The production entry point alone authenticates private acceptance and custody.
    """
    require(set(snapshot_raw) == set(public.SNAPSHOT_MAP), 'complete source snapshot required')
    require(set(evidence_raw) == set(public.SPECS) and set(private_pins) == set(public.PRIVATE_KEYS),
            'complete evidence inputs required')
    old = decode(old_raw)
    additions, dispositions = {}, {}
    for name, target in public.SNAPSHOT_MAP.items():
        raw = snapshot_raw[name]
        dispositions[name] = {'sha256': digest(raw), 'public_path': target,
                              'disposition': 'identity' if target else 'private-only'}
        if target:
            require(target not in additions or additions[target] == raw, 'public source alias conflict')
            additions[target] = raw
    require(additions[public.PREDECESSOR] == old_raw, 'snapshot candidate manifest differs')
    additions[public.PAPER+'paper.pdf'] = pdf
    additions[public.CHECKER] = checker
    projections, omissions = {}, {}
    for kind, raw in evidence_raw.items():
        value = public.project(kind, raw); path = 'publication/evidence/'+kind+'.json'
        additions[path] = encoded(value)
        projections[kind] = {'path': path, 'sha256': digest(additions[path]), 'original_sha256': digest(raw)}
        omissions[kind] = omitted_selectors(decode(raw), set(public.SPECS[kind]))
    old_source = decode(old_contract_raw)
    require(old_source['runner_grant_authority_resolved'] is True and len(old_source['bindings']) == 24,
            'resolved notice and original source relationships required')
    contract = {'schema': 'delivery-publication-source-dispositions-v1', 'authoritative_path': public.CONTRACT,
        'predecessor': {'path': public.PREDECESSOR, 'sha256': digest(old_raw), 'files': public.OLD_FILES,
            'bindings': public.OLD_BINDINGS, 'source_contract': 'reproduction/source_contract.json',
            'source_contract_sha256': digest(old_contract_raw)},
        'private_evidence': {k: {'sha256': pin(v), 'availability': 'private-custody'} for k, v in private_pins.items()},
        'execution_evidence': 'accepted privately; not independently revalidated by this public checker',
        'snapshot_members': dispositions, 'projections': projections,
        'artifacts': {n: digest(raw) for n, raw in additions.items()},
        'original_source_bindings': old_source['bindings'],
        'source_usage': copy.deepcopy(public.SOURCE_USAGE),
        **{k: False for k in public.FALSE_FLAGS}}
    additions[public.CONTRACT] = encoded(contract)
    manifest = copy.deepcopy(old)
    for name, raw in additions.items():
        inherited = next((e for e in old['files'] if e['path'] == name), None)
        if inherited:
            require(inherited['sha256'] == digest(raw) and inherited['bytes'] == len(raw), 'inherited publication collision')
        else:
            manifest['files'].append({'path': name, 'sha256': digest(raw), 'original_sha256': digest(raw),
                'bytes': len(raw), 'role': 'publication-source-contract' if name == public.CONTRACT else 'publication-artifact',
                'transform': {'kind': 'identity'}})
    manifest['bindings'].extend(public.publication_bindings(contract))
    return additions, contract, manifest, omissions


def prepare(request_file, request_sha, destination):
    request_raw, inputs = request_inputs(request_file, request_sha)
    originals = {n: capture(v['path'], v['sha256']) for n, v in inputs.items()}
    require(originals['planner'] == Path(__file__).read_bytes() and
            originals['checker'] == Path(public.__file__).read_bytes(), 'executing preparation sources differ')
    destination = snapshot.plain_path(destination)
    origins = [Path(v['path']).resolve().parent for v in inputs.values()]
    require(not destination.exists() and not any(destination.resolve().is_relative_to(p) for p in origins),
            'new exclusive destination outside all inputs required')
    root = Path(inputs['manifest']['path']).parent
    verify(root, public.CANDIDATE_SHA)
    plan, old = decode(originals['plan']), decode(originals['manifest'])
    inherited_plan(plan, old)
    old_source = capture(relative(root, 'reproduction/source_contract.json'), public.SOURCE_CONTRACT_SHA)
    bundle_pin = inputs['bundle']['sha256']
    bundle_raw = reports.capture_bundle(Path(inputs['bundle']['path']).parent, bundle_pin)
    writing = capture_snapshot(Path(inputs['snapshot_inventory']['path']).parent,
                               originals['snapshot_inventory'], bundle_pin)
    require(all(writing[public.PAPER+n] == raw for n, raw in bundle_raw.items()), 'assembly/captured writing differs')
    evidence_raw = {k: originals[k] for k in public.SPECS if k.startswith('report_')}
    evidence_raw.update({kind: bundle_raw[reports.PROOFS[key]] for key, kind in
                        (('freeze', 'replay_freeze'), ('execution', 'replay_execution'), ('replay', 'replay'))})
    additions, contract, manifest, omissions = publication_payload(writing, originals['manifest'], old_source,
        evidence_raw, {k: request_sha if k == 'request' else inputs[k]['sha256'] for k in public.PRIVATE_KEYS},
        originals['pdf'], originals['checker'])
    raw = {**additions, 'reproduction/source_contract.json': old_source,
           'B/acceptance.json': reports.read(relative(root, 'B/acceptance.json'))}
    public.validate_publication(manifest, originals['manifest'], contract, raw)
    final = copy.deepcopy(plan)
    inherited = public.index(old['files'])
    for name, content in additions.items():
        if name not in inherited:
            final['files'].append({'path': name, 'sources': [str(destination/'public'/name)],
                'original_sha256': digest(content), 'transform': {'kind': 'identity'},
                'role': 'publication-source-contract' if name == public.CONTRACT else 'publication-artifact'})
    final['bindings'].extend(public.publication_bindings(contract))
    final['status'] = 'private publication successor preparation; human release/submission approval pending'
    directory = snapshot.exclusive_directory(destination)
    try:
        for name, content in additions.items():
            snapshot.store(directory, 'public/'+name, content)
            require(not identifying_bytes(relative(destination, 'public/'+name)), 'public identifier requires disposition: '+name)
        for name, raw_input in originals.items():
            snapshot.store(directory, 'private/'+name, raw_input)
        for kind, raw_input in evidence_raw.items():
            snapshot.store(directory, 'private/projections/'+kind+'.json', raw_input)
        snapshot.store(directory, 'private/request.json', request_raw)
        snapshot.store(directory, 'private/derivation.json', encoded({'request_sha256': request_sha,
            'input_pins': inputs, 'omitted_selectors': omissions, 'public_contract_sha256': digest(additions[public.CONTRACT]),
            'new_fits': 0, 'new_responses': 0, 'package_built': False}))
        snapshot.store(directory, 'plan.json', encoded(final))
        snapshot.seal_directory(directory)
        info, opened = os.stat(destination, follow_symlinks=False), os.fstat(directory)
        require((info.st_dev, info.st_ino) == (opened.st_dev, opened.st_ino), 'preparation custody location changed')
    finally:
        os.close(directory)
    return {'plan_sha256': digest(encoded(final)), 'contract_sha256': digest(additions[public.CONTRACT]),
            'inherited_files': public.OLD_FILES, 'inherited_bindings': public.OLD_BINDINGS,
            'package_built': False, 'public_release_approved': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.request, args.request_sha256, args.destination), indent=2))

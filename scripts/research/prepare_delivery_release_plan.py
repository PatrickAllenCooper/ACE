"""Prepare an explicit private A/C release plan from completed frozen receipts.

Does not open prospective losses or alter historical records. Stage B is omitted
until its full acceptance; this is a preparation package, not a submission release.
"""
import argparse
import json
from pathlib import Path

from build_delivery_release import write
from verify_delivery_release import sha


def read(path):
    return json.loads(Path(path).read_text())


def require_acceptance(attribution, physical, gate, physical_acceptance):
    """Bind preparation to accepted originals, then recheck A's structural audit."""
    gate = read(gate)
    acceptance = read(physical_acceptance)
    if (gate['custody_audit']['full_acceptance'] is not True or
            gate['complete_receipt_sha256'] != sha(attribution/'complete.json') or
            gate['custody_audit']['registration_sha256'] != sha(attribution/'registration.json')):
        raise ValueError('attribution acceptance binding mismatch')
    from audit_delivery_attribution import audit
    audited = audit(attribution, require_complete=True)
    if audited['verified_receipt_hashes_sha256'] != gate['custody_audit']['verified_receipt_hashes_sha256']:
        raise ValueError('accepted full attribution receipt set changed')
    if acceptance['full_acceptance'] is not True:
        raise ValueError('physical acceptance missing')
    for name, digest in acceptance['receipt_hashes'].items():
        if sha(physical/name) != digest:
            raise ValueError('physical acceptance receipt changed')


def prepare(attribution, physical, out, gate, physical_acceptance):
    attribution, physical = Path(attribution).resolve(), Path(physical).resolve()
    require_acceptance(attribution, physical, gate, physical_acceptance)
    if read(physical_acceptance)['selected_gate_sha256'] != sha(gate):
        raise ValueError('physical selection gate changed')
    registration = read(attribution/'registration.json')
    protocol_root = Path(registration['root'])
    protocol = read(protocol_root/'protocol.json')
    complete = read(attribution/'complete.json')
    if (complete['n_fits'] != 480 or complete['n_histories'] != 12 or complete['new_queries'] != 0
            or complete['registration_sha256'] != sha(attribution/'registration.json')
            or sha(protocol_root/'protocol.json') != registration['protocol_sha256']):
        raise ValueError('accepted complete attribution required')
    cp = read(physical/'protocol.json')
    cd = read(physical/'complete.json')
    cs = read(physical/'fit_seal.json')
    if (len(cp['conditions']) != 11 or len(cs['conditions']) != 11 or
            cd['scores_sha256'] != sha(physical/'scores.json') or
            cd['fit_seal_sha256'] != sha(physical/'fit_seal.json')):
        raise ValueError('accepted complete physical study required')
    if {Path(n).stem for n in cp['conditions']} != set(cs['conditions']):
        raise ValueError('physical condition sets differ')
    files, bindings = [], []
    def add(source, name, role, expected=None, keys=None):
        original = sha(source)
        if expected is not None and original != expected:
            raise ValueError('historical artifact changed: '+name)
        entry = {'path': name, 'sources': [str(source)], 'role': role, 'original_sha256': original}
        if keys is not None:
            entry['transform'] = {'kind': 'project-json', 'keys': keys}
        files.append(entry)
    def bind(record, pointer, artifact, digest='sha256'):
        bindings.append({'record': record, 'pointer': pointer, 'artifact': artifact, 'digest': digest})
    # Include the archived numerical learner, oracle and grid evaluator, rather
    # than assuming a repository checkout supplies this independent dependency.
    source = Path(registration['source'])
    for name, digest in protocol['source_hashes'].items():
        add(source/name, 'source/runner/'+name, 'archived-learner', digest)
    add(protocol_root/'protocol.json', 'A/protocol.json', 'derived-protocol', keys=[
        'stage', 'acquisition_calls', 'model_api_calls', 'source_hashes', 'adapter_sha256',
        'guard_sha256', 'dependencies', 'row_sets', 'epochs', 'inits', 'primary_init',
        'optimizer', 'normalization', 'architecture', 'controls', 'matched_cpu', 'evaluation'])
    for name in protocol['source_hashes']:
        escaped = name.replace('~', '~0').replace('/', '~1')
        bind('A/protocol.json', '/source_hashes/'+escaped, 'source/runner/'+name)
    add(Path(registration['grid']), 'A/grid.npz', 'exposed-evaluation-grid', registration['grid_sha256'])
    cases = sorted({cell['case'] for cell in registration['matrix']})
    if len(cases) != 12 or '124753321' not in cases or len(registration['matrix']) != 480:
        raise ValueError('full twelve-history matrix required')
    for case in cases:
        info = protocol['cases'][case]
        add(protocol_root/case/'input.json', f'A/{case}/input.json', 'paid-input-ledger', info['input_sha256'])
        add(attribution/case/'scores.json', f'A/{case}/scores.json', 'cached-scores', complete['score_hashes'][case])
        bind(f'A/{case}/scores.json', '/grid_file_sha256', 'A/grid.npz')
        # Original online weights and metadata are never refit or reserialized.
        for name, digest in info['original_files'].items():
            if name.endswith('.pt') or name in ('meta.json', 'dataset.csv', 'observations.ndjson'):
                add(Path(info['bundle'])/name, f'A/{case}/original/'+name, 'original-online-artifact', digest)
    seal = read(attribution/'fit_seal.json')
    for cell in registration['matrix']:
        folder = Path(cell['path'])
        name = f"A/{cell['case']}/fits/{folder.name}"
        receipt = read(folder/'receipt.json')
        if not receipt['complete'] or receipt['new_queries'] != 0:
            raise ValueError('incomplete fit')
        add(folder/'receipt.json', name+'/receipt.json', 'fit-receipt', seal['receipts'][str(folder)])
        model = 'models.pt' if cell['arm'] in ('scm', 'flat') else 'regressor.pkl'
        add(folder/model, name+'/'+model, 'fitted-model', receipt['model_sha256'])
        bind(name+'/receipt.json', '/model_sha256', name+'/'+model)
        bind(name+'/receipt.json', '/input_sha256', f"A/{cell['case']}/input.json")
        bind(name+'/receipt.json', '/protocol_sha256', 'A/protocol.json', 'original_sha256')
        bind(f"A/{cell['case']}/scores.json", '/results/'+folder.name+'/receipt_sha256', name+'/receipt.json')
    add(physical/'protocol.json', 'C/protocol.json', 'derived-protocol', keys=[
        'archive_sha256', 'stage_a_gate_sha256', 'conditions', 'development_excluded',
        'selected_delivery', 'source_hashes', 'worker_sha256', 'guard_sha256', 'dependencies',
        'endpoint', 'grouping', 'split', 'epochs', 'init', 'lr', 'online', 'neural_output_scaling',
        'physics', 'fourier', 'uncertainty', 'bootstrap_seed', 'bootstrap_replicates',
        'cpu_core_hour_ceiling', 'threads', 'new_physical_queries', 'rss_bytes',
        'resource_projection_sha256', 'wall_seconds'])
    add(Path(cp['archive']), 'C/archive.zip', 'physical-archive', cp['archive_sha256'])
    add(physical/'scores.json', 'C/scores.json', 'cached-scores', cd['scores_sha256'])
    add(physical/'fit_seal.json', 'C/fit_seal.json', 'fit-seal', cd['fit_seal_sha256'])
    bind('C/protocol.json', '/archive_sha256', 'C/archive.zip')
    bind('C/fit_seal.json', '/protocol_sha256', 'C/protocol.json', 'original_sha256')
    for name, digest in cp['source_hashes'].items():
        if protocol['source_hashes'].get(name) != digest:
            raise ValueError('A/C archived source differs')
        bind('C/protocol.json', '/source_hashes/'+name.replace('/', '~1'), 'source/runner/'+name)
    for condition, summary in cs['conditions'].items():
        required = {'models.pt', 'linear_coefficients.json', 'predictions.npz'}
        if cp['selected_delivery'] not in ('scm', 'flat'):
            required.add('selected_regressor.pkl')
        if set(summary['artifact_hashes']) != required:
            raise ValueError('physical artifacts missing or unexpected')
        for name, digest in summary['artifact_hashes'].items():
            add(physical/condition/name, f'C/{condition}/{name}', 'physical-artifact', digest)
            bind('C/fit_seal.json', '/conditions/'+condition+'/artifact_hashes/'+name, f'C/{condition}/{name}')
    tool = Path(__file__).with_name('verify_delivery_release.py')
    add(tool, 'verify_delivery_release.py', 'verification-tool')
    replay = Path(__file__).with_name('replay_delivery_physical_release.py')
    add(replay, 'replay_delivery_physical_release.py', 'physical-replay-adapter')
    replay = Path(__file__).with_name('replay_delivery_attribution_release.py')
    add(replay, 'replay_delivery_attribution_release.py', 'attribution-replay-adapter')
    write(out, {'status': 'A/C preparation only; B pending; public release and human anonymity review not approved',
                'files': files, 'bindings': bindings})
    return {'planned_files': len(files), 'bindings': len(bindings), 'stage_a_fits': 480,
            'stage_c_conditions': 11, 'stage_b_included': False, 'plan_sha256': sha(out)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('attribution', 'physical', 'out', 'gate', 'physical-acceptance'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.attribution, args.physical, args.out, args.gate, args.physical_acceptance), indent=2))

"""Read-only custody audit, including live prefixes; never load outcome arrays.

Independent of the frozen fit adapter. Final acceptance also requires the full
matrix seal and all twelve score hashes. A partial audit is not an outcome gate.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from runner_delivery_confirmation import read, write, sha, utc


def require(condition, message):
    if not condition:
        raise ValueError(message)


def eligible_indices(data, row_set, head):
    meta = data['meta']
    nonroots = {n for n, parents in meta['causal_dag'].items() if parents}
    parents = meta['feature_names'] if head == 'flat' else meta['causal_dag'][head]
    target = meta['target_name'] if head == 'flat' else head
    by_id = {r['query_index']: r for r in data['rows']}
    require(len(by_id) == len(data['rows']), 'duplicate paid query ID')
    result = []
    for index in data['selection'][row_set]:
        row = by_id[index]
        interventions = set(row['interventions'])
        if (head == 'flat' and interventions & nonroots) or (head != 'flat' and target in interventions):
            continue
        values = {**row['params'], **row['intermediates'], meta['target_name']: row['outcome']}
        require(all(math.isfinite(float(values[n])) for n in [*parents, target]), 'nonfinite eligible row')
        result.append(index)
    return result


def verify_receipt(cell, receipt, reg, case_protocol, data):
    for key in ('case', 'row_set', 'arm', 'epochs', 'init'):
        require(receipt[key] == cell[key], 'receipt configuration mismatch: ' + key)
    require(receipt['complete'] is True and receipt['new_queries'] == 0 and receipt['evaluation'] is None,
            'fit receipt not complete or crossed outcome/query boundary')
    require(receipt['protocol_sha256'] == reg['protocol_sha256'], 'fit protocol changed')
    require(receipt['input_sha256'] == case_protocol['input_sha256'], 'fit inputs changed')
    require(receipt['threads'] == reg['threads'], 'fit thread request changed')
    require(0 < receipt['peak_rss_bytes'] <= reg['rss_bytes'], 'fit RSS ceiling')
    expected = set(data['meta']['node_mlps']) if cell['arm'] == 'scm' else {'flat'}
    require(set(receipt['heads']) == expected, 'missing/extra mechanism heads')
    for head, stats in receipt['heads'].items():
        indices = eligible_indices(data, cell['row_set'], head)
        require(len(indices) == stats['eligible_rows'], 'eligibility count mismatch')
        digest = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
        require(digest == stats['query_indices_sha256'], 'eligibility query digest mismatch')
        updates = stats['optimizer_updates']
        if not cell.get('matched_cpu'):
            require(updates == (cell['epochs'] if cell['arm'] in ('scm', 'flat') else 0), 'optimizer update mismatch')
        else:
            require(updates > 0, 'no matched CPU optimizer updates')
        for key in ('fit_cpu_seconds', 'fit_wall_seconds'):
            require(math.isfinite(stats[key]) and stats[key] >= 0, 'invalid fit timing')
        require(stats['parameters_or_tree_nodes'] > 0, 'invalid model size')
    require(math.isclose(sum(h['fit_cpu_seconds'] for h in receipt['heads'].values()),
                         receipt['fit_cpu_seconds'], rel_tol=1e-10, abs_tol=1e-8), 'summed CPU mismatch')
    require(math.isfinite(receipt['worker_wall_seconds']) and receipt['worker_wall_seconds'] >= 0,
            'invalid worker wall time')
    if not cell.get('matched_cpu'):
        require(receipt['matched_cpu_seconds'] is None, 'unexpected matched CPU budget')


def audit(folder, require_complete=False):
    folder = Path(folder)
    reg = read(folder/'registration.json')
    digest = sha(folder/'registration.json')
    require(read(folder/'started.json')['registration_sha256'] == digest, 'start registration changed')
    root = Path(reg['root'])
    require(sha(root/'protocol.json') == reg['protocol_sha256'], 'input protocol changed')
    protocol = read(root/'protocol.json')
    here = Path(__file__).parent
    for file, key in [('delivery_attribution.py', 'adapter_sha256'), ('delivery_attribution_batch.py', 'batch_sha256')]:
        require(sha(here/file) == reg[key], 'frozen worker changed: ' + file)
    require(sha(here/'runner_delivery_confirmation.py') == protocol['guard_sha256'], 'resource guard changed')
    require(sha(reg['grid']) == reg['grid_sha256'], 'exposed grid changed')
    for file, expected in protocol['source_hashes'].items():
        require(sha(Path(reg['source'])/file) == expected, 'frozen runner source changed')
    matrix = reg['matrix']
    require(len(matrix) == 480 and len({c['path'] for c in matrix}) == 480, 'matrix size/duplicate paths')
    cases = Counter(c['case'] for c in matrix)
    require(len(cases) == 12 and set(cases.values()) == {40} and '124753321' in cases,
            'history count or worsening case missing')
    for case in cases:
        # Compare full configurations, not just the row count in a receipt.
        expected = [(s, a, e, i, False) for s in ('final_buffer', 'online_admitted', 'all_paid')
                    for a in ('scm', 'flat') for e in (100, 30000) for i in (0, 1, 2)]
        expected += [('all_paid', a, 100, 0, False) for a in ('ridge', 'polynomial', 'tree')]
        expected += [('all_paid', 'flat', 30000, 0, True)]
        actual = [(c['row_set'], c['arm'], c['epochs'], c['init'], bool(c.get('matched_cpu')))
                  for c in matrix if c['case'] == case]
        require(Counter(actual) == Counter(expected), 'unregistered matrix cell')
    data = {}
    originals = 0
    for case in cases:
        info = protocol['cases'][case]
        require(sha(root/case/'input.json') == info['input_sha256'], 'input ledger changed')
        data[case] = read(root/case/'input.json')
        for file, expected in info['original_files'].items():
            require(sha(Path(info['bundle'])/file) == expected, 'original artifact changed: ' + case + '/' + file)
            originals += 1
    execution = read(folder/'execution.json')
    attempts = execution['attempts']
    require(len(attempts) <= 480, 'too many attempts')
    receipts = {}
    cpu = 0.
    peak = 0
    failures = []
    for index, attempt in enumerate(attempts):
        cell = matrix[index]
        require(attempt['cell'] == cell, 'execution prefix out of order or duplicate')
        if attempt['status'] != 'complete':
            failures.append({'cell': cell, 'status': attempt['status']})
            continue
        path = Path(cell['path'])
        receipt = read(path/'receipt.json')
        verify_receipt(cell, receipt, reg, protocol['cases'][cell['case']], data[cell['case']])
        model = path/('models.pt' if cell['arm'] in ('scm', 'flat') else 'regressor.pkl')
        require(sha(model) == receipt['model_sha256'], 'fitted model changed')
        if cell.get('matched_cpu'):
            budget = read(path.parent/'all_paid-scm-30000-i0'/'receipt.json')['fit_cpu_seconds']
            require(receipt['matched_cpu_seconds'] == budget and receipt['fit_cpu_seconds'] >= budget,
                    'matched CPU arm received different budget')
        receipts[cell['path']] = sha(path/'receipt.json')
        cpu += receipt['fit_cpu_seconds']
        peak = max(peak, receipt['peak_rss_bytes'], attempt['peak_tree_rss_bytes'])
    require(peak <= reg['rss_bytes'], 'supervised process tree exceeded RSS ceiling')
    require(cpu/3600 <= 80, 'fit CPU exceeds Stage A ceiling')
    sealed = (folder/'fit_seal.json').exists()
    score_files = list(folder.glob('*/scores.json'))
    if sealed:
        seal = read(folder/'fit_seal.json')
        require(len(receipts) == 480 and not failures, 'seal without complete fit matrix')
        require(seal['registration_sha256'] == digest and seal['receipts'] == receipts, 'full receipt seal mismatch')
    else:
        require(not score_files, 'scores produced before full matrix seal')
    complete = (folder/'complete.json').exists()
    if complete:
        done = read(folder/'complete.json')
        require(sealed and done['n_fits'] == 480 and done['n_histories'] == 12 and done['new_queries'] == 0,
                'invalid completion receipt')
        require(done['registration_sha256'] == digest and set(done['score_hashes']) == set(cases), 'completion binding')
        require(math.isclose(done['fit_cpu_core_hours'], cpu/3600, rel_tol=1e-10), 'complete CPU accounting')
        for case, expected in done['score_hashes'].items():
            require(sha(folder/case/'scores.json') == expected, 'score hash mismatch')
            scores = read(folder/case/'scores.json')['results']
            labels = {Path(c['path']).name for c in matrix if c['case'] == case}
            require(set(scores) == labels, 'incomplete scored matrix')
            for c in (c for c in matrix if c['case'] == case):
                require(scores[Path(c['path']).name]['receipt_sha256'] == receipts[c['path']], 'score/fit binding')
    require(not require_complete or complete, 'final acceptance requires all twelve evaluations')
    return {'at': utc(), 'audit': 'read-only attribution custody', 'valid_snapshot': True,
            'full_acceptance': complete, 'registration_sha256': digest,
            'audit_worker_sha256': sha(__file__), 'fit_source_revision': reg['ace_source_revision'],
            'verified_receipt_hashes_sha256': hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest(),
            'execution_snapshot_sha256': hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest(),
            'n_expected_fits': 480, 'n_verified_completed_fits': len(receipts), 'failed_attempts': failures,
            'fit_sealed': sealed, 'original_files_verified': originals,
            'fit_cpu_core_hours_so_far': cpu/3600, 'peak_observed_rss_bytes': peak,
            'outcome_scores_read': complete, 'new_simulator_queries': 0,
            'limitation': 'fit CPU excludes process startup/imports/evaluation; shared wall watchdog caps the full batch'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    write(args.out, audit(args.folder, args.require_complete))

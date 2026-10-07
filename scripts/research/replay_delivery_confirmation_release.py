"""Read-only confirmation checkpoint/statistic/attempt reconstruction.

An authenticated manifest and exact numerical dependencies are required.
Twelve complete histories and the separate interrupted acquisition are retained.
Charged reservations are not inferred to be returned responses. No refitting,
new responses, historical receipt changes, B acceptance or public release.
"""
import argparse
from collections import Counter
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
from verify_delivery_release import verify, sha
from replay_delivery_attribution_release import FROZEN_HISTORIES, compare

ROLES = ('seed', 'observational', 'interventional_val', 'obs_refresh', 'lookahead', 'teacher', 'breaker', 'baseline')


def read(path):
    return json.loads(Path(path).read_text())


def journal_count(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows or any(type(r.get('attempt')) is not int or r['attempt'] != i+1
                       or set(r) != {'attempt', 'at'} or not isinstance(r['at'], str)
                       for i, r in enumerate(rows)):
        raise ValueError('malformed, duplicate or reordered charged attempt journal')
    return len(rows)


def accounting(root):
    root = Path(root)
    protocol, seal = read(root/'F/protocol.json'), read(root/'F/sealed.json')
    cases = root/'F/cases'
    if {p.name for p in cases.iterdir()} != FROZEN_HISTORIES:
        raise ValueError('complete confirmation cohort changed')
    if {str(p.relative_to(cases)): sha(p) for p in cases.rglob('*') if p.is_file()} != seal['case_hashes']:
        raise ValueError('original full confirmation case seal differs')
    calls, persisted = {}, 0
    for history in sorted(FROZEN_HISTORIES):
        folder = cases/history; n = journal_count(folder/'calls.jsonl')
        done, meta = read(folder/'complete.json'), read(folder/'online/meta.json')
        rows = [json.loads(line) for line in (folder/'online/observations.ndjson').read_text().splitlines()]
        if (type(done['seed']) is not int or str(done['seed']) != history or done['inits'] != [0, 1, 2]
                or n != done['calls'] or len(rows) != n
                or any(type(r['query_index']) is not int or r['query_index'] != i or r['method'] != 'ace'
                       for i, r in enumerate(rows))):
            raise ValueError('complete history journal/response/config membership differs')
        import hashlib
        if hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest() != done['rows_sha256']:
            raise ValueError('persisted response digest differs')
        observed, counters = Counter(r['role'] for r in rows), meta['query_counts']['ace']
        if (set(observed)-set(ROLES) or set(counters)-set(ROLES)-{'total', 'startup', 'executed'}
                or any(type(v) is not int or v < 0 for v in counters.values())
                or counters['total'] != n or sum(counters.get(k, 0) for k in ROLES) != n
                or any(counters.get(k, 0) != observed.get(k, 0) for k in ROLES)
                or counters['startup'] != sum(observed.get(k, 0) for k in ROLES[:3])
                or counters['executed'] != sum(bool(r.get('selected')) for r in rows)):
            raise ValueError('paid role counts or overlapping counters differ')
        if n > protocol['resource_proposal']['call_cap_per_case']:
            raise ValueError('per-attempt call ceiling')
        calls[history] = n; persisted += len(rows)
    interrupted = root/'F/attempts/884825602-interrupted'
    if {p.name for p in (root/'F/attempts').iterdir()} != {'884825602-interrupted'} or {p.name for p in interrupted.iterdir()} != {'calls.jsonl'}:
        raise ValueError('interrupted attempt missing or merged with a response history')
    reserved = journal_count(interrupted/'calls.jsonl')
    if reserved > protocol['resource_proposal']['call_cap_per_case']:
        raise ValueError('interrupted attempt ceiling')
    prior, original, terminal = read(root/'F/prior_terminal.json'), read(root/'F/original_terminal.json'), read(root/'F/terminal.json')
    retained = set(FROZEN_HISTORIES)-{'884825602'}
    if {str(c['seed']) for c in prior['completed_cases']} != retained or len(prior['completed_cases']) != 11:
        raise ValueError('retained attempt identities differ')
    retained_calls = sum(calls[s] for s in retained); aggregate = persisted+reserved
    if (prior['complete_calls'] != retained_calls or prior['partial_journal_calls'] != reserved
            or prior['partial_seed'] != 884825602 or prior['aggregate_calls'] != retained_calls+reserved
            or original['calls'] != calls['27424209'] or original['first_case']['seed'] != 27424209
            or terminal['aggregate_calls_including_discarded'] != aggregate
            or read(root/'F/execution.json')['aggregate_calls'] != aggregate
            or aggregate > protocol['resource_proposal']['total_call_cap']):
        raise ValueError('stage or aggregate accounting disagrees with distinct journals')
    return {'complete_histories': 12, 'distinct_acquisition_attempts': 13,
            'persisted_complete_responses': persisted, 'interrupted_charged_reservations': reserved,
            'aggregate_charged_attempts': aggregate, 'prior_retained_complete_responses': retained_calls,
            'prior_aggregate_charged_attempts': retained_calls+reserved,
            'calls_per_complete_history': calls,
            'interrupted_returned_responses': 'unknown; no persisted response history',
            'retained_copies_counted_once': True, 'refit_initializations_add_acquisition_calls': False}


def replay(root, expected):
    root = Path(root).resolve(); integrity = verify(root, expected)
    reg, runtime, scores, independent = (read(root/'F'/n) for n in ('protocol.json', 'runtime.json', 'scores.json', 'statistics.json'))
    if set(map(str, reg['seeds'])) != FROZEN_HISTORIES or reg['delivery']['inits'] != [0, 1, 2]:
        raise ValueError('frozen cohort/initialization membership changed')
    for name in ('torch', 'numpy', 'scipy'):
        if importlib.metadata.version(name) != runtime['dependency_versions'][name]:
            raise ValueError('confirmation dependency differs: '+name)
    counts = accounting(root)
    if any(name == 'ace' or name.startswith('ace.') for name in sys.modules):
        raise ValueError('fresh learner process required')
    sys.path.insert(0, str(root/'source/runner'))
    import numpy as np
    import torch
    from ace.grid_eval import GridTruth, chain_predict, flat_predict, score
    from ace.oracle import MLPSurrogate
    torch.set_num_threads(1)
    wall0, cpu0 = time.monotonic(), time.process_time()
    truth = GridTruth.load(root/'A/grid.npz', target='engagement_rate', env_id=reg['environment'])
    if truth.sha256 != reg['evaluation']['array_sha256'] or truth.n_points != reg['evaluation']['points']:
        raise ValueError('confirmation exposed grid identity differs')
    def predictor(state):
        model = MLPSurrogate.from_state_dict(state).eval()
        def predict(x):
            with torch.inference_mode():
                k = reg['evaluation']['inference_chunk_rows']
                result = np.concatenate([model(torch.tensor(x[i:i+k], dtype=torch.float32)).numpy()
                                         for i in range(0, len(x), k)])
            if not np.isfinite(result).all():
                raise ValueError('nonfinite saved prediction')
            return result
        return predict
    pairs, heads = [], 0
    for seed in reg['seeds']:
        case = root/'F/cases'/str(seed); meta = read(case/'online/meta.json'); dag = meta['causal_dag']
        if meta['feature_names'] != list(truth.feature_names) or meta['target_name'] != truth.target:
            raise ValueError('confirmation schema differs')
        def metrics(states):
            expected_heads = {'flat'} | {n for n, parents in dag.items() if parents}
            if set(states) != expected_heads:
                raise ValueError('saved model head membership differs')
            return {'chain': score(chain_predict({n: predictor(s) for n, s in states.items() if n != 'flat'}, dag, truth), truth.y, truth.levels()),
                    'flat': score(flat_predict(predictor(states['flat']), truth), truth.y, truth.levels())}
        original = {'flat': torch.load(case/'online/mlp.pt', weights_only=True, map_location='cpu')}
        original.update({p.stem: torch.load(p, weights_only=True, map_location='cpu') for p in (case/'online/mlps').glob('*.pt')})
        online = metrics(original); heads += len(original); fitted = []
        for init in reg['delivery']['inits']:
            states = torch.load(case/f'init-{init}/models.pt', weights_only=True, map_location='cpu')
            fitted.append({'init': init, **metrics(states)}); heads += len(states)
        pairs.append({'seed': seed, 'online': online, 'delivery': fitted})
    maximum = compare(pairs, scores['pairs'], 'confirmation/pairs')
    spec = importlib.util.spec_from_file_location('archived_confirmation_stats', root/'source/runner/scripts/analysis/study_stats.py')
    stats = importlib.util.module_from_spec(spec); sys.modules[spec.name] = stats; spec.loader.exec_module(stats)
    floor = reg['evaluation']['error_floor']
    a = {r['seed']: statistics.median(max(1-f['chain']['exact'], floor) for f in r['delivery']) for r in pairs}
    b = {r['seed']: max(1-r['online']['chain']['exact'], floor) for r in pairs}
    comparison = stats.paired_log_ratio(a, b, floor=floor, level=reg['analysis']['confidence_level'])
    maximum = max(maximum, compare(comparison, scores['comparison'], 'confirmation/comparison'))
    for key, value in [('n', comparison['n']), ('ratio', comparison['r']), ('ci_lo', comparison['ci_lo']),
                       ('ci_hi', comparison['ci_hi']), ('exact_sign_flip_p', comparison['perm_p'])]:
        maximum = max(maximum, compare(value, independent[key], 'independent/'+key))
    confirmed = comparison['r'] <= reg['analysis']['required_ratio_max'] and comparison['ci_hi'] < 1 and comparison['perm_p'] < reg['analysis']['alpha']
    if confirmed != scores['confirmed']:
        raise ValueError('historical verdict differs')
    return {'integrity': integrity, 'accounting': counts, 'model_sets_replayed': len(pairs)*4,
            'neural_heads_replayed': heads, 'complete_histories': len(pairs),
            'maximum_absolute_numeric_difference': maximum, 'comparison_tolerance': {'rtol': 1e-10, 'atol': 1e-12},
            'statistics': comparison, 'historical_confirmed': confirmed,
            'new_optimizer_updates': 0, 'new_responses': 0,
            'replay_cpu_seconds': time.process_time()-cpu0, 'replay_wall_seconds': time.monotonic()-wall0,
            'scope': 'original online plus three-init confirmation inference/statistics/charged accounting; no refitting, historical freeze proof, B acceptance or public release'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.root, args.expected_manifest_sha256), indent=2))

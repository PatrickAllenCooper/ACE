"""Prospective primary analysis: forty systems, two histories, init0 only.

No per-row, per-history or per-initialization pseudo-replication. This module
validates the full registered primary matrix before computing any contrasts.
Incomplete/failed fits block superiority; they cannot be silently dropped.
"""
import math
from collections import defaultdict
from delivery_theory import holm

HISTORIES = {'balanced_varied_value', 'matched_random'}
ARMS = {'delivery', 'online', 'simpler'}
CONTRASTS = [(size, control) for size in (5, 30) for control in ('online', 'simpler')]


def summarize_logs(values):
    import numpy as np
    from scipy import stats
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('need finite paired system log ratios')
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    if se == 0:
        p, half = (1. if mean == 0 else 0.), 0.
    else:
        p = float(2 * stats.t.sf(abs(mean / se), n - 1))
        half = float(stats.t.ppf(.975, n - 1) * se)
    return {'n_systems': n, 'log_mean': mean, 'log_sd': sd,
            'ratio': math.exp(mean), 'ci95_marginal': [math.exp(mean-half), math.exp(mean+half)], 'p_raw': p}


def analyze(rows, system_manifest, floor=1e-12):
    """rows: graph_size,system_id,history,arm,init,nmse,status='complete'.

    Manifest maps graph size to the twenty frozen unique system IDs. Normalize
    each error in the evaluator using its frozen training-only normalizer.
    Other ablations and sensitivity initializations are analyzed separately.
    """
    if floor != 1e-12:
        raise ValueError('unregistered numeric floor')
    if set(system_manifest) != {5, 30}:
        raise ValueError('both graph-size strata required')
    for ids in system_manifest.values():
        if len(ids) != 20 or len(set(ids)) != 20:
            raise ValueError('twenty unique frozen systems per stratum required')
    errors = {}
    for row in rows:
        size, system = row['graph_size'], row['system_id']
        if (size not in system_manifest or system not in system_manifest[size]
                or row['history'] not in HISTORIES or row['arm'] not in ARMS or row['init'] != 0):
            raise ValueError('unregistered primary cell')
        if row['status'] != 'complete' or not math.isfinite(row['nmse']) or row['nmse'] < 0:
            raise ValueError('failed or nonfinite fit: retain and resolve, do not omit')
        key = size, system, row['history'], row['arm']
        if key in errors:
            raise ValueError('duplicate primary cell')
        errors[key] = row['nmse']
    expected = {(size, system, history, arm) for size, ids in system_manifest.items()
                for system in ids for history in HISTORIES for arm in ARMS}
    if set(errors) != expected:
        raise ValueError('incomplete primary matrix; no complete-case filtering')
    contrasts = []
    for size, control in CONTRASTS:
        world_logs = {}
        activations = defaultdict(int)
        for system in system_manifest[size]:
            logs = []
            for history in sorted(HISTORIES):
                d = errors[size, system, history, 'delivery']
                c = errors[size, system, history, control]
                activations['delivery'] += int(d < floor)
                activations[control] += int(c < floor)
                logs.append(math.log(max(d, floor)) - math.log(max(c, floor)))
            world_logs[system] = sum(logs) / len(HISTORIES)
        contrasts.append({'graph_size': size, 'control': control,
                          **summarize_logs(list(world_logs.values())),
                          'system_log_ratios': world_logs, 'floor_activations': dict(activations)})
    adjusted = holm([r['p_raw'] for r in contrasts])
    for result, p in zip(contrasts, adjusted):
        result['p_holm_four_tests'] = p
        result['superiority'] = (result['ratio'] <= .8 and result['ci95_marginal'][1] < 1 and p < .05)
    return {'complete': True, 'n_primary_cells': len(errors), 'primary_init': 0, 'floor': floor,
            'independent_unit': 'system; equal-weight mean of two history log ratios',
            'multiple_testing': 'Holm across two controls in each of two strata', 'contrasts': contrasts}

#!/usr/bin/env python3
"""Development-only evidence-threshold screen on the fixed transfer benchmark.

Recreates the exact target data in learned_transfer.py and refuses to score a
candidate unless its warm baseline matches archived development receipts.
The evidence trigger uses all acquired data at each budget; it is an initial
diagnostic, not the later prequential confirmation described in the v3 plan.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from agenda_runner import features, posterior
from learned_transfer import family_centers, learn_source_library, log_model_evidence

SEEDS = tuple(range(100, 112))
ODDS = (4.0, 10.0, 25.0)
BUDGETS = (120, 200, 400)


def target(seed: int, changed: int, change_type: str):
    nodes = 30
    rng = np.random.default_rng(seed + 2719)
    centers = family_centers()
    forms = rng.integers(0, 4, nodes)
    old = centers[forms] + rng.normal(0, 0.05, (nodes, 6))
    truth = old.copy()
    changed_ids = rng.choice(nodes, changed, replace=False)
    if change_type == 'family':
        truth[changed_ids] = centers[(forms[changed_ids] + 1) % 4] + rng.normal(0, 0.05, (changed, 6))
    else:
        for i in changed_ids:
            truth[i, 2 + forms[i]] += rng.choice((-1, 1)) * 0.6
    x = rng.uniform(-2, 2, (nodes, 14, 2))
    phi = features(x.reshape(-1, 2)).reshape(nodes, 14, 6)
    y = np.einsum('nid,nd->ni', phi, truth, optimize=False) + rng.normal(0, 0.15, (nodes, 14))
    test_rng = np.random.default_rng(seed + 91473)
    test_phi = features(test_rng.uniform(-2, 2, (1024, 2)))
    return old, truth, changed_ids, x, y, test_phi


def node_mse(fit: np.ndarray, truth: np.ndarray, test_phi: np.ndarray) -> float:
    residual = np.einsum('j,mj->m', fit - truth, test_phi, optimize=False)
    return float(np.mean(residual**2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, default=Path('results/local_transfer_bayes_dev_20260926'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, source_samples = learn_source_library()
    assert source_samples == 2560
    source_hash = hashlib.sha256(source.tobytes()).hexdigest()
    rows = []
    for seed in SEEDS:
        for change_type in ('family', 'coefficient'):
            for changed in (1, 3, 10):
                old, truth, changed_ids, x, y, test_phi = target(seed, changed, change_type)
                changed_mask = np.zeros(30, dtype=bool)
                changed_mask[changed_ids] = True
                archive = args.archive / change_type / f'changed_{changed}' / f'seed_{seed}' / 'metrics.csv'
                assert archive.is_file(), archive
                for budget in BUDGETS:
                    extra = budget - 120
                    counts = np.array([4 + extra // 30 + (i < extra % 30) for i in range(30)])
                    node_errors = {odds: np.empty(30) for odds in ODDS}
                    switches = {odds: np.zeros(30, dtype=bool) for odds in ODDS}
                    warm_errors = np.empty(30)
                    bayes_errors = np.empty(30)
                    for i in range(30):
                        n = counts[i]
                        xi, yi = x[i, :n], y[i, :n]
                        warm_fit = posterior(xi, yi, old[i], 20.0)[0]
                        warm_errors[i] = node_mse(warm_fit, truth[i], test_phi)
                        proposals = np.vstack((old[i], source))
                        logs = np.array([log_model_evidence(xi, yi, mu) for mu in proposals])
                        # Bayes factor of the equal-weight source family mixture
                        # against the protected old expert. No hidden labels enter.
                        log_bf = logsumexp(logs[1:]) - np.log(4) - logs[0]
                        source_weights = np.exp(logs[1:] - logsumexp(logs[1:]))
                        source_fits = np.stack([posterior(xi, yi, mu, 20.0)[0] for mu in source])
                        switched_fit = source_weights @ source_fits
                        switched_error = node_mse(switched_fit, truth[i], test_phi)
                        bayes_logs = logs + np.log([0.5] + [0.125] * 4)
                        bayes_weights = np.exp(bayes_logs - logsumexp(bayes_logs))
                        bayes_fit = bayes_weights @ np.vstack((warm_fit, source_fits))
                        bayes_errors[i] = node_mse(bayes_fit, truth[i], test_phi)
                        for odds in ODDS:
                            switches[odds][i] = log_bf > np.log(odds)
                            node_errors[odds][i] = switched_error if switches[odds][i] else warm_errors[i]
                    for method, node_err in (('warm', warm_errors), ('source_bayes_mixture', bayes_errors)):
                        with archive.open() as stream:
                            archived = [r for r in csv.DictReader(stream)
                                        if r['method'] == method and int(r['target_budget']) == budget]
                        assert len(archived) == 1, archive
                        for field, value in (
                            ('mse', node_err.mean()),
                            ('changed_mse', node_err[changed_mask].mean()),
                            ('unchanged_mse', node_err[~changed_mask].mean()),
                        ):
                            assert np.isclose(value, float(archived[0][field]), rtol=0, atol=1e-12), (
                                seed, change_type, changed, budget, method, field, value, archived[0][field])
                    for odds in ODDS:
                        for i in range(30):
                            rows.append({'seed': seed, 'change_type': change_type, 'changed': changed,
                                         'target_budget': budget, 'odds': odds, 'node': i,
                                         'changed_node': int(changed_mask[i]),
                                         'switched': int(switches[odds][i]),
                                         'warm_node_mse': warm_errors[i],
                                         'candidate_node_mse': node_errors[odds][i]})
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = args.output / 'node_metrics.csv'
    with metrics.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for odds in ODDS:
        for change_type in ('family', 'coefficient'):
            for changed in (1, 3, 10):
                for budget in BUDGETS:
                    subset = [r for r in rows if r['odds'] == odds and r['change_type'] == change_type
                              and r['changed'] == changed and r['target_budget'] == budget]
                    assert len(subset) == len(SEEDS) * 30
                    groups = {}
                    for changed_node in (0, 1):
                        group = [r for r in subset if r['changed_node'] == changed_node]
                        groups[changed_node] = (
                            sum(r['candidate_node_mse'] for r in group) /
                            sum(r['warm_node_mse'] for r in group),
                            sum(r['switched'] for r in group) / len(group),
                        )
                    summary_rows.append({'odds': odds, 'change_type': change_type,
                                         'changed': changed, 'target_budget': budget,
                                         'changed_ratio': groups[1][0],
                                         'untouched_ratio': groups[0][0],
                                         'changed_switch_rate': groups[1][1],
                                         'false_switch_rate': groups[0][1]})
    summary = args.output / 'summary.csv'
    with summary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    passing = []
    for odds in ODDS:
        arm = [r for r in summary_rows if r['odds'] == odds]
        gate = all(r['untouched_ratio'] <= 1.05 for r in arm if r['target_budget'] in (120, 200))
        gate &= all(r['changed_ratio'] <= .8 for r in arm if r['change_type'] == 'family'
                    and r['target_budget'] == 200)
        gate &= all(r['changed_ratio'] <= 1.05 for r in arm if r['change_type'] == 'coefficient'
                    and r['target_budget'] == 200)
        if gate:
            passing.append(odds)
    spec = {'status': 'development-only odds-threshold diagnostic', 'seeds': list(SEEDS),
            'odds': list(ODDS), 'budgets': list(BUDGETS),
            'source_sha256': source_hash, 'source_samples': source_samples,
            'archived_parity': 'warm and v2 Bayesian mixture, 72 settings x 3 budgets x 3 metrics, atol=1e-12',
            'limitation': 'same acquired data select and fit; no prequential holdout',
            'passing_development_odds': passing}
    (args.output / 'protocol.json').write_text(json.dumps(spec, indent=2) + '\n')
    receipt = {'rows': len(rows), 'node_metrics_sha256': hashlib.sha256(metrics.read_bytes()).hexdigest(),
               'summary_rows': len(summary_rows), 'summary_sha256': hashlib.sha256(summary.read_bytes()).hexdigest(),
               'protocol_sha256': hashlib.sha256((args.output / 'protocol.json').read_bytes()).hexdigest()}
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'validated warm and v2 mixture parity; {len(rows)} node rows; '
          f'development-gate odds={passing} -> {args.output}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Counted prequential source-switch diagnostic on existing development systems."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from agenda_runner import posterior
from learned_transfer import learn_source_library, log_model_evidence
from transfer_safe_switch_dev import BUDGETS, ODDS, SEEDS, node_mse, target


def source_log_bf(logs: np.ndarray) -> float:
    return float(logsumexp(logs[1:]) - np.log(4) - logs[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, default=Path('results/local_transfer_bayes_dev_20260926'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, source_samples = learn_source_library()
    rows = []
    for seed in SEEDS:
        for change_type in ('family', 'coefficient'):
            for changed in (1, 3, 10):
                old, truth, changed_ids, x, y, test_phi = target(seed, changed, change_type)
                changed_mask = np.zeros(30, dtype=bool)
                changed_mask[changed_ids] = True
                archive = args.archive / change_type / f'changed_{changed}' / f'seed_{seed}' / 'metrics.csv'
                with archive.open() as stream:
                    archived = list(csv.DictReader(stream))
                for budget in BUDGETS:
                    extra = budget - 120
                    counts = [4 + extra // 30 + (i < extra % 30) for i in range(30)]
                    warm_errors = []
                    candidates = []
                    for i, n in enumerate(counts):
                        proposals = np.vstack((old[i], source))
                        early = np.array([log_model_evidence(x[i, :4], y[i, :4], mu)
                                          for mu in proposals])
                        full = np.array([log_model_evidence(x[i, :n], y[i, :n], mu)
                                         for mu in proposals])
                        # Nomination uses only the first four acquired samples.
                        nominated = source_log_bf(early) > 0
                        # Log predictive Bayes factor for the subsequently
                        # acquired examples, conditional on the first four.
                        confirm = source_log_bf(full) - source_log_bf(early)
                        warm_fit = posterior(x[i, :n], y[i, :n], old[i], 20.0)[0]
                        warm_error = node_mse(warm_fit, truth[i], test_phi)
                        warm_errors.append(warm_error)
                        source_weights = np.exp(full[1:] - logsumexp(full[1:]))
                        source_fits = np.stack([posterior(x[i, :n], y[i, :n], mu, 20.0)[0]
                                                for mu in source])
                        source_fit = source_weights @ source_fits
                        source_error = node_mse(source_fit, truth[i], test_phi)
                        candidates.append((nominated, confirm, warm_error, source_error))
                    ref = [r for r in archived if r['method'] == 'warm' and
                           int(r['target_budget']) == budget]
                    assert len(ref) == 1
                    warm_errors = np.asarray(warm_errors)
                    for field, value in (('mse', warm_errors.mean()),
                                         ('changed_mse', warm_errors[changed_mask].mean()),
                                         ('unchanged_mse', warm_errors[~changed_mask].mean())):
                        assert np.isclose(value, float(ref[0][field]), rtol=0, atol=1e-12)
                    for odds in ODDS:
                        for i, (nominated, confirm, warm_error, source_error) in enumerate(candidates):
                            switched = bool(nominated and counts[i] > 4 and confirm > np.log(odds))
                            rows.append({'seed': seed, 'change_type': change_type, 'changed': changed,
                                         'target_budget': budget, 'odds': odds, 'node': i,
                                         'changed_node': int(changed_mask[i]),
                                         'nominated_at_four': int(nominated),
                                         'confirmation_samples': counts[i] - 4,
                                         'confirmation_log_bf': confirm,
                                         'switched': int(switched),
                                         'warm_node_mse': warm_error,
                                         'candidate_node_mse': source_error if switched else warm_error})
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
    protocol = {'status': 'development-only counted prequential screen', 'seeds': list(SEEDS),
                'odds': list(ODDS), 'budgets': list(BUDGETS), 'source_samples': source_samples,
                'source_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
                'nomination_samples_per_node': 4, 'later_examples_are_counted': True,
                'passing_development_odds': passing}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    receipt = {'rows': len(rows), 'summary_rows': len(summary_rows)}
    for filename in ('node_metrics.csv', 'summary.csv', 'protocol.json'):
        receipt[filename + '_sha256'] = hashlib.sha256((args.output / filename).read_bytes()).hexdigest()
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'warm parity passed; {len(rows)} node rows; prequential gate odds={passing}')


if __name__ == '__main__':
    main()

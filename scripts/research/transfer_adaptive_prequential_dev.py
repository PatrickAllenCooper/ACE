#!/usr/bin/env python3
"""Counted adaptive evidence allocation for the protected transfer switch."""
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
from transfer_prequential_dev import source_log_bf
from transfer_safe_switch_dev import SEEDS, node_mse, target

NODES = 30
ASSAY = 4
BUDGET = 200
ODDS = 4.0
MAX_PER_NODE = 14


def allocate(scores: np.ndarray) -> np.ndarray:
    """Spend 120 assay samples, then prioritize nominees using no test labels."""
    counts = np.full(NODES, ASSAY, dtype=int)
    nominated = [int(i) for i in np.argsort(-scores) if scores[i] > 0]
    fallback = [int(i) for i in np.argsort(-scores) if scores[i] <= 0]
    for pool in (nominated, fallback):
        while counts.sum() < BUDGET and any(counts[i] < MAX_PER_NODE for i in pool):
            for i in pool:
                if counts.sum() == BUDGET:
                    break
                if counts[i] < MAX_PER_NODE:
                    counts[i] += 1
    assert counts.sum() == BUDGET and counts.max() <= MAX_PER_NODE
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--uniform-reference', type=Path, default=Path(
        'results/local_transfer_prequential_dev_20260926/node_metrics.csv'))
    args = parser.parse_args()
    source, source_samples = learn_source_library()
    assert source_samples == 2560
    records = []
    for seed in SEEDS:
        for change_type in ('family', 'coefficient'):
            for changed in (1, 3, 10):
                old, truth, changed_ids, x, y, test_phi = target(seed, changed, change_type)
                changed_mask = np.zeros(NODES, dtype=bool)
                changed_mask[changed_ids] = True
                proposals = [np.vstack((old[i], source)) for i in range(NODES)]
                early_logs = [np.array([log_model_evidence(x[i, :ASSAY], y[i, :ASSAY], mu)
                                        for mu in proposals[i]]) for i in range(NODES)]
                scores = np.array([source_log_bf(logs) for logs in early_logs])
                counts = allocate(scores)
                for i in range(NODES):
                    n = int(counts[i])
                    full = np.array([log_model_evidence(x[i, :n], y[i, :n], mu)
                                     for mu in proposals[i]])
                    confirmation = source_log_bf(full) - source_log_bf(early_logs[i])
                    nominated = bool(scores[i] > 0)
                    switched = bool(nominated and confirmation > np.log(ODDS))
                    warm_fit = posterior(x[i, :n], y[i, :n], old[i], 20.0)[0]
                    source_weights = np.exp(full[1:] - logsumexp(full[1:]))
                    source_fits = np.stack([posterior(x[i, :n], y[i, :n], mu, 20.0)[0]
                                            for mu in source])
                    candidate_fit = source_weights @ source_fits if switched else warm_fit
                    records.append({'seed': seed, 'change_type': change_type,
                                    'changed': changed, 'node': i,
                                    'changed_node': int(changed_mask[i]),
                                    'assay_log_bf': scores[i], 'nominated': int(nominated),
                                    'total_samples': n, 'confirmation_samples': n - ASSAY,
                                    'confirmation_log_bf': confirmation,
                                    'switched': int(switched),
                                    'adaptive_warm_mse': node_mse(warm_fit, truth[i], test_phi),
                                    'adaptive_switch_mse': node_mse(candidate_fit, truth[i], test_phi)})
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = args.output / 'node_metrics.csv'
    with metrics.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with args.uniform_reference.open() as stream:
        reference_rows = [row for row in csv.DictReader(stream)
                          if int(row['target_budget']) == BUDGET and float(row['odds']) == ODDS]
    uniform = {(int(row['seed']), row['change_type'], int(row['changed']), int(row['node'])):
               float(row['warm_node_mse']) for row in reference_rows}
    assert len(reference_rows) == len(uniform) == len(records)
    summary = []
    for change_type in ('family', 'coefficient'):
        for changed in (1, 3, 10):
            arm = [r for r in records if r['change_type'] == change_type and r['changed'] == changed]
            for changed_node in (0, 1):
                group = [r for r in arm if r['changed_node'] == changed_node]
                ratio = sum(r['adaptive_switch_mse'] for r in group) / sum(
                    r['adaptive_warm_mse'] for r in group)
                uniform_total = sum(uniform[(r['seed'], change_type, changed, r['node'])]
                                    for r in group)
                summary.append({'change_type': change_type, 'changed': changed,
                                'changed_node': changed_node, 'mse_ratio': ratio,
                                'adaptive_warm_vs_uniform': sum(r['adaptive_warm_mse']
                                                                 for r in group) / uniform_total,
                                'adaptive_switch_vs_uniform': sum(r['adaptive_switch_mse']
                                                                   for r in group) / uniform_total,
                                'nominated': sum(r['nominated'] for r in group),
                                'switched': sum(r['switched'] for r in group),
                                'mean_samples': sum(r['total_samples'] for r in group) / len(group),
                                'nodes': len(group)})
    summary_path = args.output / 'summary.csv'
    with summary_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    gate = all(r['mse_ratio'] <= .8 for r in summary if r['change_type'] == 'family'
               and r['changed_node'] == 1)
    gate &= all(r['mse_ratio'] <= 1.05 for r in summary if r['change_type'] == 'coefficient'
                and r['changed_node'] == 1)
    gate &= all(r['mse_ratio'] <= 1.05 for r in summary if r['changed_node'] == 0)
    protocol = {'status': 'development-only adaptive prequential diagnostic',
                'seeds': list(SEEDS), 'change_types': ['family', 'coefficient'],
                'changed_counts': [1, 3, 10], 'target_budget': BUDGET,
                'source_samples': source_samples, 'source_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
                'assay_per_node': ASSAY, 'max_per_node': MAX_PER_NODE,
                'acquisition': 'score all nodes using four acquired examples; round-robin allocate remaining samples to positive-BF nominees in descending BF order, then to other nodes in descending BF order',
                'switch_odds': ODDS,
                'gate': 'family changed ratio <=0.8; coefficient changed and all untouched ratios <=1.05 versus adaptive warm at the same 200 acquired samples',
                'passes_development_gate': bool(gate),
                'no_closed_model_calls': True}
    protocol_path = args.output / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    receipt = {'node_rows': len(records), 'summary_rows': len(summary),
               'settings': len(SEEDS) * 2 * 3,
               'uniform_reference_sha256': hashlib.sha256(args.uniform_reference.read_bytes()).hexdigest(),
               'exact_budget_each_setting': all(sum(r['total_samples'] for r in records
                                                    if r['seed'] == seed and r['change_type'] == typ
                                                    and r['changed'] == k) == BUDGET
                                                for seed in SEEDS for typ in ('family', 'coefficient')
                                                for k in (1, 3, 10))}
    assert receipt['exact_budget_each_setting']
    for file in (metrics, summary_path, protocol_path):
        receipt[file.name + '_sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('adaptive prequential gate', gate, 'rows', len(records))
    for row in summary:
        print(row['change_type'], row['changed'], 'changed' if row['changed_node'] else 'untouched',
              'ratio', round(row['mse_ratio'], 3), 'switched', row['switched'], '/', row['nodes'],
              'mean samples', round(row['mean_samples'], 2))
    print('adaptive warm and switch comparisons to archived uniform warm are in summary.csv')


if __name__ == '__main__':
    main()

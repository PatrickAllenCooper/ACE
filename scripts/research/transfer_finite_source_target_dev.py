#!/usr/bin/env python3
"""Frozen uniform-budget transfer screen using counted finite source posteriors."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from agenda_runner import features
from transfer_safe_switch_dev import target


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(x: np.ndarray, y: np.ndarray, mean: np.ndarray,
        covariance: np.ndarray, noise_sd: float) -> np.ndarray:
    """Policy fit sees only a prior and acquired target examples."""
    design = features(x)
    prior_precision = np.linalg.inv(covariance)
    precision = prior_precision + design.T @ design / noise_sd**2
    rhs = prior_precision @ mean + design.T @ y / noise_sd**2
    return np.linalg.solve(precision, rhs)


def score(prediction: np.ndarray, truth: np.ndarray, test_phi: np.ndarray) -> float:
    residual = np.einsum('j,mj->m', prediction - truth, test_phi, optimize=False)
    value = float(np.mean(residual**2))
    if not np.isfinite(value):
        raise ValueError('nonfinite target score')
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('pinned ACE_SOURCE_REVISION required')
    source_suite = json.loads((args.source_root / 'suite_complete.json').read_text())
    if source_suite['summary_sha256'] != protocol['source_preflight_suite_sha256']:
        raise ValueError('source suite hash mismatch')
    if sha(args.source_root / 'summary.csv') != source_suite['summary_sha256']:
        raise ValueError('source summary changed')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    all_rows = []
    for seed in protocol['development_seeds']:
        for source_n in protocol['source_samples_per_node']:
            source_cell = args.source_root / f'seed_{seed}' / f'source_n_{source_n}'
            source_receipt = json.loads((source_cell / 'complete.json').read_text())
            source_path = source_cell / 'source_posteriors.npz'
            if sha(source_path) != source_receipt['source_posteriors_sha256']:
                raise ValueError(f'source hash mismatch: {source_cell}')
            with np.load(source_path) as source:
                source_mean = source['mean'].copy()
                source_cov = source['covariance'].copy()
            assert source_mean.shape == (30, 6) and source_cov.shape == (30, 6, 6)
            for change_type in protocol['change_types']:
                for changed in protocol['changed_node_counts']:
                    # Evaluation truth stays in this scope; fit() receives only
                    # public target prefixes and independently fitted source priors.
                    _, truth, changed_ids, x, y, test_phi = target(seed, changed, change_type)
                    changed_mask = np.zeros(30, dtype=bool)
                    changed_mask[changed_ids] = True
                    cell = args.output / f'seed_{seed}' / f'source_n_{source_n}' / change_type / f'changed_{changed}'
                    cell.mkdir(parents=True)
                    rows = []
                    for budget in protocol['target_budgets']:
                        extra = budget - 120
                        counts = np.array([4 + extra // 30 + int(i < extra % 30)
                                           for i in range(30)])
                        if counts.sum() != budget or counts.max() > x.shape[1]:
                            raise ValueError('target budget/count mismatch')
                        for node in range(30):
                            n = int(counts[node])
                            public_x, public_y = x[node, :n], y[node, :n]
                            predictions = {
                                'frozen_source': source_mean[node],
                                'scratch': fit(public_x, public_y, np.zeros(6),
                                               np.eye(6) / .25, protocol['target_noise_sd']),
                                'warm_source': fit(public_x, public_y, source_mean[node],
                                                   source_cov[node], protocol['target_noise_sd']),
                            }
                            for method, prediction in predictions.items():
                                rows.append({'seed': seed, 'source_n': source_n,
                                             'change_type': change_type, 'changed': changed,
                                             'budget': budget, 'node': node,
                                             'changed_node': int(changed_mask[node]),
                                             'target_examples': n, 'method': method,
                                             'mse': score(prediction, truth[node], test_phi)})
                    path = cell / 'node_metrics.csv'
                    with path.open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    receipt = {'source_revision': revision, 'protocol_sha256': sha(args.protocol),
                               'source_posteriors_sha256': sha(source_path),
                               'node_metrics_sha256': sha(path), 'rows': len(rows),
                               'target_examples_at_max_budget': max(protocol['target_budgets']),
                               'source_train_examples': 30 * source_n,
                               'closed_model_calls': 0}
                    (cell / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
                    all_rows.extend(rows)
    summary = []
    for source_n in protocol['source_samples_per_node']:
        for change_type in protocol['change_types']:
            for changed in protocol['changed_node_counts']:
                for budget in protocol['target_budgets']:
                    for method in protocol['policies']:
                        for changed_node in (0, 1):
                            group = [r for r in all_rows if r['source_n'] == source_n
                                     and r['change_type'] == change_type and r['changed'] == changed
                                     and r['budget'] == budget and r['method'] == method
                                     and r['changed_node'] == changed_node]
                            expected = len(protocol['development_seeds']) * (
                                changed if changed_node else 30 - changed)
                            assert len(group) == expected
                            summary.append({'source_n': source_n, 'change_type': change_type,
                                            'changed': changed, 'budget': budget,
                                            'method': method, 'changed_node': changed_node,
                                            'nodes': len(group),
                                            'mean_mse': float(np.mean([r['mse'] for r in group]))})
    summary_path = args.output / 'summary.csv'
    with summary_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    receipt = {'source_revision': revision, 'protocol_sha256': sha(args.protocol),
               'source_suite_summary_sha256': source_suite['summary_sha256'],
               'cells': 12 * 2 * 2 * 3, 'node_rows': len(all_rows),
               'source_train_examples_reused': source_suite['source_train_examples_total'],
               'target_examples_per_cell_at_max_budget': 400,
               'target_examples_total_at_max_budget': 12 * 2 * 2 * 3 * 400,
               'summary_sha256': sha(summary_path), 'closed_model_calls': 0}
    (args.output / 'suite_complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('finite-source target development:', receipt['cells'], 'cells,',
          receipt['node_rows'], 'node rows')


if __name__ == '__main__':
    main()

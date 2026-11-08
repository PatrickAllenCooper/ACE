#!/usr/bin/env python3
"""Counted prequential source-versus-scratch transfer switch diagnostic."""
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
from transfer_finite_source_target_dev import fit, score, sha
from transfer_safe_switch_dev import target


def log_evidence(x: np.ndarray, y: np.ndarray, mean: np.ndarray,
                 covariance: np.ndarray, noise_sd: float) -> float:
    design = features(x)
    residual = y - np.einsum('ij,j->i', design, mean, optimize=False)
    predictive = np.einsum('ik,kl,jl->ij', design, covariance, design,
                           optimize=False) + noise_sd**2 * np.eye(len(y))
    sign, logdet = np.linalg.slogdet(predictive)
    if sign != 1:
        raise ValueError('invalid predictive covariance')
    return float(-.5 * (len(y) * np.log(2 * np.pi) + logdet +
                          residual @ np.linalg.solve(predictive, residual)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', required=True, type=Path)
    p.add_argument('--source-root', required=True, type=Path)
    p.add_argument('--parent-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    spec = json.loads(a.protocol.read_text())
    rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != rev:
        raise ValueError('pinned ACE_SOURCE_REVISION required')
    parent_receipt = json.loads((a.parent_root / 'suite_complete.json').read_text())
    if parent_receipt['summary_sha256'] != spec['parent_target_summary_sha256']:
        raise ValueError('parent summary identity mismatch')
    if sha(a.parent_root / 'summary.csv') != parent_receipt['summary_sha256']:
        raise ValueError('parent summary hash mismatch')
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    scratch_mean = np.zeros(6)
    scratch_cov = np.eye(6) / .25
    all_rows = []
    for seed in spec['development_seeds']:
        for source_n in spec['source_samples_per_node']:
            src = a.source_root / f'seed_{seed}' / f'source_n_{source_n}'
            src_receipt = json.loads((src / 'complete.json').read_text())
            src_path = src / 'source_posteriors.npz'
            if sha(src_path) != src_receipt['source_posteriors_sha256']:
                raise ValueError('source posterior hash mismatch')
            with np.load(src_path) as z:
                means, covs = z['mean'].copy(), z['covariance'].copy()
            for change_type in spec['change_types']:
                for changed in spec['changed_node_counts']:
                    _, truth, changed_ids, x, y, test_phi = target(seed, changed, change_type)
                    changed_mask = np.zeros(30, dtype=bool)
                    changed_mask[changed_ids] = True
                    relative = Path(f'seed_{seed}') / f'source_n_{source_n}' / change_type / f'changed_{changed}'
                    parent_cell = a.parent_root / relative
                    parent_cell_receipt = json.loads((parent_cell / 'complete.json').read_text())
                    parent_file = parent_cell / 'node_metrics.csv'
                    if sha(parent_file) != parent_cell_receipt['node_metrics_sha256']:
                        raise ValueError('parent cell hash mismatch')
                    with parent_file.open() as stream:
                        parent = {(int(r['budget']), int(r['node']), r['method']): float(r['mse'])
                                  for r in csv.DictReader(stream)}
                    if len(parent) != len(spec['target_budgets']) * 30 * 3:
                        raise ValueError('parent row count mismatch')
                    cell = a.output / relative
                    cell.mkdir(parents=True)
                    rows = []
                    for budget in spec['target_budgets']:
                        extra = budget - 120
                        counts = [4 + extra // 30 + int(i < extra % 30) for i in range(30)]
                        if sum(counts) != budget:
                            raise ValueError('budget mismatch')
                        for node, n in enumerate(counts):
                            xi, yi = x[node, :n], y[node, :n]
                            assay = spec['assay_per_node']
                            source_assay = log_evidence(xi[:assay], yi[:assay],
                                                        means[node], covs[node], .15)
                            scratch_assay = log_evidence(xi[:assay], yi[:assay],
                                                         scratch_mean, scratch_cov, .15)
                            nominated = scratch_assay > source_assay
                            source_full = log_evidence(xi, yi, means[node], covs[node], .15)
                            scratch_full = log_evidence(xi, yi, scratch_mean, scratch_cov, .15)
                            confirmation = (scratch_full - source_full) - (scratch_assay - source_assay)
                            switched = bool(nominated and n > assay and confirmation > np.log(4))
                            warm = fit(xi, yi, means[node], covs[node], .15)
                            scratch = fit(xi, yi, scratch_mean, scratch_cov, .15)
                            warm_mse = score(warm, truth[node], test_phi)
                            scratch_mse = score(scratch, truth[node], test_phi)
                            for method, value in (('warm_source', warm_mse), ('scratch', scratch_mse)):
                                if not np.isclose(value, parent[budget, node, method], rtol=0, atol=1e-11):
                                    raise ValueError(f'parent parity failed: {relative}, {budget}, {node}, {method}')
                            rows.append({'seed': seed, 'source_n': source_n,
                                         'change_type': change_type, 'changed': changed,
                                         'budget': budget, 'node': node,
                                         'changed_node': int(changed_mask[node]),
                                         'target_examples': n, 'nominated': int(nominated),
                                         'confirmation_log_bf': confirmation,
                                         'switched': int(switched),
                                         'warm_mse': warm_mse, 'scratch_mse': scratch_mse,
                                         'switch_mse': scratch_mse if switched else warm_mse})
                    path = cell / 'node_metrics.csv'
                    with path.open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    (cell / 'complete.json').write_text(json.dumps({
                        'source_revision': rev, 'protocol_sha256': sha(a.protocol),
                        'source_posteriors_sha256': sha(src_path),
                        'parent_node_metrics_sha256': sha(parent_file),
                        'node_metrics_sha256': sha(path), 'rows': len(rows),
                        'target_examples_at_max_budget': max(spec['target_budgets']),
                        'closed_model_calls': 0}, indent=2) + '\n')
                    all_rows.extend(rows)
    summary = []
    for source_n in spec['source_samples_per_node']:
        for change_type in spec['change_types']:
            for changed in spec['changed_node_counts']:
                for budget in spec['target_budgets']:
                    for changed_node in (0, 1):
                        group = [r for r in all_rows if r['source_n'] == source_n and
                                 r['change_type'] == change_type and r['changed'] == changed and
                                 r['budget'] == budget and r['changed_node'] == changed_node]
                        assert len(group) == 12 * (changed if changed_node else 30 - changed)
                        summary.append({'source_n': source_n, 'change_type': change_type,
                                        'changed': changed, 'budget': budget,
                                        'changed_node': changed_node, 'nodes': len(group),
                                        'nominated': sum(r['nominated'] for r in group),
                                        'switched': sum(r['switched'] for r in group),
                                        'mean_warm_mse': float(np.mean([r['warm_mse'] for r in group])),
                                        'mean_scratch_mse': float(np.mean([r['scratch_mse'] for r in group])),
                                        'mean_switch_mse': float(np.mean([r['switch_mse'] for r in group]))})
    summary_path = a.output / 'summary.csv'
    with summary_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    receipt = {'source_revision': rev, 'protocol_sha256': sha(a.protocol),
               'parent_summary_sha256': parent_receipt['summary_sha256'],
               'cells': 144, 'node_rows': len(all_rows),
               'target_examples_total_at_max_budget': 144 * 400,
               'summary_sha256': sha(summary_path), 'closed_model_calls': 0}
    (a.output / 'suite_complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('finite-source switch:', receipt['cells'], 'cells,', receipt['node_rows'], 'rows')


if __name__ == '__main__':
    main()

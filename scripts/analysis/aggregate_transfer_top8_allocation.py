#!/usr/bin/env python3
"""Validate top-eight allocation receipts and paired 20-system contrasts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = list(a.root.glob('seed_*/source_n_*/source/complete.json'))
    target = [p for p in a.root.glob('seed_*/source_n_*/*/complete.json')
              if p.parent.name != 'source']
    if len(source) != 40 or len(target) != 80:
        raise ValueError('incomplete source/target cells')
    for path in source:
        r = json.loads(path.read_text())
        if sha(path.parent / 'source_data.npz') != r['source_data_sha256'] or sha(
                path.parent / 'source_posteriors.npz') != r['source_posteriors_sha256']:
            raise ValueError('source hash mismatch')
    groups = defaultdict(list)
    rank_rows = []
    for path in target:
        r = json.loads(path.read_text())
        actions_path = path.parent / 'actions.json'
        metrics_path = path.parent / 'node_metrics.csv'
        if sha(actions_path) != r['actions_sha256'] or sha(metrics_path) != r['node_metrics_sha256']:
            raise ValueError('target hash mismatch')
        if sha(path.parent.parent / 'source' / 'source_posteriors.npz') != r['source_posteriors_sha256']:
            raise ValueError('source identity mismatch')
        action = json.loads(actions_path.read_text())
        if sum(action['adaptive_counts']) != 200 or sum(action['uniform_counts']) != 200:
            raise ValueError('target budget mismatch')
        if action['unique_acquired_prefix_responses'] != sum(map(
                max, zip(action['adaptive_counts'], action['uniform_counts']))):
            raise ValueError('unique prefix count mismatch')
        rows = list(csv.DictReader(metrics_path.open()))
        if len(rows) != 180:
            raise ValueError('target row count mismatch')
        changed = {int(x['node']) for x in rows if int(x['changed_node'])}
        if len(changed) != 1:
            raise ValueError('changed label count mismatch')
        rank_rows.append({'seed': action['seed'], 'source_n': action['source_n'],
                          'change_type': action['change_type'],
                          'top_eight_hit': int(next(iter(changed)) in action['top_eight'])})
        for row in rows:
            key = (int(row['seed']), int(row['source_n']), row['change_type'],
                   int(row['changed_node']), row['allocation'], row['method'])
            groups[key].append(float(row['mse']))
    per_seed = []
    for key, values in sorted(groups.items()):
        seed, source_n, change_type, changed_node, allocation, method = key
        if len(values) != (1 if changed_node else 29):
            raise ValueError('per-seed group count mismatch')
        per_seed.append({'seed': seed, 'source_n': source_n,
                         'change_type': change_type, 'changed_node': changed_node,
                         'allocation': allocation, 'method': method,
                         'nodes': len(values), 'mean_mse': float(np.mean(values))})
    contrasts = []
    rng = np.random.default_rng(20260929)
    comparisons = [
        ('changed_vs_uniform_soft', 1, 'uniform', 'soft_mixture', .8),
        ('changed_vs_adaptive_scratch', 1, 'adaptive_top8', 'scratch', 1.05),
        ('untouched_vs_uniform_soft', 0, 'uniform', 'soft_mixture', 1.05),
        ('untouched_vs_adaptive_warm', 0, 'adaptive_top8', 'source_warm', 1.05),
    ]
    for source_n in (16, 64):
        for change_type in ('family', 'coefficient'):
            for label, changed_node, control_allocation, control_method, limit in comparisons:
                def vector(allocation: str, method: str) -> np.ndarray:
                    subset = [r for r in per_seed if r['source_n'] == source_n and
                              r['change_type'] == change_type and r['changed_node'] == changed_node and
                              r['allocation'] == allocation and r['method'] == method]
                    subset.sort(key=lambda r: r['seed'])
                    if len(subset) != 20:
                        raise ValueError('missing paired system')
                    return np.array([r['mean_mse'] for r in subset])
                candidate = vector('adaptive_top8', 'soft_mixture')
                control = vector(control_allocation, control_method)
                indices = rng.integers(0, 20, size=(10000, 20))
                draws = candidate[indices].mean(axis=1) / control[indices].mean(axis=1)
                lo, hi = np.quantile(draws, [.025, .975])
                ratio = float(candidate.mean() / control.mean())
                contrasts.append({'source_n': source_n, 'change_type': change_type,
                                  'contrast': label, 'ratio': ratio, 'limit': limit,
                                  'passes_point_gate': int(ratio <= limit),
                                  'paired_bootstrap_95pct_low': float(lo),
                                  'paired_bootstrap_95pct_high': float(hi)})
    a.output.mkdir(parents=True)
    per_seed_path = a.output / 'per_seed.csv'
    rank_path = a.output / 'rank_recall.csv'
    contrasts_path = a.output / 'contrasts.csv'
    write_csv(per_seed_path, per_seed)
    write_csv(rank_path, sorted(rank_rows, key=lambda r: (r['source_n'], r['change_type'], r['seed'])))
    write_csv(contrasts_path, contrasts)
    (a.output / 'complete.json').write_text(json.dumps({
        'parent_suite_sha256': sha(a.root / 'suite_complete.json'),
        'source_cells': len(source), 'target_cells': len(target),
        'per_seed_rows': len(per_seed), 'rank_rows': len(rank_rows),
        'contrast_rows': len(contrasts),
        'per_seed_sha256': sha(per_seed_path), 'rank_sha256': sha(rank_path),
        'contrasts_sha256': sha(contrasts_path),
        'bootstrap_seed': 20260929, 'bootstrap_resamples': 10000,
        'new_queries': 0, 'closed_model_calls': 0}, indent=2) + '\n')
    print('point gate', sum(x['passes_point_gate'] for x in contrasts), '/', len(contrasts))


if __name__ == '__main__':
    main()

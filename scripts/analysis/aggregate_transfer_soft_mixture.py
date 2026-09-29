#!/usr/bin/env python3
"""Validate fresh transfer custody and report paired seed uncertainty."""
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
    source_files = sorted(a.root.glob('seed_*/source_n_*/source/complete.json'))
    target_files = sorted(a.root.glob('seed_*/source_n_*/*/changed_*/complete.json'))
    if len(source_files) != 40 or len(target_files) != 240:
        raise ValueError('incomplete source or target cells')
    for path in source_files:
        r = json.loads(path.read_text())
        if sha(path.parent / 'source_data.npz') != r['source_data_sha256'] or sha(
                path.parent / 'source_posteriors.npz') != r['source_posteriors_sha256']:
            raise ValueError(f'source hash mismatch: {path}')
    groups = defaultdict(list)
    for path in target_files:
        r = json.loads(path.read_text())
        data = path.parent / 'node_metrics.csv'
        if sha(data) != r['node_metrics_sha256']:
            raise ValueError(f'target hash mismatch: {path}')
        if sha(path.parent.parent.parent / 'source' / 'source_posteriors.npz') != r['source_posteriors_sha256']:
            raise ValueError(f'target source identity mismatch: {path}')
        rows = list(csv.DictReader(data.open()))
        if len(rows) != 120:
            raise ValueError('node-method count mismatch')
        for row in rows:
            key = (int(row['seed']), int(row['source_n']), row['change_type'],
                   int(row['changed']), int(row['changed_node']), row['method'])
            groups[key].append(float(row['mse']))
    per_seed = []
    for key, values in sorted(groups.items()):
        seed, source_n, change_type, changed, changed_node, method = key
        expected = changed if changed_node else 30 - changed
        if len(values) != expected:
            raise ValueError('seed-group node count mismatch')
        per_seed.append({'seed': seed, 'source_n': source_n,
                         'change_type': change_type, 'changed': changed,
                         'changed_node': changed_node, 'method': method,
                         'nodes': len(values), 'mean_mse': float(np.mean(values))})
    summary = []
    rng = np.random.default_rng(20260928)
    for source_n in (16, 64):
        for change_type in ('family', 'coefficient'):
            for changed in (1, 3, 10):
                for changed_node in (0, 1):
                    subset = [r for r in per_seed if r['source_n'] == source_n and
                              r['change_type'] == change_type and r['changed'] == changed and
                              r['changed_node'] == changed_node]
                    vectors = {m: np.array([r['mean_mse'] for r in subset if r['method'] == m])
                               for m in ('soft_mixture', 'source_warm', 'scratch')}
                    if any(len(v) != 20 for v in vectors.values()):
                        raise ValueError('missing seed-method pairing')
                    if changed_node == 0:
                        baseline = 'source_warm'
                    elif change_type == 'family':
                        baseline = 'scratch'
                    else:
                        baseline = min(('scratch', 'source_warm'),
                                       key=lambda m: vectors[m].mean())
                    indices = rng.integers(0, 20, size=(10000, 20))
                    ratios = (vectors['soft_mixture'][indices].mean(axis=1) /
                              vectors[baseline][indices].mean(axis=1))
                    point = vectors['soft_mixture'].mean() / vectors[baseline].mean()
                    lo, hi = np.quantile(ratios, [.025, .975])
                    summary.append({'source_n': source_n, 'change_type': change_type,
                                    'changed': changed, 'changed_node': changed_node,
                                    'baseline': baseline, 'ratio': float(point),
                                    'paired_bootstrap_95pct_low': float(lo),
                                    'paired_bootstrap_95pct_high': float(hi),
                                    'passes_point_gate': int(point <= 1.05)})
    a.output.mkdir(parents=True)
    per_seed_path, summary_path = a.output / 'per_seed.csv', a.output / 'strata.csv'
    write_csv(per_seed_path, per_seed)
    write_csv(summary_path, summary)
    (a.output / 'complete.json').write_text(json.dumps({
        'source_suite_sha256': sha(a.root / 'suite_complete.json'),
        'source_cells': len(source_files), 'target_cells': len(target_files),
        'per_seed_rows': len(per_seed), 'strata_rows': len(summary),
        'per_seed_sha256': sha(per_seed_path), 'strata_sha256': sha(summary_path),
        'bootstrap_seed': 20260928, 'bootstrap_resamples': 10000,
        'new_queries': 0, 'closed_model_calls': 0}, indent=2) + '\n')
    print('point gate:', sum(r['passes_point_gate'] for r in summary), '/', len(summary))


if __name__ == '__main__':
    main()

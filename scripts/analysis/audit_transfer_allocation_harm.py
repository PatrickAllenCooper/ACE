#!/usr/bin/env python3
"""Decompose held-out transfer errors by acquired target response count."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    suite = json.loads((a.root / 'suite_complete.json').read_text())
    records = []
    for seed in range(1200, 1220):
        for source_n in (16, 64):
            cell = a.root / f'seed_{seed}' / f'source_n_{source_n}' / 'heldout_tanh'
            receipt = json.loads((cell / 'complete.json').read_text())
            assert receipt['source_revision'] == suite['source_revision']
            for name in ('actions', 'system', 'node_metrics'):
                suffix = 'json' if name != 'node_metrics' else 'csv'
                assert sha(cell / f'{name}.{suffix}') == receipt[name + '_sha256']
            action = json.loads((cell / 'actions.json').read_text())
            system = json.loads((cell / 'system.json').read_text())
            assert system['seed'] == seed and system['changed_node'] in action['top_eight']
            changed_rank = action['top_eight'].index(system['changed_node']) + 1
            rows = list(csv.DictReader((cell / 'node_metrics.csv').open()))
            lookup = {(r['allocation'], r['method'], int(r['node'])): r for r in rows}
            assert len(lookup) == 180
            for node in range(30):
                is_changed = node == system['changed_node']
                adaptive_n = action['adaptive_counts'][node]
                uniform_n = action['uniform_counts'][node]
                assert adaptive_n in (4, 14) and uniform_n in (6, 7)
                adaptive_soft = float(lookup[('adaptive_top8', 'soft_mixture', node)]['mse'])
                uniform_soft = float(lookup[('uniform', 'soft_mixture', node)]['mse'])
                adaptive_warm = float(lookup[('adaptive_top8', 'source_warm', node)]['mse'])
                records.append({'seed': seed, 'source_n': source_n, 'node': node,
                                'changed_node': int(is_changed),
                                'changed_rank': changed_rank,
                                'adaptive_n': adaptive_n, 'uniform_n': uniform_n,
                                'group': ('changed' if is_changed else
                                          'unchanged_low4' if adaptive_n == 4 else
                                          'unchanged_high14'),
                                'adaptive_soft_mse': adaptive_soft,
                                'uniform_soft_mse': uniform_soft,
                                'adaptive_warm_mse': adaptive_warm,
                                'soft_minus_warm': adaptive_soft - adaptive_warm,
                                'adaptive_minus_uniform': adaptive_soft - uniform_soft})
    summary = []
    for source_n in (16, 64):
        for group in ('unchanged_low4', 'unchanged_high14', 'unchanged_all', 'changed'):
            subset = [r for r in records if r['source_n'] == source_n and
                      (r['group'] == group or group == 'unchanged_all' and
                       not r['changed_node'])]
            if group == 'unchanged_low4':
                assert len(subset) == 20 * 22
            if group == 'unchanged_high14':
                assert len(subset) == 20 * 7
            if group == 'unchanged_all':
                assert len(subset) == 20 * 29
            if group == 'changed':
                assert len(subset) == 20
            adaptive = np.array([r['adaptive_soft_mse'] for r in subset])
            uniform = np.array([r['uniform_soft_mse'] for r in subset])
            warm = np.array([r['adaptive_warm_mse'] for r in subset])
            summary.append({'source_n': source_n, 'group': group,
                            'node_systems': len(subset),
                            'adaptive_mean': float(adaptive.mean()),
                            'uniform_mean': float(uniform.mean()),
                            'adaptive_over_uniform': float(adaptive.mean()/uniform.mean()),
                            'adaptive_minus_uniform_sum': float((adaptive-uniform).sum()),
                            'adaptive_soft_minus_warm_mean': float((adaptive-warm).mean())})
    a.output.mkdir(parents=True)
    write_csv(a.output / 'per_node.csv', records)
    write_csv(a.output / 'summary.csv', summary)
    (a.output / 'complete.json').write_text(json.dumps({
        'input_source_revision': suite['source_revision'],
        'input_suite_sha256': sha(a.root / 'suite_complete.json'),
        'input_cells': 40, 'node_rows': len(records),
        'per_node_sha256': sha(a.output / 'per_node.csv'),
        'summary_sha256': sha(a.output / 'summary.csv'),
        'new_queries': 0, 'closed_model_calls': 0}, indent=2) + '\n')
    for row in summary:
        print(row['source_n'], row['group'], row['node_systems'],
              f"ratio={row['adaptive_over_uniform']:.4f}",
              f"delta_sum={row['adaptive_minus_uniform_sum']:.4f}",
              f"soft-warm={row['adaptive_soft_minus_warm_mean']:.6f}")


if __name__ == '__main__':
    main()

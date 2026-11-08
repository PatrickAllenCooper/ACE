#!/usr/bin/env python3
"""Audit fixed-order coverage in the corrected connected-motif development grid."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from connected_acquisition import experiment


def write(path: Path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, default=Path('results/research_connected_acquisition_dev_rng_v1'))
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    metrics, actions = [], []
    for motifs in (3, 10):
        for seed in (200, 201, 202):
            archived = a.archive / 'nodes_30' / f'motifs_{motifs}' / f'seed_{seed}'
            with (archived / 'metrics.csv').open() as stream:
                original_metrics = list(csv.DictReader(stream))
            with (archived / 'actions.csv').open() as stream:
                original_actions = list(csv.DictReader(stream))
            for offset in range(motifs):
                cell_metrics, cell_actions, _ = experiment(seed, 30, motifs, .15, 4,
                                                           coverage_offset=offset)
                if offset == 0:
                    assert len(cell_metrics) == len(original_metrics)
                    assert len(cell_actions) == len(original_actions)
                    for new, old in zip(cell_metrics, original_metrics):
                        for key, value in new.items():
                            if isinstance(value, float):
                                assert math.isclose(value, float(old[key]), abs_tol=1e-10, rel_tol=0), (seed, motifs, key)
                            else:
                                assert str(value) == old[key], (seed, motifs, key)
                    for new, old in zip(cell_actions, original_actions):
                        assert all(str(value) == old[key] for key, value in new.items()), (seed, motifs)
                metrics.extend({'coverage_offset': offset, **row} for row in cell_metrics)
                actions.extend({'seed': seed, 'nodes': 30, 'motifs': motifs,
                                'coverage_offset': offset, **row} for row in cell_actions)
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output / 'metrics.csv', metrics)
    write(a.output / 'actions.csv', actions)
    summary = []
    for motifs in (3, 10):
        for offset in range(motifs):
            by_seed = {}
            for seed in (200, 201, 202):
                cell = {r['method']: r for r in metrics if r['motifs'] == motifs and
                        r['seed'] == seed and r['coverage_offset'] == offset}
                by_seed[seed] = {m: float(cell[m]['feasible_motif_mse'])
                                 for m in ('coverage_single', 'coverage_pair', 'risk_pair')}
            summary.append({'motifs': motifs, 'coverage_offset': offset,
                            'coverage_pair_mean': sum(v['coverage_pair'] for v in by_seed.values()) / 3,
                            'risk_pair_mean': sum(v['risk_pair'] for v in by_seed.values()) / 3,
                            'coverage_single_mean': sum(v['coverage_single'] for v in by_seed.values()) / 3,
                            'risk_better_than_coverage_pair': sum(v['risk_pair'] < v['coverage_pair']
                                                                  for v in by_seed.values())})
    write(a.output / 'summary.csv', summary)
    protocol = {'status': 'post hoc development diagnostic', 'seeds': [200, 201, 202],
                'nodes': 30, 'motifs': [3, 10], 'root_sd': .15, 'penalty': 4,
                'budget': 400, 'rotations': 'all cyclic starts per motif count',
                'offset_zero_archived_parity': 'actions exact, metrics absolute tolerance 1e-10'}
    (a.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    receipt = {'metric_rows': len(metrics), 'action_rows': len(actions), 'summary_rows': len(summary)}
    for name in ('metrics.csv', 'actions.csv', 'summary.csv', 'protocol.json'):
        receipt[name + '_sha256'] = hashlib.sha256((a.output / name).read_bytes()).hexdigest()
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'validated offset-zero parity; {len(metrics)} metric rows, {len(actions)} actions, '
          f'{len(summary)} rotation summaries')


if __name__ == '__main__':
    main()

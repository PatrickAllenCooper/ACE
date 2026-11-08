#!/usr/bin/env python3
"""Post hoc order-independent coverage audit on archived connected motifs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from connected_acquisition import experiment


def write(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def permutations(seed: int, motifs: int, count: int) -> list[tuple[int, ...]]:
    if motifs == 3:
        import itertools
        return list(itertools.permutations(range(motifs)))
    rng = np.random.default_rng(910000 + seed)
    orders = {tuple(range(motifs))}
    while len(orders) < count:
        orders.add(tuple(int(x) for x in rng.permutation(motifs)))
    return [tuple(range(motifs))] + sorted(orders - {tuple(range(motifs))})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=Path(
        'results/research_connected_acquisition_dev_rng_v1'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--k10-orders', type=int, default=64)
    args = parser.parse_args()
    if not 2 <= args.k10_orders <= 256:
        raise ValueError('Select 2-256 distinct k10 orders')
    metrics, actions, summaries = [], [], []
    for motifs in (3, 10):
        for seed in (200, 201, 202):
            archive = args.archive / 'nodes_30' / f'motifs_{motifs}' / f'seed_{seed}'
            with (archive / 'metrics.csv').open() as stream:
                old_metrics = {row['method']: row for row in csv.DictReader(stream)}
            with (archive / 'actions.csv').open() as stream:
                old_actions = list(csv.DictReader(stream))
            risk_signature = None
            for order_index, order in enumerate(permutations(seed, motifs, args.k10_orders)):
                rows, act, _ = experiment(seed, 30, motifs, .15, 4,
                                          coverage_order=order)
                by_method = {row['method']: row for row in rows}
                risk_actions = [row for row in act if row['method'] == 'risk_pair']
                if risk_signature is None:
                    risk_signature = risk_actions
                    archived_risk = [row for row in old_actions
                                     if row['method'] == 'risk_pair']
                    assert len(risk_actions) == len(archived_risk)
                    assert all(all(str(value) == old[key] for key, value in new.items())
                               for new, old in zip(risk_actions, archived_risk))
                else:
                    assert risk_actions == risk_signature
                if order_index == 0:
                    for method, row in by_method.items():
                        assert abs(row['feasible_motif_mse'] -
                                   float(old_metrics[method]['feasible_motif_mse'])) < 1e-10
                    assert len(act) == len(old_actions)
                    assert all(all(str(value) == old[key] for key, value in new.items())
                               for new, old in zip(act, old_actions))
                label = ','.join(map(str, order))
                metrics.extend({'order_index': order_index, 'coverage_order': label,
                                **row} for row in rows)
                actions.extend({'seed': seed, 'nodes': 30, 'motifs': motifs,
                                'order_index': order_index, 'coverage_order': label,
                                **row} for row in act)
                summaries.append({'seed': seed, 'motifs': motifs,
                                  'order_index': order_index, 'coverage_order': label,
                                  'risk_pair_mse': by_method['risk_pair']['feasible_motif_mse'],
                                  'coverage_pair_mse': by_method['coverage_pair']['feasible_motif_mse'],
                                  'coverage_single_mse': by_method['coverage_single']['feasible_motif_mse']})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('metrics.csv', metrics), ('actions.csv', actions),
                       ('summary.csv', summaries)):
        write(args.output / name, rows)
    protocol = {'status': 'post hoc development diagnostic', 'seeds': [200, 201, 202],
                'motifs': [3, 10], 'k3_orders': 6,
                'k10_orders_per_system': args.k10_orders,
                'permutation_rng': 'numpy.default_rng(910000 + system seed)',
                'root_sd': .15, 'penalty': 4, 'budget': 400,
                'identity_order': 'exact archived outcome and action parity',
                'risk_actions': 'unchanged across coverage permutations'}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    receipt = {'metric_rows': len(metrics), 'action_rows': len(actions),
               'summary_rows': len(summaries)}
    for name in ('metrics.csv', 'actions.csv', 'summary.csv', 'protocol.json'):
        receipt[name + '_sha256'] = hashlib.sha256((args.output / name).read_bytes()).hexdigest()
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for motifs in (3, 10):
        group = [row for row in summaries if row['motifs'] == motifs]
        print('k', motifs, 'risk_below_pair', sum(row['risk_pair_mse'] < row['coverage_pair_mse']
                                                  for row in group), '/', len(group),
              'risk_below_single', sum(row['risk_pair_mse'] < row['coverage_single_mse']
                                      for row in group), '/', len(group))
    print('validated identity action parity, fixed risk actions, and', len(summaries),
          'system/order scenarios')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run the frozen balanced-pair development control and verify archived parity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from connected_acquisition import experiment, write_csv


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=Path(
        'results/research_connected_acquisition_dev_rng_v1'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    protocol = Path('docs/development/guidance/protocol_connected_balanced_dev_2026-09-28.json')
    settings = json.loads(protocol.read_text())
    all_metrics, all_actions = [], []
    for motifs in settings['motifs']:
        for seed in settings['seeds']:
            archive = args.archive / 'nodes_30' / f'motifs_{motifs}' / f'seed_{seed}'
            with (archive / 'metrics.csv').open() as stream:
                old_metrics = list(csv.DictReader(stream))
            with (archive / 'actions.csv').open() as stream:
                old_actions = list(csv.DictReader(stream))
            rows, actions, _ = experiment(seed, 30, motifs, .15, 4,
                                          include_balanced=True)
            assert len(rows) == len(old_metrics) + 1
            assert len(actions) > len(old_actions)
            for new, old in zip(rows, old_metrics):
                for key, value in new.items():
                    if isinstance(value, float):
                        assert abs(value - float(old[key])) < 1e-10, (seed, motifs, key)
                    else:
                        assert str(value) == old[key], (seed, motifs, key)
            assert all(all(str(value) == old[key] for key, value in new.items())
                       for new, old in zip(actions, old_actions)), (seed, motifs)
            balanced = rows[-1]
            balanced_actions = actions[len(old_actions):]
            assert balanced['method'] == 'balanced_risk_pair'
            assert len(balanced_actions) == balanced['steps']
            assert all(row['method'] == 'balanced_risk_pair' for row in balanced_actions)
            assert balanced['cost_spent'] == balanced_actions[-1]['cumulative_cost']
            assert balanced['cost_spent'] == next(r['cost_spent'] for r in rows
                                                 if r['method'] == 'risk_pair')
            visits = [0] * motifs
            for action in balanced_actions:
                assert action['motif'] in [j for j, count in enumerate(visits)
                                           if count == min(visits)]
                visits[action['motif']] += 1
            assert max(visits) - min(visits) <= 1
            all_metrics.extend(rows)
            all_actions.extend({'seed': seed, 'motifs': motifs, **row} for row in actions)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / 'metrics.csv'
    actions_path = args.output / 'actions.csv'
    write_csv(metrics_path, all_metrics)
    write_csv(actions_path, all_actions)
    receipt = {'schema_version': 1, 'kind': 'connected_balanced_dev',
               'protocol_sha256': sha(protocol), 'metric_rows': len(all_metrics),
               'action_rows': len(all_actions), 'archived_parity': 'exact actions; metrics tol 1e-10',
               'balanced_visits': 'difference <= 1',
               'metrics_sha256': sha(metrics_path), 'actions_sha256': sha(actions_path)}
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for motifs in settings['motifs']:
        cells = [r for r in all_metrics if r['motifs'] == motifs]
        for method in ('balanced_risk_pair', 'risk_pair', 'coverage_pair', 'coverage_single'):
            vals = [r['feasible_motif_mse'] for r in cells if r['method'] == method]
            print(f'k={motifs} {method}: mean={sum(vals)/len(vals):.6f}, values={vals}')
    print(f'validated {len(all_metrics)} metrics, {len(all_actions)} actions')


if __name__ == '__main__':
    main()

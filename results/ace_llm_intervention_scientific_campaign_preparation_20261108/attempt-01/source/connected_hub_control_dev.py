#!/usr/bin/env python3
"""Run frozen hub controls, verifying all original fanout arms against custody."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from connected_acquisition import experiment, write_csv


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_cell(seed: int, motifs: int, first: Path, second: Path) -> Path:
    root = first if seed <= 302 else second
    return root / f'motifs_{motifs}' / f'seed_{seed}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--first', type=Path, default=Path('results/local_connected_fanout_dev_20260928'))
    parser.add_argument('--second', type=Path, default=Path('results/research_connected_fanout_extension_dev_v1'))
    parser.add_argument('--protocol', type=Path, default=Path(
        'docs/development/guidance/protocol_connected_hub_control_dev_2026-09-28.json'))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    metrics, actions = [], []
    for motifs in protocol['motifs']:
        for seed in protocol['seeds']:
            old = prior_cell(seed, motifs, args.first, args.second)
            with (old / 'metrics.csv').open() as stream:
                archived_metrics = list(csv.DictReader(stream))
            with (old / 'actions.csv').open() as stream:
                archived_actions = list(csv.DictReader(stream))
            rows, act, _ = experiment(seed, 30, motifs, .15, 4,
                                      include_balanced=True, topology='fanout',
                                      include_hub=True)
            assert len(rows) == len(archived_metrics) + 2
            for new, old_row in zip(rows, archived_metrics):
                for key, value in new.items():
                    if isinstance(value, float):
                        assert math.isclose(value, float(old_row[key]), rel_tol=0,
                                            abs_tol=1e-10), (seed, motifs, key)
                    else:
                        assert str(value) == old_row[key], (seed, motifs, key)
            assert all(all(str(value) == old_row[key] for key, value in new.items())
                       for new, old_row in zip(act, archived_actions)), (seed, motifs)
            for method in protocol['new_arms']:
                m = next(row for row in rows if row['method'] == method)
                aa = [a for a in act if a['method'] == method]
                assert len(aa) == 5 and m['cost_spent'] == 360 and m['samples'] == 40
                assert [a['motif'] for a in aa[:3]] == [0, 0, 0]
                assert len(set(a['motif'] for a in aa[3:])) == 2
                assert all(a['motif'] > 0 for a in aa[3:])
                assert [a['levels'] for a in aa] == [
                    '-2.0,-2.0', '-2.0,2.0', '2.0,2.0', '2.0,2.0', '-2.0,-2.0']
                assert sum(a['masked_child_labels'] for a in aa) == m['masked_child_labels']
            metrics.extend({'seed': seed, 'motifs': motifs, **r} for r in rows)
            actions.extend({'seed': seed, 'motifs': motifs, **a} for a in act)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (('metrics.csv', metrics), ('actions.csv', actions)):
        write_csv(args.output / name, rows)
    receipt = {'schema_version': 1, 'kind': 'connected_hub_control_dev',
               'cells': 12, 'metric_rows': len(metrics), 'action_rows': len(actions),
               'archived_parity': 'actions exact; metrics tolerance 1e-10',
               'protocol_sha256': digest(args.protocol),
               'metrics_sha256': digest(args.output / 'metrics.csv'),
               'actions_sha256': digest(args.output / 'actions.csv')}
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for motifs in protocol['motifs']:
        cells = [{r['method']: r for r in metrics if r['seed'] == seed and
                  r['motifs'] == motifs} for seed in protocol['seeds']]
        risk = 'risk_pair'
        for method in (risk, *protocol['new_arms']):
            values = [c[method]['feasible_motif_mse'] for c in cells]
            print(f'k={motifs} {method}: mean={sum(values)/len(values):.6f}, values={values}')
        for method in protocol['new_arms']:
            print(f'k={motifs} risk wins vs {method}: '
                  f'{sum(c[risk]["feasible_motif_mse"] < c[method]["feasible_motif_mse"] for c in cells)}/6')
    print(f'validated {len(metrics)} metrics, {len(actions)} actions, archived parity')


if __name__ == '__main__':
    main()

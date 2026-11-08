#!/usr/bin/env python3
"""Frozen motif-selection/fixed-value development comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from connected_acquisition import experiment


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--archive', type=Path, default=Path(
        'results/local_connected_factorial_hub_dev_20260929'))
    p.add_argument('--protocol', type=Path, default=Path(
        'docs/development/guidance/protocol_connected_fixed_value_dev_2026-09-29.json'))
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    protocol = json.loads(a.protocol.read_text())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('ACE_SOURCE_REVISION must match HEAD')
    old_receipt = json.loads((a.archive / 'complete.json').read_text())
    for name in ('metrics.csv', 'actions.csv'):
        assert digest(a.archive / name) == old_receipt[name.removesuffix('.csv') + '_sha256']
    old_metrics = list(csv.DictReader((a.archive / 'metrics.csv').open()))
    old_actions = list(csv.DictReader((a.archive / 'actions.csv').open()))
    metrics, actions = [], []
    for seed in protocol['seeds']:
        rows, own_actions, _ = experiment(seed, 30, 10, .15, 4,
            include_balanced=True, topology='binary_tree', include_hub=True,
            include_factorial=True, include_fixed_value=True)
        prior_rows = [r for r in old_metrics if int(r['seed']) == seed]
        prior_actions = [r for r in old_actions if int(r['seed']) == seed]
        assert len(rows) == len(prior_rows) + 1
        for new, prior in zip(rows, prior_rows):
            for key, value in new.items():
                if isinstance(value, float):
                    assert math.isclose(value, float(prior[key]), rel_tol=0,
                                        abs_tol=1e-10), (seed, key)
                else:
                    assert str(value) == prior[key], (seed, key)
        assert all(all(str(v) == prior[k] for k, v in new.items())
                   for new, prior in zip(own_actions, prior_actions))
        new = next(r for r in rows if r['method'] == protocol['new_arm'])
        own = [r for r in own_actions if r['method'] == protocol['new_arm']]
        assert len(own) == 5 and new['cost_spent'] == 360 and new['samples'] == 40
        by_motif = {}
        cycle = ['-2.0,-2.0', '-2.0,2.0', '2.0,-2.0', '2.0,2.0']
        for action in own:
            motif = action['motif']
            count = by_motif.get(motif, 0)
            assert action['levels'] == cycle[count % 4]
            by_motif[motif] = count + 1
        metrics.extend({'seed': seed, **r} for r in rows)
        actions.extend({'seed': seed, **r} for r in own_actions)
    a.output.mkdir(parents=True)
    write_csv(a.output / 'metrics.csv', metrics)
    write_csv(a.output / 'actions.csv', actions)
    receipt = {'schema_version': 1, 'kind': 'connected_fixed_value_dev',
               'source_revision': revision, 'protocol_sha256': digest(a.protocol),
               'systems': len(protocol['seeds']), 'metric_rows': len(metrics),
               'action_rows': len(actions), 'new_synthetic_responses': 120,
               'closed_model_calls': 0,
               'metrics_sha256': digest(a.output / 'metrics.csv'),
               'actions_sha256': digest(a.output / 'actions.csv')}
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for method in (protocol['new_arm'], *protocol['comparators']):
        values = [float(r['feasible_motif_mse']) for r in metrics if r['method'] == method]
        print(method, values, 'mean', sum(values)/len(values))
    print('validated archived parity, costs, fixed-value cycles and hashes')


if __name__ == '__main__':
    main()

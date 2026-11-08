#!/usr/bin/env python3
"""Run the frozen factorial-hub development screen with archived-arm parity."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--archive', type=Path, default=Path(
        'results/local_connected_binary_tree_dev_20260928'))
    parser.add_argument('--protocol', type=Path, default=Path(
        'docs/development/guidance/protocol_connected_factorial_hub_dev_2026-09-29.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != rev:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    metrics, actions = [], []
    for seed in protocol['seeds']:
        old = args.archive / 'motifs_10' / f'seed_{seed}'
        receipt = json.loads((old / 'complete.json').read_text())
        for name in ('metrics.csv', 'actions.csv', 'system.json'):
            assert digest(old / name) == receipt[name + '_sha256']
        old_metrics = list(csv.DictReader((old / 'metrics.csv').open()))
        old_actions = list(csv.DictReader((old / 'actions.csv').open()))
        rows, own_actions, spec = experiment(seed, 30, 10, .15, 4,
            include_balanced=True, topology='binary_tree', include_hub=True,
            include_factorial=True)
        assert len(rows) == len(old_metrics) + 1
        for new, prior in zip(rows, old_metrics):
            for key, value in new.items():
                if isinstance(value, float):
                    assert math.isclose(value, float(prior[key]), rel_tol=0,
                                        abs_tol=1e-10), (seed, key)
                else:
                    assert str(value) == prior[key], (seed, key)
        assert all(all(str(v) == prior[k] for k, v in new.items())
                   for new, prior in zip(own_actions, old_actions))
        new = next(row for row in rows if row['method'] == protocol['new_arm'])
        new_actions = [a for a in own_actions if a['method'] == protocol['new_arm']]
        assert len(new_actions) == 5 and new['cost_spent'] == 360 and new['samples'] == 40
        assert [a['motif'] for a in new_actions] == [0, 0, 0, 0, 1]
        assert [a['levels'] for a in new_actions] == [
            '-2.0,-2.0', '-2.0,2.0', '2.0,-2.0', '2.0,2.0', '-2.0,-2.0']
        metrics.extend({'seed': seed, **r} for r in rows)
        actions.extend({'seed': seed, **a} for a in own_actions)
    args.output.mkdir(parents=True)
    write_csv(args.output / 'metrics.csv', metrics)
    write_csv(args.output / 'actions.csv', actions)
    receipt = {'schema_version': 1, 'kind': 'connected_factorial_hub_dev',
               'source_revision': rev, 'protocol_sha256': digest(args.protocol),
               'systems': len(protocol['seeds']), 'metric_rows': len(metrics),
               'action_rows': len(actions), 'new_queries': 120,
               'closed_model_calls': 0,
               'metrics_sha256': digest(args.output / 'metrics.csv'),
               'actions_sha256': digest(args.output / 'actions.csv')}
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for method in (protocol['new_arm'], *protocol['comparators']):
        values = [float(r['feasible_motif_mse']) for r in metrics if r['method'] == method]
        print(method, values, 'mean', sum(values) / len(values))
    print('validated archived parity, actions, costs and hashes')


if __name__ == '__main__':
    main()

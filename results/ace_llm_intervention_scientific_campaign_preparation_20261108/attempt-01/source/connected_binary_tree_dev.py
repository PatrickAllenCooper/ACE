#!/usr/bin/env python3
"""Frozen bounded-degree connected-SCM screen with exact per-cell receipts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from connected_acquisition import experiment, write_csv
from connected_motif import make_system, sample


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoke(seed: int, nodes: int, motifs: int, root_sd: float) -> None:
    system = make_system(seed, nodes, motifs, root_sd, topology='binary_tree')
    assert all(system.parents[j][0] == system.children[(j - 1) // 2]
               for j in range(1, motifs))
    direct = Counter(p[0] for p in system.parents[1:])
    assert max(direct.values()) <= 2 and direct[system.children[0]] == 2
    assert len(system.edges) == nodes - 1 and all(a < b for a, b in system.edges)
    action = (1, (system.children[0], system.parents[1][1]), (2., -2.))
    _, _, natural = sample(system, np.random.default_rng(17), 8, action)
    assert not natural[:, 0].any() and natural[:, 1:].all()


def run_cell(output: Path, seed: int, motifs: int, settings: dict, rev: str) -> dict:
    smoke(seed, settings['nodes'], motifs, settings['root_sd'])
    rows, actions, spec = experiment(
        seed, settings['nodes'], motifs, settings['root_sd'],
        settings['actuator_penalty'], budget=settings['cost_budget'],
        batch=settings['batch'], include_balanced=True,
        topology='binary_tree', include_hub=True)
    assert [r['method'] for r in rows] == settings['arms']
    assert spec['source_revision'] == rev and spec['topology'] == 'binary_tree'
    for row in rows:
        own = [a for a in actions if a['method'] == row['method']]
        assert len(own) == row['steps']
        assert row['samples'] == settings['batch'] * len(own)
        assert row['cost_spent'] == row['samples'] + settings['actuator_penalty'] * row['actuator_uses']
        assert row['cost_spent'] <= settings['cost_budget']
        assert own[-1]['cumulative_cost'] == row['cost_spent']
        assert sum(a['masked_child_labels'] for a in own) == row['masked_child_labels']
        assert all(a['natural_child_labels'] + a['masked_child_labels'] ==
                   settings['batch'] * motifs for a in own)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'metrics.csv', rows)
    write_csv(output / 'actions.csv', actions)
    (output / 'system.json').write_text(json.dumps(spec, indent=2, sort_keys=True) + '\n')
    receipt = {'schema_version': 1, 'kind': 'connected_binary_tree_dev',
               'seed': seed, 'motifs': motifs, 'source_revision': rev,
               'rows': len(rows), 'actions': len(actions),
               **{name + '_sha256': digest(output / name)
                  for name in ('metrics.csv', 'actions.csv', 'system.json')}}
    (output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return {'seed': seed, 'motifs': motifs,
            **{r['method'] + '_mse': r['feasible_motif_mse'] for r in rows}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=Path(
        'docs/development/guidance/protocol_connected_binary_tree_dev_2026-09-28.json'))
    args = parser.parse_args()
    settings = json.loads(args.protocol.read_text())
    rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != rev:
        raise ValueError('ACE_SOURCE_REVISION must equal the checked-out source revision')
    summaries = []
    for motifs in settings['motifs']:
        for seed in settings['seeds']:
            summaries.append(run_cell(args.output / f'motifs_{motifs}' / f'seed_{seed}',
                                      seed, motifs, settings, rev))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'summary.csv', summaries)
    receipt = {'schema_version': 1, 'kind': 'connected_binary_tree_dev_suite',
               'source_revision': rev, 'protocol_sha256': digest(args.protocol),
               'cells': len(summaries), 'summary_sha256': digest(args.output / 'summary.csv')}
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'validated {len(summaries)} bounded-degree development cells')


if __name__ == '__main__':
    main()

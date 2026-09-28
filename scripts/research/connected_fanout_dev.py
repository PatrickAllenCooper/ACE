#!/usr/bin/env python3
"""Run the frozen fanout development screen with per-cell custody receipts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from connected_acquisition import experiment, write_csv
from connected_motif import make_system, sample


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoke() -> None:
    import numpy as np
    chain = make_system(300, 30, 10, .15)
    fanout = make_system(300, 30, 10, .15, topology='fanout')
    assert chain.parents[2][0] != fanout.parents[2][0]
    assert all(p[0] == fanout.children[0] for p in fanout.parents[1:])
    assert len(set(fanout.edges)) == len(fanout.edges)
    assert all(a < b for a, b in fanout.edges)
    assert len(fanout.edges) == 2 * fanout.motifs + 30 - (2 * fanout.motifs + 1)
    action = (1, (fanout.children[0], fanout.parents[1][1]), (2., -2.))
    _, _, natural = sample(fanout, np.random.default_rng(1), 8, action)
    assert not natural[:, 0].any() and natural[:, 1:].all()


def run_cell(output: Path, seed: int, motifs: int, settings: dict, source_rev: str) -> dict:
    rows, actions, spec = experiment(seed, settings['nodes'], motifs,
                                     settings['root_sd'], settings['actuator_penalty'],
                                     budget=settings['cost_budget'], batch=settings['batch'],
                                     include_balanced=True, topology='fanout')
    assert [r['method'] for r in rows] == settings['arms']
    assert all(r['cost_spent'] <= settings['cost_budget'] for r in rows)
    assert all(r['cost_spent'] == r['steps'] * settings['batch'] *
               (1 + settings['actuator_penalty'] * (2 if r['method'].endswith('pair') else 1))
               for r in rows)
    for row in rows:
        arm_actions = [a for a in actions if a['method'] == row['method']]
        assert len(arm_actions) == row['steps']
        assert arm_actions[-1]['cumulative_cost'] == row['cost_spent']
        assert sum(a['masked_child_labels'] for a in arm_actions) == row['masked_child_labels']
    assert spec['topology'] == 'fanout'
    assert spec['source_revision'] == source_rev
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'metrics.csv', rows)
    write_csv(output / 'actions.csv', actions)
    (output / 'system.json').write_text(json.dumps(spec, indent=2, sort_keys=True) + '\n')
    receipt = {'schema_version': 1, 'kind': 'connected_fanout_dev',
               'seed': seed, 'motifs': motifs, 'source_revision': source_rev,
               'rows': len(rows), 'actions': len(actions),
               **{name + '_sha256': digest(output / name)
                  for name in ('metrics.csv', 'actions.csv', 'system.json')}}
    (output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return {'seed': seed, 'motifs': motifs,
            **{r['method'] + '_mse': r['feasible_motif_mse'] for r in rows}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(
        'docs/development/guidance/protocol_connected_fanout_dev_2026-09-28.json'))
    args = parser.parse_args()
    settings = json.loads(args.protocol.read_text())
    source_rev = os.environ.get('ACE_SOURCE_REVISION')
    if not source_rev or len(source_rev) != 40:
        raise ValueError('Set ACE_SOURCE_REVISION to the committed implementation SHA')
    smoke()
    summaries = []
    for motifs in settings['motifs']:
        for seed in settings['seeds']:
            summaries.append(run_cell(args.output / f'motifs_{motifs}' / f'seed_{seed}',
                                      seed, motifs, settings, source_rev))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'summary.csv', summaries)
    receipt = {'schema_version': 1, 'kind': 'connected_fanout_dev_suite',
               'source_revision': source_rev, 'protocol_sha256': digest(args.protocol),
               'cells': len(summaries), 'summary_sha256': digest(args.output / 'summary.csv')}
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'validated {len(summaries)} fanout cells; summary={args.output / "summary.csv"}')


if __name__ == '__main__':
    main()

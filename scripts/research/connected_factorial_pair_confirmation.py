#!/usr/bin/env python3
"""Fresh-system, frozen comparison of risk and factorial pair acquisition."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import t

from connected_acquisition import experiment


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def validate_cell(cell: Path, config: dict, revision: str) -> dict[str, dict]:
    receipt = json.loads((cell / 'complete.json').read_text())
    assert receipt['source_revision'] == revision
    assert receipt['seed'] == int(cell.name.split('_')[1])
    for name in ('metrics.csv', 'actions.csv', 'system.json'):
        assert digest(cell / name) == receipt[name + '_sha256']
    spec = json.loads((cell / 'system.json').read_text())
    assert spec['seed'] == receipt['seed'] and spec['source_revision'] == revision
    for field, key in [('nodes', 'nodes'), ('motifs', 'motifs'),
                       ('root_sd', 'root_sd'), ('penalty', 'actuator_penalty'),
                       ('budget', 'cost_budget')]:
        assert spec[field] == config[key]
    assert spec['topology'] == config['topology']
    rows = list(csv.DictReader((cell / 'metrics.csv').open()))
    actions = list(csv.DictReader((cell / 'actions.csv').open()))
    assert [r['method'] for r in rows] == config['arms']
    assert len(actions) == receipt['actions'] and len(rows) == receipt['rows']
    for row in rows:
        own = [a for a in actions if a['method'] == row['method']]
        assert len(own) == int(row['steps'])
        assert int(row['samples']) == config['batch'] * len(own)
        assert int(row['cost_spent']) == int(row['samples']) + config['actuator_penalty'] * int(row['actuator_uses'])
        assert int(row['cost_spent']) <= config['cost_budget']
        assert int(own[-1]['cumulative_cost']) == int(row['cost_spent'])
        assert math.isfinite(float(row['feasible_motif_mse']))
        assert float(row['feasible_motif_mse']) >= 0
        if row['method'].endswith('pair'):
            assert int(row['cost_spent']) == 360 and int(row['samples']) == 40
    if 'factorial_hub_pair' in config['arms']:
        own = [a for a in actions if a['method'] == 'factorial_hub_pair']
        assert [int(a['motif']) for a in own] == [0, 0, 0, 0, 1]
        assert [a['levels'] for a in own] == [
            '-2.0,-2.0', '-2.0,2.0', '2.0,-2.0', '2.0,2.0', '-2.0,-2.0']
    if 'risk_motif_fixed_value_pair' in config['arms']:
        own = [a for a in actions if a['method'] == 'risk_motif_fixed_value_pair']
        visits = {}
        cycle = ['-2.0,-2.0', '-2.0,2.0', '2.0,-2.0', '2.0,2.0']
        for action in own:
            motif = action['motif']
            assert action['levels'] == cycle[visits.get(motif, 0) % 4]
            visits[motif] = visits.get(motif, 0) + 1
    return {r['method']: r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=Path(
        'docs/development/guidance/protocol_connected_factorial_pair_confirmation_2026-09-29.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.protocol.read_text())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    assert len(config['fresh_seeds']) == len(set(config['fresh_seeds']))
    assert len(config['fresh_seeds']) >= 20
    assert not set(config['fresh_seeds']) & set(config['excluded_development_seeds'])
    args.output.mkdir(parents=True)
    summary = []
    for seed in config['fresh_seeds']:
        rows, actions, spec = experiment(
            seed, config['nodes'], config['motifs'], config['root_sd'],
            config['actuator_penalty'], budget=config['cost_budget'],
            batch=config['batch'], topology=config['topology'],
            include_balanced=True, include_hub=True, include_factorial=True,
            include_fixed_value=('risk_motif_fixed_value_pair' in config['arms']))
        assert [r['method'] for r in rows] == config['arms']
        cell = args.output / f'seed_{seed}'
        cell.mkdir()
        write_csv(cell / 'metrics.csv', rows)
        write_csv(cell / 'actions.csv', actions)
        (cell / 'system.json').write_text(json.dumps(spec, indent=2, sort_keys=True) + '\n')
        receipt = {'schema_version': 1, 'kind': 'connected_factorial_pair_confirmation',
                   'seed': seed, 'source_revision': revision,
                   'rows': len(rows), 'actions': len(actions),
                   **{name + '_sha256': digest(cell / name)
                      for name in ('metrics.csv', 'actions.csv', 'system.json')}}
        (cell / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
        validated = validate_cell(cell, config, revision)
        summary.append({'seed': seed, **{
            method + '_mse': validated[method]['feasible_motif_mse']
            for method in config['arms']}})
    write_csv(args.output / 'summary.csv', summary)
    risk = np.array([float(r['risk_pair_mse']) for r in summary])
    comparator = config.get('primary_comparator', 'factorial_hub_pair')
    fixed = np.array([float(r[comparator+'_mse']) for r in summary])
    difference = risk - fixed
    difference_mean = float(np.mean(difference))
    half_width = float(t.ppf(.975, len(difference)-1) *
                       np.std(difference, ddof=1) / np.sqrt(len(difference)))
    ratio = float(np.mean(risk) / np.mean(fixed))
    analysis = {'primary_comparator': comparator,
                'primary_ratio_of_means': ratio,
                'primary_mean_paired_difference': difference_mean,
                'primary_paired_t_95_interval': [difference_mean-half_width,
                                                 difference_mean+half_width],
                'primary_gate_pass': bool(ratio < config.get('primary_ratio_threshold', .8)
                                          and difference_mean+half_width < 0),
                'means': {method: float(np.mean([float(r[method+'_mse']) for r in summary]))
                          for method in config['arms']},
                'risk_wins_against_comparator': int(np.sum(difference < 0))}
    (args.output / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    total_samples = sum(int(r['samples']) for seed in config['fresh_seeds']
                        for r in csv.DictReader((args.output / f'seed_{seed}' /
                                                 'metrics.csv').open()))
    receipt = {'schema_version': 1, 'kind': 'connected_factorial_pair_confirmation_suite',
               'source_revision': revision, 'protocol_sha256': digest(args.protocol),
               'systems': len(summary), 'arms_per_system': len(config['arms']),
               'synthetic_response_arm_counts': total_samples,
               'closed_model_calls': 0,
               'summary_sha256': digest(args.output / 'summary.csv'),
               'analysis_sha256': digest(args.output / 'analysis.json')}
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(analysis, indent=2))
    print('validated', len(summary), 'fresh systems,', total_samples,
          'arm-response counts')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Exploratory direct IVR-versus-variance comparison on archived PEV replication."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

from validate_cell import valid


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('results/research_pev_endpoint_replication_v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    primary_receipt = json.loads((args.root / 'comparison_receipt.json').read_text())
    assert primary_receipt['comparison_sha256'] == sha(args.root / 'comparison.csv')
    with (args.root / 'comparison.csv').open() as stream:
        comparison = list(csv.DictReader(stream))
    seeds = list(range(6000, 6020))
    by_key = {(int(r['seed']), r['method']): r for r in comparison}
    assert len(comparison) == len(by_key) == 60
    rows, source_digests = [], []
    for seed in seeds:
        arm = {}
        systems = {}
        for method in ('pev', 'pev_var'):
            directory = args.root / 'shift30' / method / f'seed_{seed}'
            ok, reason = valid(directory, 'persistent')
            assert ok, (directory, reason)
            system = json.loads((directory / 'system.json').read_text())
            systems[method] = system
            with (directory / 'trajectory.csv').open() as stream:
                steps = list(csv.DictReader(stream))
            assert len(steps) == 32 and int(steps[-1]['query_samples']) == 2000
            assert float(steps[-1]['feasible_mean_nonroot_loss']) == float(
                by_key[(seed, method)]['final_feasible_mean_nonroot'])
            arm[method] = steps
            source_digests.append((seed, method, sha(directory / 'trajectory.csv')))
        assert all(systems['pev'][key] == systems['pev_var'][key] for key in
                   ('family', 'seed', 'graph', 'forms', 'coeffs', 'nodes', 'noise_std'))
        a, b = arm['pev'], arm['pev_var']
        paired_a = float(by_key[(seed, 'pev')]['final_feasible_mean_nonroot'])
        paired_b = float(by_key[(seed, 'pev_var')]['final_feasible_mean_nonroot'])
        rows.append({'seed': seed, 'pev_mse': paired_a, 'pev_var_mse': paired_b,
                     'var_minus_pev_mse': paired_b - paired_a,
                     'same_target_steps': sum(x['target'] == y['target'] for x, y in zip(a, b)),
                     'same_target_value_steps': sum((x['target'], x['value']) ==
                                                    (y['target'], y['value']) for x, y in zip(a, b)),
                     'target_set_intersection': len({x['target'] for x in a} &
                                                    {y['target'] for y in b}),
                     'pev_unique_targets': len({x['target'] for x in a}),
                     'var_unique_targets': len({y['target'] for y in b}),
                     'pev_mean_abs_value': sum(abs(float(x['value'])) for x in a) / 32,
                     'var_mean_abs_value': sum(abs(float(y['value'])) for y in b) / 32})
    differences = np.array([r['var_minus_pev_mse'] for r in rows])
    mean = float(differences.mean())
    ci = stats.t.interval(.95, 19, loc=mean, scale=stats.sem(differences))
    summary = {'status': 'exploratory secondary contrast, not a frozen primary hypothesis',
               'systems': len(rows), 'source_revision': primary_receipt['source_revision'],
               'metric': 'final noise-free feasible nonroot MSE, lower is better',
               'difference_orientation': 'PEV-var minus PEV; positive favors PEV',
               'mean_pev': float(np.mean([r['pev_mse'] for r in rows])),
               'mean_pev_var': float(np.mean([r['pev_var_mse'] for r in rows])),
               'mean_difference': mean, 'paired_t_ci95': list(map(float, ci)),
               'paired_t_p_two_sided': float(stats.ttest_1samp(differences, 0).pvalue),
               'wilcoxon_p_two_sided': float(stats.wilcoxon(differences).pvalue),
               'pev_lower_mse_systems': sum(differences > 0),
               'var_lower_mse_systems': sum(differences < 0),
               'mean_same_target_steps_of_32': float(np.mean([r['same_target_steps'] for r in rows])),
               'mean_same_target_value_steps_of_32': float(np.mean([
                   r['same_target_value_steps'] for r in rows])),
               'mean_target_set_intersection': float(np.mean([
                   r['target_set_intersection'] for r in rows])),
               'mean_unique_targets_pev': float(np.mean([
                   r['pev_unique_targets'] for r in rows])),
               'mean_unique_targets_var': float(np.mean([
                   r['var_unique_targets'] for r in rows])),
               'mean_abs_action_value_pev': float(np.mean([
                   r['pev_mean_abs_value'] for r in rows])),
               'mean_abs_action_value_var': float(np.mean([
                   r['var_mean_abs_value'] for r in rows])),
               'no_new_queries': True, 'closed_model_calls': 0}
    args.output.mkdir(parents=True, exist_ok=False)
    per_system = args.output / 'per_system.csv'
    with per_system.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_file = args.output / 'summary.json'
    summary_file.write_text(json.dumps(summary, indent=2) + '\n')
    receipt = {'schema_version': 1, 'kind': 'pev_scoring_diagnostic',
               'validated_cells': 40, 'source_comparison_sha256': sha(args.root / 'comparison.csv'),
               'source_trajectory_manifest_sha256': hashlib.sha256(json.dumps(
                   source_digests, separators=(',', ':')).encode()).hexdigest(),
               'per_system_sha256': sha(per_system), 'summary_sha256': sha(summary_file)}
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('20 systems; var-minus-PEV mean', mean, '95% CI', ci,
          'paired p', summary['paired_t_p_two_sided'], 'PEV wins',
          summary['pev_lower_mse_systems'])


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Prospectively frozen finite-source transfer mixture on untouched systems."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.special import expit

from agenda_runner import features, posterior
from transfer_finite_source_target_dev import fit, score, sha
from transfer_finite_source_switch_dev import log_evidence
from transfer_safe_switch_dev import target


def source_observations(seed: int, n: int, spec: dict):
    latent_old, _, _, _, _, _ = target(seed, 1, 'family')
    rng = np.random.default_rng(seed + spec['source_stream_offset'])
    x = rng.uniform(-2, 2, (30, n, 2))
    design = features(x.reshape(-1, 2)).reshape(30, n, 6)
    y = np.einsum('nid,nd->ni', design, latent_old, optimize=False)
    y += rng.normal(0, spec['source_noise_sd'], (30, n))
    means, covs = [], []
    for node in range(30):
        mean, covariance = posterior(x[node], y[node], np.zeros(6), .25,
                                     sigma=spec['source_noise_sd'])
        means.append(mean)
        covs.append(covariance)
    return x, y, np.stack(means), np.stack(covs)


def predictions(x: np.ndarray, y: np.ndarray, source_mean: np.ndarray,
                source_cov: np.ndarray, noise_sd: float):
    """Policy boundary: only source posterior and acquired target x/y enter."""
    zero, broad = np.zeros(6), np.eye(6) / .25
    warm = fit(x, y, source_mean, source_cov, noise_sd)
    scratch = fit(x, y, zero, broad, noise_sd)
    log_bf = (log_evidence(x, y, zero, broad, noise_sd) -
              log_evidence(x, y, source_mean, source_cov, noise_sd))
    scratch_weight = float(expit(log_bf + np.log(.1 / .9)))
    mixture = (1 - scratch_weight) * warm + scratch_weight * scratch
    return {'frozen_source': source_mean, 'source_warm': warm,
            'scratch': scratch, 'soft_mixture': mixture}, scratch_weight


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    spec = json.loads(a.protocol.read_text())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('pinned ACE_SOURCE_REVISION required')
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    all_rows = []
    source_count = target_count = 0
    for seed in spec['system_seeds']:
        for source_n in spec['source_training_examples_per_node']:
            source_cell = a.output / f'seed_{seed}' / f'source_n_{source_n}' / 'source'
            source_cell.mkdir(parents=True)
            sx, sy, means, covs = source_observations(seed, source_n, spec)
            np.savez_compressed(source_cell / 'source_data.npz', x=sx, y=sy)
            np.savez_compressed(source_cell / 'source_posteriors.npz',
                                mean=means, covariance=covs)
            (source_cell / 'complete.json').write_text(json.dumps({
                'source_revision': revision, 'protocol_sha256': sha(a.protocol),
                'source_examples': 30 * source_n,
                'source_data_sha256': sha(source_cell / 'source_data.npz'),
                'source_posteriors_sha256': sha(source_cell / 'source_posteriors.npz'),
                'closed_model_calls': 0}, indent=2) + '\n')
            source_count += 30 * source_n
            for change_type in spec['change_types']:
                for changed in spec['changed_node_counts']:
                    _, truth, changed_ids, x, y, test_phi = target(seed, changed, change_type)
                    changed_mask = np.zeros(30, dtype=bool)
                    changed_mask[changed_ids] = True
                    budget = spec['target_budget']
                    extra = budget - 4 * 30
                    counts = [4 + extra // 30 + int(i < extra % 30) for i in range(30)]
                    if sum(counts) != budget or max(counts) > x.shape[1]:
                        raise ValueError('invalid acquired target count')
                    cell = a.output / f'seed_{seed}' / f'source_n_{source_n}' / change_type / f'changed_{changed}'
                    cell.mkdir(parents=True)
                    rows = []
                    for node, n in enumerate(counts):
                        fits, scratch_weight = predictions(x[node, :n], y[node, :n],
                                                           means[node], covs[node], .15)
                        for method, prediction in fits.items():
                            rows.append({'seed': seed, 'source_n': source_n,
                                         'change_type': change_type, 'changed': changed,
                                         'node': node, 'changed_node': int(changed_mask[node]),
                                         'target_examples': n, 'method': method,
                                         'scratch_weight': scratch_weight,
                                         'mse': score(prediction, truth[node], test_phi)})
                    path = cell / 'node_metrics.csv'
                    with path.open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    (cell / 'complete.json').write_text(json.dumps({
                        'source_revision': revision, 'protocol_sha256': sha(a.protocol),
                        'source_posteriors_sha256': sha(source_cell / 'source_posteriors.npz'),
                        'node_metrics_sha256': sha(path), 'rows': len(rows),
                        'target_examples': budget, 'closed_model_calls': 0}, indent=2) + '\n')
                    target_count += budget
                    all_rows.extend(rows)
    summary = []
    for source_n in spec['source_training_examples_per_node']:
        for change_type in spec['change_types']:
            for changed in spec['changed_node_counts']:
                for changed_node in (0, 1):
                    for method in ('source_warm', 'scratch', 'frozen_source', 'soft_mixture'):
                        group = [r for r in all_rows if r['source_n'] == source_n and
                                 r['change_type'] == change_type and r['changed'] == changed and
                                 r['changed_node'] == changed_node and r['method'] == method]
                        expected = len(spec['system_seeds']) * (changed if changed_node else 30-changed)
                        if len(group) != expected:
                            raise ValueError('summary node count mismatch')
                        summary.append({'source_n': source_n, 'change_type': change_type,
                                        'changed': changed, 'changed_node': changed_node,
                                        'method': method, 'nodes': len(group),
                                        'mean_mse': float(np.mean([r['mse'] for r in group])),
                                        'mean_scratch_weight': float(np.mean(
                                            [r['scratch_weight'] for r in group]))})
    summary_path = a.output / 'summary.csv'
    with summary_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (a.output / 'suite_complete.json').write_text(json.dumps({
        'source_revision': revision, 'protocol_sha256': sha(a.protocol),
        'source_cells': len(spec['system_seeds']) * 2,
        'target_cells': len(spec['system_seeds']) * 2 * 2 * 3,
        'source_training_examples_total': source_count,
        'target_examples_total': target_count,
        'node_method_rows': len(all_rows), 'summary_sha256': sha(summary_path),
        'closed_model_calls': 0}, indent=2) + '\n')
    print('soft mixture:', len(spec['system_seeds']), 'systems,', target_count,
          'target and', source_count, 'source examples')


if __name__ == '__main__':
    main()

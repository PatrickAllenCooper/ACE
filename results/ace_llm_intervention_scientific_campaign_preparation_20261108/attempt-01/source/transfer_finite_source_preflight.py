#!/usr/bin/env python3
"""Counted finite-source posterior preflight, with source/evaluator files split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import norm

from agenda_runner import features, posterior
from transfer_safe_switch_dev import target


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.protocol.read_text())
    rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != rev:
        raise ValueError('pinned ACE_SOURCE_REVISION required')
    z90 = float(norm.ppf(.95))
    rows = []
    for seed in spec['development_seeds']:
        latent_source, _, _, _, _, _ = target(seed, 1, 'family')
        assert latent_source.shape == (30, 6)
        for n in spec['source_training_examples_per_node']:
            cell = a.output / f'seed_{seed}' / f'source_n_{n}'
            if cell.exists():
                raise FileExistsError(cell)
            cell.mkdir(parents=True)
            train_rng = np.random.default_rng(seed + spec['source_training_stream_offset'])
            x = train_rng.uniform(-2, 2, (30, n, 2))
            phi = features(x.reshape(-1, 2)).reshape(30, n, 6)
            y = np.einsum('nid,nd->ni', phi, latent_source, optimize=False)
            y += train_rng.normal(0, spec['source_noise_sd'], (30, n))
            means, covariances = [], []
            for i in range(30):
                mean, covariance = posterior(x[i], y[i], np.zeros(6), .25,
                                             sigma=spec['source_noise_sd'])
                means.append(mean)
                covariances.append(covariance)
            means, covariances = np.stack(means), np.stack(covariances)
            np.savez_compressed(cell / 'source_data.npz', x=x, y=y)
            np.savez_compressed(cell / 'source_posteriors.npz', mean=means, covariance=covariances)
            hold_rng = np.random.default_rng(seed + spec['source_holdout_stream_offset'])
            h = spec['source_holdout_examples_per_node']
            xh = hold_rng.uniform(-2, 2, (30, h, 2))
            ph = features(xh.reshape(-1, 2)).reshape(30, h, 6)
            truth = np.einsum('nid,nd->ni', ph, latent_source, optimize=False)
            observed = truth + hold_rng.normal(0, spec['source_noise_sd'], (30, h))
            pred = np.einsum('nid,nd->ni', ph, means, optimize=False)
            var = spec['source_noise_sd'] ** 2 + np.einsum(
                'nid,ndk,nik->ni', ph, covariances, ph, optimize=False)
            within = np.abs(observed - pred) <= z90 * np.sqrt(var)
            node_mse = np.mean((truth - pred) ** 2, axis=1)
            if not np.isfinite(var).all() or np.min(var) <= 0 or np.min(node_mse) <= 0:
                raise ValueError('invalid source posterior or zero node error')
            if any(np.min(np.linalg.eigvalsh(cov)) <= 0 for cov in covariances):
                raise ValueError('non-positive posterior covariance')
            evaluation = {'seed': seed, 'source_samples_per_node': n,
                          'source_train_examples': 30 * n,
                          'source_holdout_examples': 30 * h,
                          'clean_source_mse': float(node_mse.mean()),
                          'min_node_clean_source_mse': float(node_mse.min()),
                          'predictive_90pct_coverage': float(within.mean()),
                          'source_data_sha256': sha(cell / 'source_data.npz'),
                          'source_posteriors_sha256': sha(cell / 'source_posteriors.npz'),
                          'source_revision': rev,
                          'protocol_sha256': sha(a.protocol),
                          'target_outcomes_read_by_fitter': 0,
                          'closed_model_calls': 0}
            evaluation_path = cell / 'evaluation.json'
            evaluation_path.write_text(json.dumps(evaluation, indent=2) + '\n')
            (cell / 'complete.json').write_text(json.dumps({
                'source_revision': rev, 'source_data_sha256': evaluation['source_data_sha256'],
                'source_posteriors_sha256': evaluation['source_posteriors_sha256'],
                'evaluation_sha256': sha(evaluation_path),
                'source_train_examples': 30 * n,
                'source_holdout_examples': 30 * h}, indent=2) + '\n')
            rows.append({key: evaluation[key] for key in (
                'seed', 'source_samples_per_node', 'source_train_examples',
                'source_holdout_examples', 'clean_source_mse',
                'min_node_clean_source_mse', 'predictive_90pct_coverage')})
    with (a.output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {'source_revision': rev, 'protocol_sha256': sha(a.protocol),
               'cells': len(rows), 'source_systems': len(spec['development_seeds']),
               'source_train_examples_total': sum(r['source_train_examples'] for r in rows),
               'source_holdout_examples_total': sum(r['source_holdout_examples'] for r in rows),
               'summary_sha256': sha(a.output / 'summary.csv'),
               'private_target_results_read': 0, 'closed_model_calls': 0}
    (a.output / 'suite_complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('source preflight', len(rows), 'cells', receipt['source_train_examples_total'],
          'counted source examples')


if __name__ == '__main__':
    main()

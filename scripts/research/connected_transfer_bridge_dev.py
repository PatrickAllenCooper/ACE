#!/usr/bin/env python3
"""Development bridge from connected SCM observations to local transfer evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from connected_motif import make_system, sample


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def posterior(phi: np.ndarray, y: np.ndarray, mean: np.ndarray,
              covariance: np.ndarray, noise_sd: float):
    precision = np.linalg.inv(covariance)
    updated = np.linalg.inv(precision + phi.T @ phi / noise_sd**2)
    return updated @ (precision @ mean + phi.T @ y / noise_sd**2), updated


def log_predictive(phi: np.ndarray, y: np.ndarray, mean: np.ndarray,
                   covariance: np.ndarray, noise_sd: float) -> float:
    predictive = noise_sd**2 * np.eye(len(y)) + phi @ covariance @ phi.T
    chol = np.linalg.cholesky(predictive)
    residual = np.linalg.solve(chol, y - phi @ mean)
    return float(-.5 * (len(y)*np.log(2*np.pi) +
                        2*np.log(np.diag(chol)).sum() + residual @ residual))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    a.output.mkdir(parents=True)
    rows = []
    broad_mean, broad_cov = np.zeros(3), 4*np.eye(3)
    for seed in range(1600, 1612):
        source = make_system(seed, 30, 10, .15, topology='fanout')
        coefficients = source.coefficients.copy()
        coefficients[0, 0] += 1.0
        coefficients[5, 2] += .5
        coefficients[7, 0] -= .35
        target = replace(source, coefficients=coefficients)
        source_values, source_phi, source_natural = sample(
            source, np.random.default_rng(seed + 101), 64)
        target_values, target_phi, target_natural = sample(
            target, np.random.default_rng(seed + 202), 14)
        paired_source, _, _ = sample(source, np.random.default_rng(seed+303), 128)
        paired_target, _, _ = sample(target, np.random.default_rng(seed+303), 128)
        if not source_natural.all() or not target_natural.all():
            raise AssertionError('missing natural mechanism observations')
        for n in (16, 64):
            scores = []
            source_fits = []
            for motif, child in enumerate(source.children):
                mean, cov = posterior(source_phi[:n, motif], source_values[:n, child],
                                      broad_mean, broad_cov, .15)
                source_fits.append(mean)
                assay_phi = target_phi[:4, motif]
                assay_y = target_values[:4, child]
                score = (log_predictive(assay_phi, assay_y, broad_mean, broad_cov, .15) -
                         log_predictive(assay_phi, assay_y, mean, cov, .15))
                scores.append(score)
            order = list(np.lexsort((np.arange(10), -np.asarray(scores))))
            for motif, child in enumerate(source.children):
                parents = source.parents[motif]
                changed = motif in (0, 5, 7)
                if changed != (not np.array_equal(source.coefficients[motif],
                                                   target.coefficients[motif])):
                    raise AssertionError('change label mismatch')
                # Paired observational distributions isolate the propagation
                # caused by the upstream change, separate from local changes.
                parent_shift = float(np.mean((paired_source[:, parents[0]] -
                                              paired_target[:, parents[0]])**2))
                rows.append({'seed': seed, 'source_n': n, 'motif': motif,
                             'child_node': child, 'changed': int(changed),
                             'evidence_rank': order.index(motif)+1,
                             'scratch_minus_source_log_predictive': scores[motif],
                             'paired_first_parent_shift_mse': parent_shift,
                             'source_posterior_coefficient_error': float(np.linalg.norm(
                                 source_fits[motif]-source.coefficients[motif])**2),
                             'source_revision': revision})
    with (a.output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for n in (16, 64):
        subset = [r for r in rows if r['source_n'] == n]
        summary[str(n)] = {
            'mean_changed_top4_recall': float(np.mean([
                sum(r['evidence_rank'] <= 4 for r in subset
                    if r['seed'] == seed and r['changed']) / 3
                for seed in range(1600, 1612)])),
            'mean_unchanged_top4_false_positives': float(np.mean([
                sum(r['evidence_rank'] <= 4 for r in subset
                    if r['seed'] == seed and not r['changed'])
                for seed in range(1600, 1612)])),
            'unchanged_downstream_motifs_with_parent_shift': int(sum(
                not r['changed'] and r['paired_first_parent_shift_mse'] > .001
                for r in subset))}
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    receipt = {'source_revision': revision, 'systems': 12, 'source_sizes': [16,64],
               'motifs_per_system': 10, 'changed_motifs_per_system': 3,
               'source_generated_trajectories': 12*64,
               'target_generated_trajectories': 12*14,
               'paired_diagnostic_trajectories_per_domain': 12*128,
               'source_motif_training_response_uses_per_source_size': [12*10*16,12*10*64],
               'target_assay_motif_response_uses_per_source_size': 12*10*4,
               'unique_target_assay_motif_responses': 12*10*4,
               'closed_model_calls': 0,
               'metrics_sha256': sha(a.output/'metrics.csv'),
               'summary_sha256': sha(a.output/'summary.json')}
    (a.output/'complete.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

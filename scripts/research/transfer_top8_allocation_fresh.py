#!/usr/bin/env python3
"""Frozen top-eight assay allocation on fresh finite-source transfer systems."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from transfer_finite_source_target_dev import score, sha
from transfer_finite_source_switch_dev import log_evidence
from transfer_soft_mixture_fresh import predictions, source_observations
from transfer_safe_switch_dev import target


def allocate(x: np.ndarray, y: np.ndarray, means: np.ndarray,
             covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    scratch_mean, scratch_cov = np.zeros(6), np.eye(6) / .25
    scores = np.array([
        log_evidence(x[node, :4], y[node, :4], scratch_mean, scratch_cov, .15) -
        log_evidence(x[node, :4], y[node, :4], means[node], covariances[node], .15)
        for node in range(30)])
    order = np.lexsort((np.arange(30), -scores))
    top_eight = [int(i) for i in order[:8]]
    adaptive = np.full(30, 4, dtype=int)
    adaptive[top_eight] += 10
    uniform = np.array([4 + 80 // 30 + int(i < 80 % 30) for i in range(30)])
    if adaptive.sum() != uniform.sum() or adaptive.sum() != 200:
        raise ValueError('target allocation budget mismatch')
    return adaptive, uniform, top_eight


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
    source_total = unique_total = target_arm_total = 0
    for seed in spec['system_seeds']:
        for source_n in spec['source_training_examples_per_node']:
            source_cell = a.output / f'seed_{seed}' / f'source_n_{source_n}' / 'source'
            source_cell.mkdir(parents=True)
            sx, sy, means, covariances = source_observations(seed, source_n, spec)
            np.savez_compressed(source_cell / 'source_data.npz', x=sx, y=sy)
            np.savez_compressed(source_cell / 'source_posteriors.npz',
                                mean=means, covariance=covariances)
            (source_cell / 'complete.json').write_text(json.dumps({
                'source_revision': revision, 'protocol_sha256': sha(a.protocol),
                'source_examples': 30 * source_n,
                'source_data_sha256': sha(source_cell / 'source_data.npz'),
                'source_posteriors_sha256': sha(source_cell / 'source_posteriors.npz'),
                'closed_model_calls': 0}, indent=2) + '\n')
            source_total += 30 * source_n
            for change_type in spec['change_types']:
                _, truth, changed_ids, x, y, test_phi = target(seed, 1, change_type)
                changed = int(changed_ids[0])
                adaptive, uniform, top_eight = allocate(x, y, means, covariances)
                if max(adaptive) > x.shape[1]:
                    raise ValueError('not enough target responses')
                cell = a.output / f'seed_{seed}' / f'source_n_{source_n}' / change_type
                cell.mkdir(parents=True)
                action = {'seed': seed, 'source_n': source_n, 'change_type': change_type,
                          'assay_responses': 120, 'top_eight': top_eight,
                          'adaptive_counts': adaptive.tolist(),
                          'uniform_counts': uniform.tolist(),
                          'target_responses_per_policy': 200,
                          'unique_acquired_prefix_responses': int(np.maximum(adaptive, uniform).sum()),
                          'simulator_generated_potential_responses': int(x.shape[0] * x.shape[1])}
                action_path = cell / 'actions.json'
                action_path.write_text(json.dumps(action, indent=2) + '\n')
                rows = []
                for allocation, counts in (('adaptive_top8', adaptive), ('uniform', uniform)):
                    for node, n in enumerate(counts):
                        fits, weight = predictions(x[node, :n], y[node, :n],
                                                   means[node], covariances[node], .15)
                        for method in spec['predictors']:
                            rows.append({'seed': seed, 'source_n': source_n,
                                         'change_type': change_type, 'allocation': allocation,
                                         'node': node, 'changed_node': int(node == changed),
                                         'target_examples': int(n), 'method': method,
                                         'scratch_weight': weight,
                                         'mse': score(fits[method], truth[node], test_phi)})
                path = cell / 'node_metrics.csv'
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                (cell / 'complete.json').write_text(json.dumps({
                    'source_revision': revision, 'protocol_sha256': sha(a.protocol),
                    'source_posteriors_sha256': sha(source_cell / 'source_posteriors.npz'),
                    'actions_sha256': sha(action_path), 'node_metrics_sha256': sha(path),
                    'rows': len(rows), 'target_responses_per_policy': 200,
                    'unique_acquired_prefix_responses': action['unique_acquired_prefix_responses'],
                    'closed_model_calls': 0}, indent=2) + '\n')
                all_rows.extend(rows)
                target_arm_total += 400
                unique_total += action['unique_acquired_prefix_responses']
    summary = []
    for source_n in spec['source_training_examples_per_node']:
        for change_type in spec['change_types']:
            for changed_node in (0, 1):
                for allocation in ('adaptive_top8', 'uniform'):
                    for method in spec['predictors']:
                        group = [r for r in all_rows if r['source_n'] == source_n and
                                 r['change_type'] == change_type and r['changed_node'] == changed_node
                                 and r['allocation'] == allocation and r['method'] == method]
                        expected = 20 * (1 if changed_node else 29)
                        if len(group) != expected:
                            raise ValueError('summary node count mismatch')
                        summary.append({'source_n': source_n, 'change_type': change_type,
                                        'changed_node': changed_node, 'allocation': allocation,
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
        'target_cells': len(spec['system_seeds']) * 2 * 2,
        'source_training_examples_total': source_total,
        'target_arm_responses_total': target_arm_total,
        'unique_acquired_prefix_responses_total': unique_total,
        'node_method_rows': len(all_rows), 'summary_sha256': sha(summary_path),
        'closed_model_calls': 0}, indent=2) + '\n')
    print('top-eight:', len(spec['system_seeds']), 'systems,', target_arm_total,
          'arm responses;', unique_total, 'unique acquired prefixes')


if __name__ == '__main__':
    main()

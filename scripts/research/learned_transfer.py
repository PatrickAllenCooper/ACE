#!/usr/bin/env python3
"""Fixed-data transfer screen with a library learned from independent sources.

This numerical gate precedes any neural module/hypernetwork build. Every target
method sees identical acquired examples; the source-library cost is explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from agenda_runner import features, posterior


def family_centers() -> np.ndarray:
    centers = np.zeros((4, 6))
    centers[:, :2] = (0.6, -0.3)
    for j in range(4):
        centers[j, j + 2] = 0.85
    return centers


def learn_source_library(seed: int = 9173, tasks_per_family: int = 10,
                         samples_per_task: int = 64) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    centers = family_centers()
    learned = []
    for family in range(4):
        estimates = []
        for _ in range(tasks_per_family):
            theta = centers[family] + rng.normal(0, 0.05, 6)
            x = rng.uniform(-2, 2, (samples_per_task, 2))
            phi = features(x)
            y = np.einsum('ij,j->i', phi, theta, optimize=False) + rng.normal(0, 0.15, samples_per_task)
            estimates.append(posterior(x, y, np.zeros(6), 0.25)[0])
        learned.append(np.mean(estimates, axis=0))
    return np.stack(learned), 4 * tasks_per_family * samples_per_task


def experiment(seed: int, changed: int, change_type: str,
               nodes: int = 30, guard_margin: float | None = None) -> tuple[list[dict], dict]:
    if not 0 < changed < nodes or change_type not in ('family', 'coefficient'):
        raise ValueError('Invalid change setting')
    source, source_samples = learn_source_library()
    rng = np.random.default_rng(seed + 2719)
    centers = family_centers()
    source_forms = rng.integers(0, 4, nodes)
    old = centers[source_forms] + rng.normal(0, 0.05, (nodes, 6))
    truth = old.copy()
    changed_ids = rng.choice(nodes, changed, replace=False)
    if change_type == 'family':
        truth[changed_ids] = centers[(source_forms[changed_ids] + 1) % 4] + rng.normal(0, 0.05, (changed, 6))
    else:
        for i in changed_ids:
            truth[i, 2 + source_forms[i]] += rng.choice((-1, 1)) * 0.6
    # A common passive assay, followed by a common fixed sequence of extra
    # examples per node. All methods use exactly the same target data.
    x = rng.uniform(-2, 2, (nodes, 14, 2))
    phi = features(x.reshape(-1, 2)).reshape(nodes, 14, 6)
    y = np.einsum('nid,nd->ni', phi, truth, optimize=False) + rng.normal(0, 0.15, (nodes, 14))
    test_rng = np.random.default_rng(seed + 91473)
    test_phi = features(test_rng.uniform(-2, 2, (1024, 2)))
    rows = []
    for budget in (120, 200, 400):
        extra = budget - 4 * nodes
        counts = np.array([4 + extra // nodes + (i < extra % nodes) for i in range(nodes)])
        if counts.max() > 14:
            raise ValueError('Insufficient fixed target examples')
        methods = ['scratch', 'warm', 'source_retrieval', 'source_mixture']
        if guard_margin is not None:
            methods.append('source_guarded')
        for method in methods:
            estimates = np.zeros_like(truth)
            selected_old = 0
            for i in range(nodes):
                proposals = np.vstack((old[i], source))
                passive_pred = np.einsum('ij,kj->ki', phi[i, :4], proposals, optimize=False)
                sse = np.sum((passive_pred - y[i, :4])**2, axis=1)
                if method == 'scratch':
                    estimates[i] = posterior(x[i, :counts[i]], y[i, :counts[i]], np.zeros(6), 0.25)[0]
                elif method == 'warm':
                    estimates[i] = posterior(x[i, :counts[i]], y[i, :counts[i]], old[i], 20.0)[0]
                elif method in ('source_retrieval', 'source_guarded'):
                    chosen = int(np.argmin(sse))
                    if method == 'source_guarded':
                        source_choice = int(np.argmin(sse[1:])) + 1
                        chosen = source_choice if sse[0] - sse[source_choice] > guard_margin else 0
                    selected_old += chosen == 0
                    estimates[i] = posterior(x[i, :counts[i]], y[i, :counts[i]], proposals[chosen], 20.0)[0]
                else:
                    logweights = -sse / (2 * 0.15**2) + np.log([0.5] + [0.125] * 4)
                    weights = np.exp(logweights - logweights.max())
                    weights /= weights.sum()
                    fits = np.stack([posterior(x[i, :counts[i]], y[i, :counts[i]], mu, 20.0)[0]
                                     for mu in proposals])
                    estimates[i] = weights @ fits
                    selected_old += weights[0]
            err = np.einsum('nd,md->nm', estimates - truth, test_phi, optimize=False)
            node_error = np.mean(err**2, axis=1)
            mask = np.zeros(nodes, dtype=bool)
            mask[changed_ids] = True
            rows.append({'seed': seed, 'changed': changed, 'change_type': change_type,
                         'method': method, 'target_budget': budget,
                         'source_samples': source_samples, 'mse': float(np.mean(node_error)),
                         'changed_mse': float(np.mean(node_error[mask])),
                         'unchanged_mse': float(np.mean(node_error[~mask])),
                         'old_selection_mass': float(selected_old / nodes) if method.startswith('source_') else ''})
    spec = {'seed': seed, 'nodes': nodes, 'changed': changed, 'change_type': change_type,
            'source_seed': 9173, 'source_samples': source_samples,
            'guard_margin': guard_margin,
            'source_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
            'changed_node_ids': changed_ids.tolist(),
            'source_revision': os.environ.get('ACE_SOURCE_REVISION', 'local')}
    return rows, spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--changed', type=int, choices=(1, 3, 10), required=True)
    p.add_argument('--change-type', choices=('family', 'coefficient'), required=True)
    p.add_argument('--guard-margin', type=float, default=None,
                   help='Include a source guarded by this passive SSE improvement threshold')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.guard_margin is not None and a.guard_margin < 0:
        p.error('--guard-margin must be nonnegative')
    rows, spec = experiment(a.seed, a.changed, a.change_type, guard_margin=a.guard_margin)
    a.output.mkdir(parents=True, exist_ok=True)
    metrics = a.output / 'metrics.csv'
    with metrics.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output / 'system.json').write_text(json.dumps(spec, indent=2, sort_keys=True) + '\n')
    receipt = {'schema_version': 2 if a.guard_margin is not None else 1,
               'kind': 'learned_transfer', 'rows': len(rows),
               'metrics_sha256': hashlib.sha256(metrics.read_bytes()).hexdigest(),
               'system_sha256': hashlib.sha256((a.output / 'system.json').read_bytes()).hexdigest()}
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'{a.change_type} k={a.changed} seed={a.seed}: {len(rows)} rows -> {a.output}')


if __name__ == '__main__':
    main()

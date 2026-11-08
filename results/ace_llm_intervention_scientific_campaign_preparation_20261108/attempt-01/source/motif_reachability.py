#!/usr/bin/env python3
"""Disjoint-motif SCM screen: vary graph size and hard-motif count separately.

Each motif has two roots and an observed interaction child. Extra graph nodes
are independent observed roots. Every acquired batch observes the full system;
interventions replace root values in one motif. This is a known-DAG numerical
screen, not a connected-graph or neural-learner experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


METHODS = ('random_single', 'coverage_single', 'random_pair', 'coverage_pair', 'risk_pair')
VALUES = (-2., 2.)


def candidates(motifs: int, pair: bool):
    if pair:
        return [(j, (0, 1), (a, b)) for j in range(motifs) for a in VALUES for b in VALUES]
    return [(j, (axis,), (v,)) for j in range(motifs) for axis in (0, 1) for v in VALUES]


def features(x):
    return np.stack((x[..., 0], x[..., 1], x[..., 0] * x[..., 1]), axis=-1)


def experiment(seed: int, nodes: int, motifs: int, background_sd: float,
               penalty: int, budget: int = 400, batch: int = 8):
    if nodes < 3 * motifs or motifs < 1 or background_sd < 0 or penalty < 0 or budget < batch:
        raise ValueError('Invalid SCM size, motif count, variation, penalty, or budget')
    rng_truth = np.random.default_rng(seed + 71221)
    truth = np.column_stack((rng_truth.normal(.45, .08, motifs),
                             rng_truth.normal(-.3, .08, motifs),
                             rng_truth.choice((-1., 1.), motifs) * rng_truth.uniform(.7, 1.1, motifs)))
    # The serialized edges specify an actual, though disconnected, DAG.
    system = {'seed': seed, 'nodes': nodes, 'motifs': motifs, 'background_sd': background_sd,
              'edges': [[3*j, 3*j+2] for j in range(motifs)] +
                       [[3*j+1, 3*j+2] for j in range(motifs)],
              'coefficients': truth.tolist(), 'root_sd': background_sd,
              'child_noise_sd': .15, 'source_revision': os.environ.get('ACE_SOURCE_REVISION', 'local')}
    q = np.diag((4/3, 4/3, 16/9))
    rows = []
    for method_i, method in enumerate(METHODS):
        rng = np.random.default_rng(seed * 101 + method_i + 19)
        cov = np.repeat(np.eye(3)[None, :, :], motifs, axis=0)
        mean = np.zeros((motifs, 3))
        menu = candidates(motifs, method.endswith('pair'))
        unit_cost = 1 + penalty * len(menu[0][1])
        spent = steps = 0
        while spent + batch * unit_cost <= budget:
            if method.startswith('random'):
                action = menu[int(rng.integers(len(menu)))]
            elif method.startswith('coverage'):
                per_motif = len(menu) // motifs
                action = menu[(steps % motifs) * per_motif + (steps // motifs) % per_motif]
            else:
                # A common Monte Carlo candidate context set reduces scoring noise.
                base = rng.normal(0, background_sd, (128, motifs, 2))
                scores = []
                for j, axes, vals in menu:
                    x = base[:, j, :].copy()
                    for axis, value in zip(axes, vals):
                        x[:, axis] = value
                    z = features(x)
                    next_cov = np.linalg.inv(np.linalg.inv(cov[j]) +
                                             batch * (z.T @ z / len(z)) / .15**2)
                    scores.append(float(np.trace(q @ (cov[j] - next_cov))))
                action = menu[int(np.argmax(scores))]
            j, axes, vals = action
            x = rng.normal(0, background_sd, (batch, motifs, 2))
            for axis, value in zip(axes, vals):
                x[:, j, axis] = value
            z = features(x)
            y = np.einsum('bjd,jd->bj', z, truth, optimize=False) + rng.normal(0, .15, (batch, motifs))
            for k in range(motifs):
                precision = np.linalg.inv(cov[k])
                cov[k] = np.linalg.inv(precision + z[:, k].T @ z[:, k] / .15**2)
                mean[k] = cov[k] @ (precision @ mean[k] + z[:, k].T @ y[:, k] / .15**2)
            spent += batch * unit_cost
            steps += 1
        error = mean - truth
        risk = np.einsum('mi,ij,mj->m', error, q, error, optimize=False)
        rows.append({'seed': seed, 'nodes': nodes, 'motifs': motifs, 'method': method,
                     'background_sd': background_sd, 'penalty': penalty, 'budget': budget,
                     'cost_spent': spent, 'samples': steps * batch,
                     'actuator_uses': steps * batch * len(menu[0][1]), 'steps': steps,
                     'motif_mse': float(np.mean(risk)),
                     'interaction_mse': float(np.mean(error[:, 2]**2)),
                     'posterior_risk': float(np.mean([np.trace(q @ c) for c in cov]))})
    return rows, system


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--nodes', required=True, type=int)
    p.add_argument('--motifs', required=True, type=int)
    p.add_argument('--background-sd', required=True, type=float)
    p.add_argument('--penalty', required=True, type=int, choices=(0, 1, 4))
    p.add_argument('--budget', type=int, default=400)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    rows, system = experiment(a.seed, a.nodes, a.motifs, a.background_sd,
                              a.penalty, a.budget)
    a.output.mkdir(parents=True, exist_ok=True)
    metrics, spec = a.output / 'metrics.csv', a.output / 'system.json'
    with metrics.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    spec.write_text(json.dumps(system, indent=2, sort_keys=True) + '\n')
    (a.output / 'complete.json').write_text(json.dumps({
        'schema_version': 1, 'kind': 'motif_reachability', 'rows': len(rows),
        'metrics_sha256': hashlib.sha256(metrics.read_bytes()).hexdigest(),
        'system_sha256': hashlib.sha256(spec.read_bytes()).hexdigest()}, indent=2) + '\n')
    print(f'seed={a.seed} nodes={a.nodes} motifs={a.motifs}: {len(rows)} arms')


if __name__ == '__main__':
    main()

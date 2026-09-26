#!/usr/bin/env python3
"""Matched exact-posterior acquisition on the connected-motif SCM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from connected_motif import action_menu, make_system, sample


METHODS = ('random_single', 'coverage_single', 'random_pair', 'coverage_pair', 'risk_pair')
Q = np.diag((4/3, 4/3, 16/9))


class SealedEvaluator:
    def __init__(self, system, seed):
        rng = np.random.default_rng(seed + 77153)
        padding_rng = np.random.default_rng(seed + 77154)
        # An independent, fixed, feasible panel; never passed to a policy.
        self.panel = []
        for action in action_menu(system, True):
            _, phi, mask = sample(system, rng, 64, action, padding_rng=padding_rng)
            self.panel.append((phi, mask))
        self.truth = system.coefficients.copy()

    def evaluate(self, mean):
        error = mean - self.truth
        broad = float(np.mean(np.einsum('mi,ij,mj->m', error, Q, error, optimize=False)))
        feasible = []
        for phi, mask in self.panel:
            # Only natural child outputs enter the feasible comparison.
            pred_error = np.einsum('bmd,md->bm', phi, error, optimize=False)
            feasible.extend((pred_error[mask] ** 2).tolist())
        return broad, float(np.mean(feasible))


def choose(method, public_system, rng, padding_rng, mean, cov, step, batch):
    menu = action_menu(public_system, method.endswith('pair'))
    if method.startswith('random'):
        return menu[int(rng.integers(len(menu)))]
    if method.startswith('coverage'):
        per_motif = len(menu) // public_system.motifs
        return menu[(step % public_system.motifs) * per_motif +
                    (step // public_system.motifs) % per_motif]
    scores = []
    for action in menu:
        # Student-predictive contexts only: public_system contains zero truth.
        _, phi, mask = sample(public_system, rng, 64, action, coefficients=mean,
                              padding_rng=padding_rng)
        score = 0.
        for j in range(public_system.motifs):
            z = phi[mask[:, j], j]
            if not len(z):
                continue
            moment = z.T @ z / len(z)
            next_cov = np.linalg.inv(np.linalg.inv(cov[j]) + batch * moment / .15**2)
            score += float(np.trace(Q @ (cov[j] - next_cov)))
        scores.append(score)
    return menu[int(np.argmax(scores))]


def experiment(seed, nodes, motifs, root_sd, penalty, budget=400, batch=8):
    if penalty < 0 or budget < batch:
        raise ValueError('Invalid cost parameters')
    system = make_system(seed, nodes, motifs, root_sd)
    public = replace(system, coefficients=np.zeros_like(system.coefficients))
    evaluator = SealedEvaluator(system, seed)
    rows, actions = [], []
    for mi, method in enumerate(METHODS):
        rng = np.random.default_rng(seed * 113 + mi + 31)
        padding_rng = np.random.default_rng(seed * 113 + mi + 80031)
        mean = np.zeros((motifs, 3))
        cov = np.repeat(np.eye(3)[None, :, :], motifs, axis=0)
        spent = samples = actuators = masked = step = 0
        unit_cost = 1 + penalty * (2 if method.endswith('pair') else 1)
        while spent + batch * unit_cost <= budget:
            action = choose(method, public, rng, padding_rng, mean, cov, step, batch)
            values, phi, natural = sample(system, rng, batch, action,
                                          padding_rng=padding_rng)
            for j, child in enumerate(system.children):
                z = phi[natural[:, j], j]
                y = values[natural[:, j], child]
                if not len(z):
                    continue
                precision = np.linalg.inv(cov[j])
                cov[j] = np.linalg.inv(precision + z.T @ z / system.child_sd**2)
                mean[j] = cov[j] @ (precision @ mean[j] + z.T @ y / system.child_sd**2)
            masked += int(np.size(natural) - natural.sum())
            spent += batch * unit_cost
            samples += batch
            actuators += batch * len(action[1])
            actions.append({'method': method, 'step': step, 'motif': action[0],
                            'targets': ','.join(map(str, action[1])),
                            'levels': ','.join(map(str, action[2])),
                            'natural_child_labels': int(natural.sum()),
                            'masked_child_labels': int(np.size(natural) - natural.sum()),
                            'cumulative_cost': spent, 'cumulative_samples': samples})
            step += 1
        broad, feasible = evaluator.evaluate(mean)
        rows.append({'seed': seed, 'nodes': nodes, 'motifs': motifs, 'method': method,
                     'root_sd': root_sd, 'penalty': penalty, 'budget': budget,
                     'cost_spent': spent, 'samples': samples, 'actuator_uses': actuators,
                     'masked_child_labels': masked, 'steps': step,
                     'broad_motif_mse': broad, 'feasible_motif_mse': feasible,
                     'posterior_risk': float(np.mean([np.trace(Q @ c) for c in cov]))})
    spec = {'seed': seed, 'nodes': nodes, 'motifs': motifs, 'root_sd': root_sd,
            'penalty': penalty, 'budget': budget, 'batch': batch,
            'rng_schema': 'separate_padding_v1',
            'parents': system.parents, 'children': system.children,
            'edges': system.edges, 'coefficients': system.coefficients.tolist(),
            'source_revision': os.environ.get('ACE_SOURCE_REVISION', 'local')}
    return rows, actions, spec


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--nodes', required=True, type=int)
    p.add_argument('--motifs', required=True, type=int)
    p.add_argument('--root-sd', required=True, type=float)
    p.add_argument('--penalty', required=True, type=int, choices=(0, 1, 4))
    p.add_argument('--budget', type=int, default=400)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    rows, actions, spec = experiment(a.seed, a.nodes, a.motifs, a.root_sd,
                                     a.penalty, a.budget)
    a.output.mkdir(parents=True, exist_ok=True)
    metric_file, action_file, system_file = (a.output / x for x in
                                              ('metrics.csv', 'actions.csv', 'system.json'))
    write_csv(metric_file, rows)
    write_csv(action_file, actions)
    system_file.write_text(json.dumps(spec, indent=2, sort_keys=True) + '\n')
    receipt = {'schema_version': 2, 'kind': 'connected_acquisition', 'rows': len(rows),
               'actions': len(actions), 'source_revision': spec['source_revision'],
               'metrics_sha256': hashlib.sha256(metric_file.read_bytes()).hexdigest(),
               'actions_sha256': hashlib.sha256(action_file.read_bytes()).hexdigest(),
               'system_sha256': hashlib.sha256(system_file.read_bytes()).hexdigest()}
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'{a.seed} N={a.nodes} k={a.motifs}: {len(rows)} arms, {len(actions)} actions')


if __name__ == '__main__':
    main()

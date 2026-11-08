#!/usr/bin/env python3
"""Matched-menu numerical screen for interaction identification (no model API)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SINGLE = [((i,), (v,)) for i in range(2) for v in (-2., 2.)]
PAIR = [((0, 1), (a, b)) for a in (-2., 2.) for b in (-2., 2.)]
METHODS = ('random_single', 'ivr_single', 'random_pair', 'coverage_pair', 'ivr_pair')


def context(action, rng, n, background_sd):
    x = rng.normal(0, background_sd, (n, 2))
    for j, val in zip(*action):
        x[:, j] = val
    return x


def phi(x):
    return np.column_stack((x[:, 0], x[:, 1], x[:, 0] * x[:, 1]))


def experiment(seed: int, budget: int, background_sd: float, penalty: int,
               batch: int = 8) -> list[dict]:
    if budget < batch or background_sd < 0 or penalty < 0:
        raise ValueError('Invalid budget, background variation, or penalty')
    truth_rng = np.random.default_rng(seed + 7169)
    truth = np.array([.45, -.3, truth_rng.choice((-1., 1.)) * truth_rng.uniform(.7, 1.1)])
    q = np.diag([4 / 3, 4 / 3, 16 / 9])
    rows = []
    for method_i, method in enumerate(METHODS):
        rng = np.random.default_rng(seed * 101 + method_i + 17)
        cov, mean = np.eye(3), np.zeros(3)
        spent = samples = actuator_uses = steps = 0
        menu = SINGLE if method.endswith('single') else PAIR
        unit_cost = 1 + penalty * len(menu[0][0])
        while spent + batch * unit_cost <= budget:
            if method.startswith('random'):
                action = menu[int(rng.integers(len(menu)))]
            elif method == 'coverage_pair':
                action = menu[steps % len(menu)]
            else:
                # Expected covariance contraction under each permissible action;
                # no simulator outcome or truth enters this selection.
                scores = []
                precision = np.linalg.inv(cov)
                for action_candidate in menu:
                    z = phi(context(action_candidate, rng, 128, background_sd))
                    next_cov = np.linalg.inv(precision + batch * (z.T @ z / len(z)) / .15**2)
                    scores.append(float(np.trace(q @ (cov - next_cov))))
                action = menu[int(np.argmax(scores))]
            z = phi(context(action, rng, batch, background_sd))
            y = z @ truth + rng.normal(0, .15, batch)
            precision = np.linalg.inv(cov)
            cov = np.linalg.inv(precision + z.T @ z / .15**2)
            mean = cov @ (precision @ mean + z.T @ y / .15**2)
            spent += batch * unit_cost
            samples += batch
            actuator_uses += batch * len(action[0])
            steps += 1
        error = mean - truth
        rows.append({'seed': seed, 'method': method, 'background_sd': background_sd,
                     'penalty': penalty, 'budget': budget, 'cost_spent': spent,
                     'samples': samples, 'actuator_uses': actuator_uses, 'steps': steps,
                     'mse': float(error @ q @ error),
                     'interaction_error': float(error[2] ** 2),
                     'posterior_risk': float(np.trace(q @ cov))})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--budget', type=int, default=400)
    p.add_argument('--background-sd', type=float, required=True)
    p.add_argument('--penalty', type=int, choices=(0, 1, 4), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = experiment(a.seed, a.budget, a.background_sd, a.penalty)
    a.output.mkdir(parents=True, exist_ok=True)
    metrics = a.output / 'metrics.csv'
    with metrics.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {'schema_version': 1, 'kind': 'design_b2', 'rows': len(rows),
               'source_revision': os.environ.get('ACE_SOURCE_REVISION', 'local'),
               'metrics_sha256': hashlib.sha256(metrics.read_bytes()).hexdigest()}
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'{a.seed} bg={a.background_sd} penalty={a.penalty}: {len(rows)} rows')


if __name__ == '__main__':
    main()

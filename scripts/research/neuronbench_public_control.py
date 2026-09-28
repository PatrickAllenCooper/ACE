#!/usr/bin/env python3
"""Public-only action and forecast controls for NeuronBench.

This module imports no benchmark code. It reads serialized public protocols,
observations, and forecast specifications; truth remains in a separate scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(segments) -> np.ndarray:
    durations = np.array([float(pair[0]) for pair in segments])
    amps = np.array([float(pair[1]) for pair in segments])
    total = durations.sum()
    if total <= 0 or not np.isfinite(durations).all() or not np.isfinite(amps).all():
        raise ValueError('invalid public waveform')
    return np.array([total / 400, float(amps.min()) / 30,
                     float(amps.max()) / 30, float(durations @ amps / total) / 30,
                     float(durations[amps < 0].sum()) / 400,
                     float(durations[amps > 0].sum()) / 400,
                     len(segments) / 3, float(np.any(amps < 0))], dtype=float)


def matrix(protocols) -> np.ndarray:
    return np.stack([features(segments) for _, segments in protocols])


def choose(public_problem: dict, method: str, budget: int, seed: int) -> list[int]:
    pool = public_problem['protocols']
    test = public_problem['forecast_protocols']
    if not 0 < budget <= len(pool):
        raise ValueError('invalid budget')
    if method == 'random':
        return np.random.default_rng(seed).choice(len(pool), budget, replace=False).tolist()
    if method != 'coverage':
        raise ValueError('unknown method')
    candidates = matrix(pool)
    target = matrix(test)
    scale = np.std(np.vstack([candidates, target]), axis=0)
    scale[scale < .05] = 1
    distances = np.sum(((target[:, None, :] - candidates[None, :, :]) / scale) ** 2,
                       axis=2)
    chosen = []
    for _ in range(budget):
        best = min((j for j in range(len(pool)) if j not in chosen),
                   key=lambda j: (float(np.minimum(
                       distances[:, chosen].min(axis=1) if chosen else np.inf,
                       distances[:, j]).mean()), j))
        chosen.append(best)
    return chosen


def plan(problem_path: Path, method: str, budget: int, seed: int, output: Path):
    problem = json.loads(problem_path.read_text())
    indices = choose(problem, method, budget, seed)
    actions = [problem['protocols'][i] for i in indices]
    if len({label for label, _ in actions}) != budget:
        raise ValueError('duplicate action')
    report = {'method': method, 'selection_seed': seed, 'budget': budget,
              'problem_sha256': digest(problem_path), 'indices': indices,
              'actions': actions, 'closed_model_calls': 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(method, [label for label, _ in actions])


def forecast(public_dir: Path, output: Path, ridge: float):
    problem_path = public_dir / 'problem.json'
    problem = json.loads(problem_path.read_text())
    observations = json.loads((public_dir / 'observations.json').read_text())
    lookup = {label: seg for label, seg in problem['protocols']}
    labels = [row['protocol_label'] for row in observations]
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError('missing or duplicate observations')
    if any(label not in lookup or row['cost'] != 1 for label, row in zip(labels, observations)):
        raise ValueError('invalid action or cost')
    train = matrix([(label, lookup[label]) for label in labels])
    test = matrix(problem['forecast_protocols'])
    center = np.vstack([train, test]).mean(axis=0)
    scale = np.vstack([train, test]).std(axis=0)
    scale[scale < .05] = 1
    x = (train - center) / scale
    z = (test - center) / scale
    y = np.array([float(row['spike_count']) for row in observations])
    if not np.isfinite(y).all() or ridge <= 0:
        raise ValueError('invalid observations or penalty')
    x = np.column_stack([np.ones(len(x)), x])
    z = np.column_stack([np.ones(len(z)), z])
    regularizer = np.eye(x.shape[1]) * ridge
    regularizer[0, 0] = 0
    coefficients = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
    prediction = np.clip(z @ coefficients, 0, None)
    result = {'predicted_spike_counts': {label: float(value) for
                 (label, _), value in zip(problem['forecast_protocols'], prediction)},
              'observations_used': len(observations),
              'environment_samples_used': int(sum(row['cost'] for row in observations)),
              'ridge_penalty': ridge, 'problem_sha256': digest(problem_path),
              'observations_sha256': digest(public_dir / 'observations.json'),
              'private_files_read': 0, 'closed_model_calls': 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('forecasts', len(prediction), 'from', len(observations), 'public observations')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    choose_p = sub.add_parser('plan')
    choose_p.add_argument('--problem', type=Path, required=True)
    choose_p.add_argument('--method', choices=('random', 'coverage'), required=True)
    choose_p.add_argument('--budget', type=int, required=True)
    choose_p.add_argument('--seed', type=int, required=True)
    choose_p.add_argument('--output', type=Path, required=True)
    forecast_p = sub.add_parser('forecast')
    forecast_p.add_argument('--public', type=Path, required=True)
    forecast_p.add_argument('--ridge', type=float, required=True)
    forecast_p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'plan':
        plan(a.problem, a.method, a.budget, a.seed, a.output)
    else:
        forecast(a.public, a.output, a.ridge)


if __name__ == '__main__':
    main()

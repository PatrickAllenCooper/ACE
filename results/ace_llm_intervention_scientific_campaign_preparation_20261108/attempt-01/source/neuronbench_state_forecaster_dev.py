#!/usr/bin/env python3
"""Public-only forced integrate-and-fire development gate for NeuronBench."""
from __future__ import annotations

import argparse
import hashlib
import json
import itertools
from pathlib import Path

import numpy as np

from neuronbench_public_control import matrix
from neuronbench_public_timing import current_and_test_start


DT = 0.1  # milliseconds per archived voltage sample
TAU_ADAPT = 100.0
TAU_RECOVERY = 100.0
REFRACTORY_SAMPLES = 20  # 2 ms
THRESHOLDS = (-45.0, -40.0, -35.0, -30.0, -25.0)
RIDGE = 0.1


def bounded_linear_fit(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Enumerate the 16 active sets of this four-coefficient NNLS problem."""
    best, best_sse = np.zeros(matrix.shape[1]), float(np.sum(target ** 2))
    for size in range(1, matrix.shape[1] + 1):
        for active in itertools.combinations(range(matrix.shape[1]), size):
            candidate = np.linalg.lstsq(matrix[:, active], target, rcond=None)[0]
            if np.any(candidate < 0):
                continue
            residual = target - np.einsum('ij,j->i', matrix[:, active], candidate)
            sse = float(np.sum(residual ** 2))
            if sse < best_sse:
                best_sse = sse
                best = np.zeros(matrix.shape[1])
                best[list(active)] = candidate
    return best


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def traces(public: Path):
    problem = json.loads((public / 'problem.json').read_text())
    rows = json.loads((public / 'observations.json').read_text())
    lookup = dict(problem['protocols'])
    if len(rows) != 4 or len({r['protocol_label'] for r in rows}) != 4:
        raise ValueError('expected four distinct observed protocols')
    result = []
    for row in rows:
        current, start = current_and_test_start(lookup[row['protocol_label']])
        with np.load(public / row['trace_file']) as data:
            v = np.asarray(data['voltage'], dtype=float)
            idx = np.asarray(data['obs_idx'], dtype=int)
        if len(v) != len(current) or not np.isfinite(v).all() or start != row['test_start']:
            raise ValueError('public trace/stimulus mismatch')
        if not np.array_equal(np.diff(idx), np.full(len(idx) - 1, 10)):
            raise ValueError('invalid recorded-index stride')
        result.append((row['protocol_label'], current, start, v, float(row['spike_count'])))
    return result


def state_features(current: np.ndarray, voltage: np.ndarray):
    spike = np.r_[False, (voltage[:-1] < 0) & (voltage[1:] >= 0)]
    a = np.zeros(len(voltage))
    h = np.zeros(len(voltage))
    decay_a, decay_h = np.exp(-DT / TAU_ADAPT), np.exp(-DT / TAU_RECOVERY)
    for t in range(1, len(voltage)):
        a[t] = a[t - 1] * decay_a + float(spike[t])
        h[t] = h[t - 1] * decay_h + DT * max(-current[t - 1], 0)
    # Exclude spike upstroke/reset and immediate surrounding samples.
    keep = (voltage[:-1] < -35) & (voltage[1:] < -35)
    for i in np.flatnonzero(spike):
        keep[max(i - 20, 0):min(i + 30, len(keep))] = False
    x = np.column_stack((-voltage[:-1], current[:-1], -a[:-1], h[:-1]))
    derivative = np.diff(voltage) / DT
    return x[keep], derivative[keep]


def fit(rows):
    features, derivatives = [], []
    for _, current, _, voltage, _ in rows:
        x, y = state_features(current, voltage)
        features.append(x)
        derivatives.append(y)
    x = np.vstack(features)
    y = np.concatenate(derivatives)
    if len(y) < 100 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('insufficient valid subthreshold samples')
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1
    z = (x - center) / scale
    augmented = np.vstack((z, np.sqrt(RIDGE) * np.eye(z.shape[1])))
    target = np.r_[y - y.mean(), np.zeros(z.shape[1])]
    beta = bounded_linear_fit(augmented, target) / scale
    intercept = float(y.mean() - np.sum(center * beta))
    resting = float(np.median(np.concatenate([r[3][:100] for r in rows])))
    return {'intercept': intercept, 'coefficients': beta.tolist(),
            'initial_voltage': resting, 'subthreshold_samples': len(y),
            'subthreshold_derivative_rmse': float(np.sqrt(np.mean((
                intercept + np.einsum('ij,j->i', x, beta) - y) ** 2)))}


def simulate(current: np.ndarray, start: int, model: dict, threshold: float) -> int:
    b0, (bv, bi, ba, bh) = model['intercept'], model['coefficients']
    v, a, h = model['initial_voltage'], 0.0, 0.0
    decay_a, decay_h = np.exp(-DT / TAU_ADAPT), np.exp(-DT / TAU_RECOVERY)
    count = refractory = 0
    for t, drive in enumerate(current):
        if refractory:
            refractory -= 1
            v = model['initial_voltage']
        else:
            v += DT * (b0 - bv * v + bi * drive - ba * a + bh * h)
            if v >= threshold:
                if t >= start:
                    count += 1
                v = model['initial_voltage']
                a += 1
                refractory = REFRACTORY_SAMPLES
        a *= decay_a
        h = h * decay_h + DT * max(-drive, 0)
    return count


def select_threshold(rows, model):
    errors = []
    for threshold in THRESHOLDS:
        error = sum((simulate(current, start, model, threshold) - count) ** 2
                    for _, current, start, _, count in rows)
        errors.append(error)
    return THRESHOLDS[int(np.argmin(errors))], float(min(errors))


def public_cv(public: Path, output: Path) -> None:
    rows = traces(public)
    problem = json.loads((public / 'problem.json').read_text())
    lookup = dict(problem['protocols'])
    descriptors = matrix([(r[0], lookup[r[0]]) for r in rows])
    counts = np.array([r[4] for r in rows])
    scale = descriptors.std(axis=0)
    scale[scale < .05] = 1
    center = descriptors.mean(axis=0)
    folds = []
    for i, held in enumerate(rows):
        training = [r for j, r in enumerate(rows) if j != i]
        model = fit(training)
        threshold, train_sse = select_threshold(training, model)
        forecast = simulate(held[1], held[2], model, threshold)
        keep = np.arange(4) != i
        x, y, z = descriptors[keep], counts[keep], descriptors[i]
        nearest = float(y[np.argmin(np.sum(((x - z) / scale) ** 2, axis=1))])
        transformed = np.column_stack((np.ones(len(x)), (x - center) / scale))
        point = np.r_[1., (z - center) / scale]
        penalty = np.eye(transformed.shape[1])
        penalty[0, 0] = 0
        coef = np.linalg.solve(transformed.T @ transformed + penalty,
                               transformed.T @ y)
        ridge = max(float(np.sum(point * coef)), 0.)
        folds.append({'heldout_label': held[0], 'actual_public_count': held[4],
                      'predicted_count': forecast, 'threshold': threshold,
                      'mean_prediction': float(y.mean()),
                      'nearest_prediction': nearest, 'ridge_prediction': ridge,
                      'training_count_sse': train_sse,
                      'subthreshold_derivative_rmse': model['subthreshold_derivative_rmse'],
                      'subthreshold_samples': model['subthreshold_samples']})
    result = {'method': 'forced_integrate_and_fire_dev_v1',
              'fixed_parameters': {'dt_ms': DT, 'tau_adapt_ms': TAU_ADAPT,
                                   'tau_recovery_ms': TAU_RECOVERY,
                                   'refractory_samples': REFRACTORY_SAMPLES,
                                   'threshold_grid': THRESHOLDS, 'ridge': RIDGE},
              'public_problem_sha256': sha(public / 'problem.json'),
              'public_observations_sha256': sha(public / 'observations.json'),
              'public_trace_sha256': {r['trace_file']: sha(public / r['trace_file'])
                                      for r in json.loads((public / 'observations.json').read_text())},
              'public_observations': 4, 'new_oracle_actions': 0,
              'private_files_read': 0, 'closed_model_calls': 0,
              'folds': folds,
              'public_loo_mae': float(np.mean([abs(f['predicted_count'] - f['actual_public_count'])
                                              for f in folds])),
              'baseline_public_loo_mae': {name: float(np.mean([
                  abs(f[name + '_prediction'] - f['actual_public_count']) for f in folds]))
                  for name in ('mean', 'nearest', 'ridge')}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(public, result['public_loo_mae'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    public_cv(args.public, args.output)

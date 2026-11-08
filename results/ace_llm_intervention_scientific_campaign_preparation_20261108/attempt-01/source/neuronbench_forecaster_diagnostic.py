#!/usr/bin/env python3
"""Retrospective public-only forecasters and separate private scorer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from neuronbench_public_control import matrix


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(public: Path, output: Path, method: str) -> None:
    problem_path = public / 'problem.json'
    observations_path = public / 'observations.json'
    problem = json.loads(problem_path.read_text())
    observations = json.loads(observations_path.read_text())
    lookup = {label: segments for label, segments in problem['protocols']}
    labels = [row['protocol_label'] for row in observations]
    if len(labels) != len(set(labels)) or len(labels) != 4 or any(
            row['cost'] != 1 or label not in lookup for row, label in zip(observations, labels)):
        raise ValueError('invalid acquired public observations')
    if set(labels) & set(problem['test_protocol_labels']):
        raise ValueError('acquisition/forecast overlap')
    train = matrix([(label, lookup[label]) for label in labels])
    test = matrix(problem['forecast_protocols'])
    y = np.array([float(row['spike_count']) for row in observations])
    if not np.isfinite(y).all() or len(test) != 6:
        raise ValueError('invalid targets or public forecasts')
    if method == 'mean':
        values = np.full(len(test), y.mean())
    elif method == 'nearest':
        scale = np.vstack([train, test]).std(axis=0)
        scale[scale < .05] = 1
        distances = np.sum(((test[:, None, :] - train[None, :, :]) / scale) ** 2, axis=2)
        values = y[np.argmin(distances, axis=1)]
    else:
        raise ValueError('unknown method')
    values = np.clip(values, 0, None)
    report = {'method': method, 'predicted_spike_counts': {
                  label: float(value) for (label, _), value in zip(
                      problem['forecast_protocols'], values)},
              'problem_sha256': digest(problem_path),
              'observations_sha256': digest(observations_path),
              'observations_used': 4, 'environment_samples_used': 4,
              'new_oracle_actions': 0, 'private_files_read': 0,
              'closed_model_calls': 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')


def score(public: Path, targets_path: Path, predictions_path: Path,
          output: Path) -> None:
    problem = json.loads((public / 'problem.json').read_text())
    targets = json.loads(targets_path.read_text())
    predictions = json.loads(predictions_path.read_text())
    values = predictions['predicted_spike_counts']
    labels = problem['test_protocol_labels']
    if set(values) != set(targets) or set(values) != set(labels) or len(labels) != 6:
        raise ValueError('incomplete forecast target set')
    if predictions['problem_sha256'] != digest(public / 'problem.json') or predictions[
            'observations_sha256'] != digest(public / 'observations.json'):
        raise ValueError('public input provenance mismatch')
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError('nonfinite prediction')
    mse = max(sum((values[label] - targets[label]) ** 2 for label in labels) / 6 - .25, 0)
    report = {'floored_spike_forecast_mse': mse, 'floor': .25,
              'forecast_labels': 6, 'new_oracle_actions': 0,
              'predictions_sha256': digest(predictions_path),
              'private_targets_sha256': digest(targets_path)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    pp = sub.add_parser('predict')
    pp.add_argument('--public', type=Path, required=True)
    pp.add_argument('--method', choices=('mean', 'nearest'), required=True)
    pp.add_argument('--output', type=Path, required=True)
    sp = sub.add_parser('score')
    sp.add_argument('--public', type=Path, required=True)
    sp.add_argument('--targets', type=Path, required=True)
    sp.add_argument('--predictions', type=Path, required=True)
    sp.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'predict':
        predict(a.public, a.output, a.method)
    else:
        score(a.public, a.targets, a.predictions, a.output)


if __name__ == '__main__':
    main()

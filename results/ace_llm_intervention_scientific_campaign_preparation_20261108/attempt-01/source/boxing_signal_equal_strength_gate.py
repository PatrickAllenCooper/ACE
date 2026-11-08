#!/usr/bin/env python3
"""Public-only test of the equal-strength cue in BoxingGym Signal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from fit_boxing_signal_fixed_data import kernel, read_coordinates
from fit_boxing_signal_prior_gate import centers


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(x: np.ndarray, y: np.ndarray, width: float, equal: bool):
    loc = centers(x, y, 3)
    k = kernel(x, loc, width)
    basis = np.column_stack((np.ones(len(x)), k.sum(axis=1) if equal else k))
    weights = nnls(basis, y)[0]
    return loc, weights


def predict(x: np.ndarray, loc: np.ndarray, weights: np.ndarray,
            width: float, equal: bool) -> np.ndarray:
    k = kernel(x, loc, width)
    basis = np.column_stack((np.ones(len(x)), k.sum(axis=1) if equal else k))
    return np.maximum(basis @ weights, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.protocol.read_text())
    x, y = read_coordinates(args.public / 'observations.csv', True)
    assert y is not None
    if len(x) != spec['public_queries']:
        raise ValueError('unexpected number of public observations')
    train, val = np.array(spec['fit_indices']), np.array(spec['validation_indices'])
    if sorted(np.r_[train, val].tolist()) != list(range(len(x))):
        raise ValueError('invalid public split')
    result = {}
    for equal, name in ((True, 'equal_amplitude'), (False, 'free_amplitudes')):
        loo = {}
        for width in spec['widths']:
            errors = []
            for held in train:
                keep = train[train != held]
                loc, weights = fit(x[keep], y[keep], width, equal)
                errors.append(float((predict(x[[held]], loc, weights, width, equal)[0] - y[held]) ** 2))
            loo[width] = float(np.mean(errors))
        width = min(spec['widths'], key=lambda w: loo[w])
        loc, weights = fit(x[train], y[train], width, equal)
        pred = predict(x[val], loc, weights, width, equal)
        result[name] = {'width': width, 'fit_loo_mse': loo[width],
                        'validation_mae': float(np.mean(np.abs(pred - y[val]))),
                        'validation_predictions': pred.tolist(),
                        'centers': loc.tolist(), 'weights': weights.tolist()}
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / 'public_diagnostic.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    receipt = {'protocol_sha256': digest(args.protocol),
               'observations_sha256': digest(args.public / 'observations.csv'),
               'public_observations': len(x), 'validation_observations': len(val),
               'private_files_read': 0, 'model_calls': 0,
               'diagnostic_sha256': digest(output)}
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({name: info['validation_mae'] for name, info in result.items()}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Public-only fixed-data numerical controls for BoxingGym signal localization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, nnls


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_coordinates(path: Path, responses: bool) -> tuple[np.ndarray, np.ndarray | None]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    x = np.array([[float(r['x1']), float(r['x2'])] for r in rows])
    y = np.array([float(r['response']) for r in rows]) if responses else None
    if not np.isfinite(x).all() or (y is not None and not np.isfinite(y).all()):
        raise ValueError('nonfinite public data')
    return x, y


def squared_distance(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.sum((x[:, None, :] - z[None, :, :]) ** 2, axis=2)


def kernel(x: np.ndarray, z: np.ndarray, width: float) -> np.ndarray:
    return np.exp(-squared_distance(x, z) / (2 * width ** 2))


def fit_rbf(x: np.ndarray, y: np.ndarray, width: float, penalty: float):
    center = float(y.mean())
    alpha = np.linalg.solve(kernel(x, x, width) + penalty * np.eye(len(x)), y - center)
    return center, alpha


def predict_rbf(train: np.ndarray, query: np.ndarray, center: float,
                alpha: np.ndarray, width: float) -> np.ndarray:
    return np.maximum(center + kernel(query, train, width) @ alpha, 0)


def source_centers(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    chosen = []
    for index in np.argsort(-y, kind='stable'):
        if not chosen or min(np.linalg.norm(x[index] - x[j]) for j in chosen) >= .3:
            chosen.append(int(index))
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        chosen.extend(int(i) for i in np.argsort(-y, kind='stable') if int(i) not in chosen)
    return x[chosen[:3]]


def gaussian_basis(x: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), kernel(x, centers, width)])


def fit_gaussian(x: np.ndarray, y: np.ndarray, width: float):
    centers = source_centers(x, y)
    weights = nnls(gaussian_basis(x, centers, width), y)[0]
    return centers, weights


def privileged_predict(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # The upstream equation and constants are privileged; the fitted centers
    # are inferred from public observations only.
    return .1 + np.sum(1.0 / (.0001 + squared_distance(x, centers)), axis=1)


def fit_privileged(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    first = source_centers(x, y)
    rng = np.random.default_rng(20260928)
    starts = [first] + [rng.uniform(-1.5, 1.5, (3, 2)) for _ in range(4)]
    fits = []
    for start in starts:
        result = least_squares(
            lambda flat: privileged_predict(x, flat.reshape(3, 2)) - y,
            start.ravel(), bounds=(-3 * np.ones(6), 3 * np.ones(6)),
            max_nfev=400)
        fits.append(result)
    best = min(fits, key=lambda fit: float(np.sum(fit.fun ** 2)))
    return best.x.reshape(3, 2), float(np.sum(best.fun ** 2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-revision', required=True)
    a = p.parse_args()
    spec = json.loads(a.protocol.read_text())
    x, y = read_coordinates(a.public / 'observations.csv', True)
    query, _ = read_coordinates(a.public / 'forecast_questions.csv', False)
    if len(x) != spec['training_queries_per_world'] or len(query) != spec[
            'heldout_queries_per_world']:
        raise ValueError('public data count mismatch')
    rbf_choices = [(w, lam) for w in (.2, .5, 1., 2.) for lam in (.01, .1, 1.)]
    def rbf_loo(choice):
        w, lam = choice
        errors = []
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            center, alpha = fit_rbf(x[keep], y[keep], w, lam)
            pred = predict_rbf(x[keep], x[[i]], center, alpha, w)[0]
            errors.append((pred - y[i]) ** 2)
        return float(np.mean(errors))
    width, penalty = min(rbf_choices, key=rbf_loo)
    center, alpha = fit_rbf(x, y, width, penalty)
    rbf = predict_rbf(x, query, center, alpha, width)
    def gaussian_loo(width):
        errors = []
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            centers, weights = fit_gaussian(x[keep], y[keep], width)
            pred = float((gaussian_basis(x[[i]], centers, width) @ weights)[0])
            errors.append((pred - y[i]) ** 2)
        return float(np.mean(errors))
    gwidth = min((.3, .6, 1.2), key=gaussian_loo)
    gcenters, gweights = fit_gaussian(x, y, gwidth)
    gaussian = np.maximum(gaussian_basis(query, gcenters, gwidth) @ gweights, 0)
    pcenters, training_sse = fit_privileged(x, y)
    privileged = privileged_predict(query, pcenters)
    predictions = {'rbf_kernel': rbf, 'gaussian_three_source': gaussian,
                   'privileged_inverse_quadratic': privileged}
    if any(len(value) != len(query) or not np.isfinite(value).all() for value in predictions.values()):
        raise ValueError('nonfinite forecast')
    a.output.mkdir(parents=True, exist_ok=True)
    forecast_path = a.output / 'predictions.csv'
    with forecast_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['index', 'x1', 'x2', *predictions])
        for i, xy in enumerate(query):
            writer.writerow([i, *xy, *(float(predictions[name][i]) for name in predictions)])
    model = {'rbf_kernel': {'width': width, 'penalty': penalty,
                            'public_loo_mse': rbf_loo((width, penalty))},
             'gaussian_three_source': {'width': gwidth,
                                       'centers': gcenters.tolist(),
                                       'weights': gweights.tolist(),
                                       'public_loo_mse': gaussian_loo(gwidth)},
             'privileged_inverse_quadratic': {'fitted_centers': pcenters.tolist(),
                                              'public_training_sse': training_sse}}
    model_path = a.output / 'models.json'
    model_path.write_text(json.dumps(model, indent=2) + '\n')
    receipt = {'source_revision': a.source_revision,
               'protocol_sha256': sha(a.protocol),
               'observations_sha256': sha(a.public / 'observations.csv'),
               'forecast_questions_sha256': sha(a.public / 'forecast_questions.csv'),
               'training_queries': len(x), 'forecast_questions': len(query),
               'private_files_read': 0, 'closed_model_calls': 0,
               'predictions_sha256': sha(forecast_path), 'models_sha256': sha(model_path)}
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('public-only fitted predictions', len(query), 'per', len(predictions), 'methods')


if __name__ == '__main__':
    main()

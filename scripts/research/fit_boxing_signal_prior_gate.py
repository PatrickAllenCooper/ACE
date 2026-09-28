#!/usr/bin/env python3
"""Public-only correct/wrong source-count proposal and validation gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from fit_boxing_signal_fixed_data import (kernel, read_coordinates, fit_rbf,
                                           predict_rbf)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def centers(x: np.ndarray, y: np.ndarray, count: int) -> np.ndarray:
    chosen = []
    for index in np.argsort(-y, kind='stable'):
        if not chosen or min(np.linalg.norm(x[index] - x[j]) for j in chosen) >= .3:
            chosen.append(int(index))
        if len(chosen) == count:
            break
    if len(chosen) < count:
        chosen.extend(int(i) for i in np.argsort(-y, kind='stable') if int(i) not in chosen)
    return x[chosen[:count]]


def source_fit(x: np.ndarray, y: np.ndarray, count: int, width: float):
    loc = centers(x, y, count)
    basis = np.column_stack([np.ones(len(x)), kernel(x, loc, width)])
    weights = nnls(basis, y)[0]
    return loc, weights


def source_predict(q: np.ndarray, loc: np.ndarray, weights: np.ndarray,
                   width: float) -> np.ndarray:
    basis = np.column_stack([np.ones(len(q)), kernel(q, loc, width)])
    return np.maximum(basis @ weights, 0)


def select_source_width(x: np.ndarray, y: np.ndarray, count: int) -> float:
    def loo(width):
        errors = []
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            loc, weights = source_fit(x[keep], y[keep], count, width)
            pred = source_predict(x[[i]], loc, weights, width)[0]
            errors.append((pred - y[i]) ** 2)
        return float(np.mean(errors))
    return min((.3, .6, 1.2), key=loo)


def select_rbf(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    def loo(choice):
        width, penalty = choice
        errors = []
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            center, alpha = fit_rbf(x[keep], y[keep], width, penalty)
            pred = predict_rbf(x[keep], x[[i]], center, alpha, width)[0]
            errors.append((pred - y[i]) ** 2)
        return float(np.mean(errors))
    return min(((w, p) for w in (.2, .5, 1., 2.) for p in (.01, .1, 1.)), key=loo)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-revision', required=True)
    a = p.parse_args()
    spec = json.loads(a.protocol.read_text())
    x, y = read_coordinates(a.public / 'observations.csv', True)
    q, _ = read_coordinates(a.public / 'forecast_questions.csv', False)
    if len(x) != spec['acquired_queries'] or len(q) != 32:
        raise ValueError('unexpected public input count')
    train = slice(0, 8)
    validation = slice(8, 16)
    width_rbf, penalty = select_rbf(x[train], y[train])
    width_1 = select_source_width(x[train], y[train], 1)
    width_3 = select_source_width(x[train], y[train], 3)
    base_center, base_alpha = fit_rbf(x[train], y[train], width_rbf, penalty)
    val_base = predict_rbf(x[train], x[validation], base_center,
                           base_alpha, width_rbf)
    val_errors = {'fallback_rbf': float(np.mean(np.abs(val_base - y[validation])))}
    selected = {}
    for count, width in ((1, width_1), (3, width_3)):
        loc, weights = source_fit(x[train], y[train], count, width)
        val = source_predict(x[validation], loc, weights, width)
        name = f'source_{count}'
        val_errors[name] = float(np.mean(np.abs(val - y[validation])))
        selected[name] = bool(val_errors[name] < val_errors['fallback_rbf'])
    # Validation labels have now made the gate decision. All fits below may
    # use the entire counted 16-query dataset, with hyperparameters fixed.
    base_center, base_alpha = fit_rbf(x, y, width_rbf, penalty)
    fallback = predict_rbf(x, q, base_center, base_alpha, width_rbf)
    loc1, weights1 = source_fit(x, y, 1, width_1)
    loc3, weights3 = source_fit(x, y, 3, width_3)
    wrong = source_predict(q, loc1, weights1, width_1)
    correct = source_predict(q, loc3, weights3, width_3)
    predictions = {'fallback_rbf': fallback, 'correct_3source': correct,
                   'wrong_1source': wrong,
                   'gate_correct': correct if selected['source_3'] else fallback,
                   'gate_wrong': wrong if selected['source_1'] else fallback}
    if any(len(v) != 32 or not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError('nonfinite public-only forecast')
    a.output.mkdir(parents=True, exist_ok=True)
    pred_path = a.output / 'predictions.csv'
    with pred_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['index', 'x1', 'x2', *predictions])
        for i, point in enumerate(q):
            writer.writerow([i, *point, *(float(predictions[name][i]) for name in predictions)])
    decision = {'train_public_indices': list(range(8)),
                'validation_public_indices': list(range(8, 16)),
                'validation_mae': val_errors,
                'selected_proposals': selected,
                'rbf_width': width_rbf, 'rbf_penalty': penalty,
                'source_1_width': width_1, 'source_3_width': width_3,
                'source_1_centers': loc1.tolist(),
                'source_3_centers': loc3.tolist(),
                'private_files_read': 0, 'closed_model_calls': 0}
    decision_path = a.output / 'decision.json'
    decision_path.write_text(json.dumps(decision, indent=2) + '\n')
    receipt = {'source_revision': a.source_revision,
               'protocol_sha256': sha(a.protocol),
               'observations_sha256': sha(a.public / 'observations.csv'),
               'forecast_questions_sha256': sha(a.public / 'forecast_questions.csv'),
               'training_queries': 16, 'heldout_questions': 32,
               'private_files_read': 0, 'closed_model_calls': 0,
               'predictions_sha256': sha(pred_path),
               'decision_sha256': sha(decision_path)}
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('public-only gate', selected, 'validation errors', val_errors)


if __name__ == '__main__':
    main()

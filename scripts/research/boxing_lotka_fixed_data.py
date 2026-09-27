#!/usr/bin/env python3
"""Fit fixed-data controls from public BoxingGym observations only.

The mechanistic arm is a privileged ceiling: its equation and initial state
come from the upstream simulator, never from an LM. No private files are read.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares


def read_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return np.array([float(r['time']) for r in rows]), np.array(
        [[float(r['response_1']), float(r['response_2'])] for r in rows])


def features(times, family, width):
    x = np.asarray(times) / 50.0
    if family == 'rbf':
        centers = np.linspace(0, 1, width)
        return np.column_stack([np.ones(len(x)),
                                np.exp(-0.5 * ((x[:, None] - centers) * width / 1.5) ** 2)])
    return np.column_stack([np.ones(len(x)), x, *[
        func(2 * np.pi * k * x) for k in range(1, width + 1)
        for func in (np.sin, np.cos)]])


def ridge_fit(x, y, penalty):
    regularizer = np.eye(x.shape[1]) * penalty
    regularizer[0, 0] = 0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


def choose_ridge(times, response, family):
    choices = [(w, p) for w in ((3, 5, 8) if family == 'rbf' else (1, 2, 3))
               for p in (0.01, 0.1, 1.0)]
    def loocv(choice):
        width, penalty = choice
        errors = []
        for i in range(len(times)):
            keep = np.arange(len(times)) != i
            beta = ridge_fit(features(times[keep], family, width), response[keep], penalty)
            pred = features(times[[i]], family, width) @ beta
            errors.append(np.mean((pred - response[[i]]) ** 2))
        return float(np.mean(errors))
    width, penalty = min(choices, key=loocv)
    beta = ridge_fit(features(times, family, width), response, penalty)
    return {'family': family, 'width': width, 'penalty': penalty,
            'coefficients': beta.tolist(), 'loocv_mse': loocv((width, penalty))}


def simulate(times, parameters):
    alpha, beta, gamma, delta = parameters
    def rhs(y, _time):
        prey, predator = y
        return [alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator]
    try:
        times = np.asarray(times)
        ordered = np.unique(np.r_[0, times])
        values = odeint(rhs, [40., 9.], ordered, mxstep=2000)
        return values[np.searchsorted(ordered, times)]
    except Exception:
        return np.full((len(times), 2), 1e6)


def fit_mechanistic(times, response):
    starts = ([0.1, 0.02, 0.4, 0.01], [0.07, 0.015, 0.3, 0.008],
              [0.14, 0.03, 0.5, 0.012])
    fits = []
    for start in starts:
        fit = least_squares(lambda p: ((simulate(times, p) - response) / [40, 10]).ravel(),
                            start, bounds=([0.01, 0.001, 0.05, 0.001],
                                           [0.25, 0.06, 0.8, 0.025]), max_nfev=250)
        fits.append(fit)
    best = min(fits, key=lambda fit: np.sum(fit.fun ** 2))
    return {'family': 'privileged_lotka_volterra', 'parameters': best.x.tolist(),
            'training_scaled_sse': float(np.sum(best.fun ** 2)),
            'success': bool(best.success)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    times, response = read_rows(args.public / 'observations.csv')
    if len(times) != 8 or not np.isfinite(response).all():
        raise ValueError('Expected eight finite public observations')
    models = {family: choose_ridge(times, response, family) for family in ('rbf', 'fourier')}
    models['privileged_lotka_volterra'] = fit_mechanistic(times, response)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'models.json').write_text(json.dumps(models, indent=2) + '\n')
    print({name: model.get('loocv_mse', model.get('training_scaled_sse'))
           for name, model in models.items()})


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Fit a validated structural proposal using eight public observations only."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares

from boxing_lotka_fixed_data import choose_ridge, read_rows
from boxing_lotka_proposal import ALLOWED


def coupled_predict(times, parameters):
    alpha, beta, gamma, delta, prey0, predator0 = parameters
    def rhs(y, _time):
        prey, predator = y
        return [alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator]
    try:
        times = np.asarray(times)
        grid = np.unique(np.r_[0, times])
        values = odeint(rhs, [prey0, predator0], grid, mxstep=2000)
        prediction = values[np.searchsorted(grid, times)]
        if not np.isfinite(prediction).all() or np.max(np.abs(prediction)) > 1e5:
            raise ValueError('unstable ODE')
        return prediction
    except Exception:
        return np.full((len(times), 2), 1e6)


def independent_predict(times, parameters):
    initial = np.asarray(parameters[:2])
    rate = np.asarray(parameters[2:])
    return initial * np.exp(np.asarray(times)[:, None] * rate)


def fit_coupled(times, response):
    starts = ([.1, .02, .4, .01, 20, 5], [.07, .015, .3, .008, 50, 20],
              [.14, .03, .5, .012, 80, 10])
    bounds = ([.005, .0005, .005, .0005, 1, 1],
              [.3, .08, 1, .04, 150, 100])
    fits = [least_squares(lambda p: ((coupled_predict(times, p) - response) / [40, 10]).ravel(),
                          start, bounds=bounds, max_nfev=350) for start in starts]
    best = min(fits, key=lambda fit: float(np.sum(fit.fun ** 2)))
    return {'family': 'coupled_bilinear_ode', 'parameters': best.x.tolist(),
            'training_scaled_sse': float(np.sum(best.fun ** 2)),
            'fit_success': bool(best.success)}


def fit_independent(times, response):
    start = [max(1, response[:, 0].mean()), max(1, response[:, 1].mean()), 0, 0]
    fit = least_squares(lambda p: ((independent_predict(times, p) - response) / [40, 10]).ravel(),
                        start, bounds=([.1, .1, -.2, -.2], [200, 200, .2, .2]),
                        max_nfev=350)
    return {'family': 'independent_exponential', 'parameters': fit.x.tolist(),
            'training_scaled_sse': float(np.sum(fit.fun ** 2)),
            'fit_success': bool(fit.success)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--proposal', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    proposal = json.loads(a.proposal.read_text())
    if set(proposal) != {'family', 'reason'} or proposal['family'] not in ALLOWED:
        raise ValueError('invalid typed proposal')
    times, response = read_rows(a.public / 'observations.csv')
    if len(times) != 8 or not np.isfinite(response).all():
        raise ValueError('expected eight finite public observations')
    if proposal['family'] == 'coupled_bilinear_ode':
        model = fit_coupled(times, response)
    elif proposal['family'] == 'independent_exponential':
        model = fit_independent(times, response)
    else:
        candidates = [choose_ridge(times, response, family) for family in ('rbf', 'fourier')]
        model = min(candidates, key=lambda item: item['loocv_mse'])
        model['family'] = 'unsure_fallback_' + model['family']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(model, indent=2) + '\n')
    print(model['family'])


if __name__ == '__main__':
    main()

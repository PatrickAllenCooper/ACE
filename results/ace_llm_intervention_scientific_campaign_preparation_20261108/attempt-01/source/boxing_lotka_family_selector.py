#!/usr/bin/env python3
"""Numerical-only selector over the same typed families offered to Qwen.

Eight-fold leave-one-out selection uses public observations only. The selected
fit is frozen before the separate held-out scorer is run.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from boxing_lotka_fixed_data import choose_ridge, features, read_rows
from fit_boxing_lotka_proposal import (coupled_predict, fit_coupled,
                                       fit_independent, independent_predict)


def predict(model, times):
    family = model['family']
    if family == 'coupled_bilinear_ode':
        return coupled_predict(times, model['parameters'])
    if family == 'independent_exponential':
        return independent_predict(times, model['parameters'])
    return features(times, family, model['width']) @ np.asarray(model['coefficients'])


def fit(family, times, response):
    if family == 'coupled_bilinear_ode':
        return fit_coupled(times, response)
    if family == 'independent_exponential':
        return fit_independent(times, response)
    return choose_ridge(times, response, family)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    times, response = read_rows(a.public / 'observations.csv')
    if len(times) != 8 or not np.isfinite(response).all():
        raise ValueError('expected eight finite public observations')
    families = ('coupled_bilinear_ode', 'independent_exponential', 'rbf', 'fourier')
    scores = {}
    for family in families:
        errors = []
        for i in range(8):
            keep = np.arange(8) != i
            fitted = fit(family, times[keep], response[keep])
            pred = predict(fitted, times[[i]])
            errors.append(float(np.mean((pred - response[[i]]) ** 2)))
        scores[family] = float(np.mean(errors))
    chosen = min(families, key=lambda name: scores[name])
    model = fit(chosen, times, response)
    report = {'selected_family': chosen, 'public_loocv_mse': scores,
              'selected_model': model, 'n_train': 8,
              'private_files_read': 0, 'closed_model_calls': 0}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(chosen, scores)


if __name__ == '__main__':
    main()

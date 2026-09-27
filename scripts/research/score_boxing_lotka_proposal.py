#!/usr/bin/env python3
"""Evaluate a frozen fitted proposal on a separately read private panel."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from boxing_lotka_fixed_data import features, read_rows
from fit_boxing_lotka_proposal import coupled_predict, independent_predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    model = json.loads(a.model.read_text())
    times, actual = read_rows(a.private / 'holdout.csv')
    if len(times) != 16 or not np.isfinite(actual).all():
        raise ValueError('expected sixteen finite held-out responses')
    family = model['family']
    if family == 'coupled_bilinear_ode':
        pred = coupled_predict(times, model['parameters'])
    elif family == 'independent_exponential':
        pred = independent_predict(times, model['parameters'])
    elif family in ('unsure_fallback_rbf', 'unsure_fallback_fourier'):
        base = family.removeprefix('unsure_fallback_')
        pred = features(times, base, model['width']) @ np.asarray(model['coefficients'])
    else:
        raise ValueError('unrecognized fitted family')
    pred = np.clip(pred, 0, None)
    row = {'family': family, 'n_train': 8, 'n_heldout': 16,
           'mae': float(np.abs(pred - actual).mean()),
           'rmse': float(np.sqrt(np.mean((pred - actual) ** 2)))}
    if not all(np.isfinite(row[key]) for key in ('mae', 'rmse')):
        raise ValueError('nonfinite score')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == '__main__':
    main()

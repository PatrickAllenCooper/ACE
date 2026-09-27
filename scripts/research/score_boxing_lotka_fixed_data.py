#!/usr/bin/env python3
"""Score frozen fixed-data models in a separate process using private holdout."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from boxing_lotka_fixed_data import features, read_rows, simulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    models = json.loads(args.models.read_text())
    times, actual = read_rows(args.private / 'holdout.csv')
    if len(times) != 16 or not np.isfinite(actual).all():
        raise ValueError('Expected sixteen finite held-out observations')
    rows = []
    for name, model in models.items():
        if name == 'privileged_lotka_volterra':
            prediction = simulate(times, model['parameters'])
        else:
            prediction = features(times, name, model['width']) @ np.array(model['coefficients'])
        prediction = np.clip(prediction, 0, None)
        rows.append({'arm': name, 'n_train': 8, 'n_heldout': len(times),
                     'mae': float(np.abs(prediction - actual).mean()),
                     'rmse': float(np.sqrt(np.mean((prediction - actual) ** 2)))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(rows)


if __name__ == '__main__':
    main()

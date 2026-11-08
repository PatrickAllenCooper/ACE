#!/usr/bin/env python3
"""Exact nuisance-adjusted information for a two-parent interaction toy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def information(x1: np.ndarray, x2: np.ndarray, sigma: float) -> tuple[int, float]:
    nuisance = np.column_stack((np.ones(len(x1)), x1, x2))
    interaction = x1 * x2
    full = np.column_stack((nuisance, interaction))
    projection = np.einsum('ij,j->i', nuisance,
                           np.linalg.lstsq(nuisance, interaction, rcond=None)[0],
                           optimize=False)
    residual = interaction - projection
    value = float(np.dot(residual, residual) / (len(x1) * sigma**2))
    return int(np.linalg.matrix_rank(full)), 0.0 if value < 1e-12 else value


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    n, sigma = 400, .15
    rng = np.random.default_rng(5123)
    rows = []
    for tau in (0., .01, .1, .5):
        designs = {
            'single_fixed_x1': (np.ones(n), rng.normal(0, tau, n)),
            'single_two_x1_levels': (np.tile([-1., 1.], n // 2), rng.normal(0, tau, n)),
            'passive_independent': (rng.normal(0, tau, n), rng.normal(0, tau, n)),
            'joint_fixed_pair': (np.ones(n), np.ones(n)),
            'joint_factorial': (np.tile([-1., -1., 1., 1.], n // 4),
                                np.tile([-1., 1., -1., 1.], n // 4)),
        }
        for design, (x1, x2) in designs.items():
            rank, empirical = information(x1, x2, sigma)
            theoretical = {
                'single_fixed_x1': 0.,
                'single_two_x1_levels': tau**2 / sigma**2,
                'passive_independent': tau**4 / sigma**2,
                'joint_fixed_pair': 0.,
                'joint_factorial': 1 / sigma**2,
            }[design]
            rows.append({'tau': tau, 'design': design, 'n': n,
                         'design_rank': rank, 'empirical_information_per_response': empirical,
                         'population_information_per_response': theoretical})
    path = a.output / 'information.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert all(r['empirical_information_per_response'] == 0 for r in rows
               if r['design'] in ('single_fixed_x1', 'joint_fixed_pair'))
    assert all(abs(r['empirical_information_per_response'] - 1 / sigma**2) < 1e-10
               for r in rows if r['design'] == 'joint_factorial')
    (a.output / 'complete.json').write_text(json.dumps({
        'rows': len(rows), 'responses_per_design': n, 'noise_sd': sigma,
        'rng_seed': 5123,
        'information_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'simulator_oracle_queries': 0, 'closed_model_calls': 0}, indent=2) + '\n')
    print('verified', len(rows), 'information cases')


if __name__ == '__main__':
    main()

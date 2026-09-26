#!/usr/bin/env python3
"""Validate and summarize the frozen Bayesian transfer confirmation grid."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from validate_cell import valid


def ratio_interval(numerator, denominator, seed=51911):
    a, b = np.asarray(numerator), np.asarray(denominator)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(a), (10000, len(a)))
    ratios = a[draws].mean(axis=1) / b[draws].mean(axis=1)
    return float(a.mean() / b.mean()), tuple(np.quantile(ratios, (.025, .975)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    a = p.parse_args()
    manifest = list(csv.DictReader((a.root / 'submitted.tsv').open(), delimiter='\t'))
    seeds = sorted(int(Path(entry['output']).name.removeprefix('seed_')) for entry in manifest)
    assert seeds == list(range(4000, 4020)), 'Manifest is not the frozen seed grid'
    data = {}
    hashes, revisions = set(), set()
    for seed in seeds:
        for typ in ('family', 'coefficient'):
            for k in (1, 3, 10):
                d = a.root / f'seed_{seed}' / typ / f'changed_{k}'
                ok, detail = valid(d, 'learned_transfer')
                assert ok, (d, detail)
                receipt = json.loads((d / 'complete.json').read_text())
                assert receipt['schema_version'] == 3
                spec = json.loads((d / 'system.json').read_text())
                assert (spec['seed'], spec['change_type'], spec['changed']) == (seed, typ, k)
                hashes.add(spec['source_sha256'])
                revisions.add(spec['source_revision'])
                with (d / 'metrics.csv').open() as stream:
                    for row in csv.DictReader(stream):
                        data[seed, typ, k, int(row['target_budget']), row['method']] = row
    assert len(data) == 20 * 2 * 3 * 3 * 5
    assert len(hashes) == len(revisions) == 1
    print(f'validated 120/120 cells, 1800 rows, one source hash, revision {next(iter(revisions))}')
    for typ in ('family', 'coefficient'):
        for k in (1, 3, 10):
            print(f'{typ} k={k}')
            for budget in (120, 200, 400):
                def metric(method, field):
                    return np.array([float(data[s, typ, k, budget, method][field]) for s in seeds])
                warm = metric('warm', 'changed_mse')
                retrieval = metric('source_retrieval', 'changed_mse')
                bayes = metric('source_bayes_mixture', 'changed_mse')
                unchanged_warm = metric('warm', 'unchanged_mse')
                unchanged_bayes = metric('source_bayes_mixture', 'unchanged_mse')
                changed_ratio, changed_ci = ratio_interval(bayes, warm)
                untouched_ratio, untouched_ci = ratio_interval(unchanged_bayes, unchanged_warm)
                print(f'  budget={budget}: changed warm={warm.mean():.5g} '
                      f'retrieval={retrieval.mean():.5g} bayes={bayes.mean():.5g}; '
                      f'bayes/warm={changed_ratio:.3f} CI={changed_ci[0]:.3f}..{changed_ci[1]:.3f}; '
                      f'unchanged bayes/warm={untouched_ratio:.3f} '
                      f'CI={untouched_ci[0]:.3f}..{untouched_ci[1]:.3f}')


if __name__ == '__main__':
    main()

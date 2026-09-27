#!/usr/bin/env python3
"""Validate and score the frozen fresh-seed endpoint-coverage comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from validate_cell import valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=Path(
        'docs/development/guidance/protocol_pev_endpoint_replication_2026-09-27.json'))
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    seeds = protocol['seeds']
    methods = protocol['methods']
    assert len(seeds) == 20 and len(methods) == 3
    manifest = list(csv.DictReader((args.root / 'submitted.tsv').open(), delimiter='\t'))
    expected = {(method, seed) for method in methods for seed in seeds}
    actual = {(Path(row['output']).parent.name, int(Path(row['output']).name.removeprefix('seed_')))
              for row in manifest}
    assert len(manifest) == len(actual) == 60 and actual == expected
    assert len({row['revision'] for row in manifest}) == 1
    records = []
    for seed in seeds:
        reference = None
        for method in methods:
            directory = args.root / 'shift30' / method / f'seed_{seed}'
            ok, reason = valid(directory, 'persistent')
            assert ok, (directory, reason)
            system = json.loads((directory / 'system.json').read_text())
            receipt = json.loads((directory / 'complete.json').read_text())
            assert receipt['schema_version'] == 3
            assert receipt['query_samples'] == receipt['budget'] == 2000
            with (directory / 'trajectory.csv').open() as stream:
                steps = list(csv.DictReader(stream))
            assert len(steps) == receipt['steps'] == 32
            assert int(steps[-1]['query_samples']) == 2000
            if reference is None:
                reference = system
            else:
                assert (system['family'], system['seed'], system['graph'], system['forms'],
                        system['nodes'], system['noise_std']) == (
                        reference['family'], reference['seed'], reference['graph'],
                        reference['forms'], reference['nodes'], reference['noise_std'])
                assert all(abs(value - reference['coeffs'][node][parent]) <= 1e-12
                           for node, coeffs in system['coeffs'].items()
                           for parent, value in coeffs.items())
            values = [float(row['value']) for row in steps]
            if method == 'nonleaf_extreme_coverage_ens':
                assert all(abs(value) == 5.0 for value in values)
            outcome = float(steps[-1]['feasible_mean_nonroot_loss'])
            assert math.isfinite(outcome)
            records.append({'seed': seed, 'method': method, 'query_samples': 2000,
                            'final_feasible_mean_nonroot': outcome,
                            'mean_abs_action_value': sum(map(abs, values)) / len(values),
                            'unique_targets': len({row['target'] for row in steps})})
    output = args.root / 'comparison.csv'
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    by_method = {(row['method'], row['seed']): row['final_feasible_mean_nonroot'] for row in records}
    results = {}
    for comparator in ('pev', 'pev_var'):
        differences = np.array([by_method[('nonleaf_extreme_coverage_ens', seed)]
                                - by_method[(comparator, seed)] for seed in seeds])
        mean = float(differences.mean())
        sem = float(stats.sem(differences))
        interval = stats.t.interval(0.95, len(seeds) - 1, loc=mean, scale=sem)
        pvalue = float(stats.ttest_1samp(differences, 0).pvalue)
        results[comparator] = {'mean_difference': mean, 'ci95': list(map(float, interval)),
                               'paired_t_p': pvalue, 'wins': int((differences < 0).sum()),
                               'differences': differences.tolist()}
    receipt = {'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
               'comparison_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'source_revision': manifest[0]['revision'], 'validated_cells': 60,
               'paired_systems': 20, 'results': results}
    (args.root / 'comparison_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for method in methods:
        print(method, np.mean([by_method[(method, seed)] for seed in seeds]))
    for comparator, result in results.items():
        print(f'endpoint coverage - {comparator}: {result}')


if __name__ == '__main__':
    main()

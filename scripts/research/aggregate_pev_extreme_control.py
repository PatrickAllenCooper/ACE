#!/usr/bin/env python3
"""Validate endpoint-value control cells against the archived shift30 replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from validate_cell import valid

SEEDS = (42, 123, 456)
NEW = ('nonleaf_extreme_random_ens', 'nonleaf_extreme_coverage_ens')
OLD = ('nonleaf_random_ens', 'nonleaf_coverage_ens', 'pev', 'pev_var')


def cell(path: Path):
    ok, reason = valid(path, 'persistent')
    assert ok, (path, reason)
    spec = json.loads((path / 'system.json').read_text())
    receipt = json.loads((path / 'complete.json').read_text())
    assert receipt['schema_version'] == 3 and receipt['query_samples'] == receipt['budget'] == 2000
    with (path / 'trajectory.csv').open() as stream:
        steps = list(csv.DictReader(stream))
    assert len(steps) == receipt['steps'] == 32
    return spec, steps


def same_system(a, b):
    assert (a['family'], a['seed'], a['graph'], a['forms']) == (
        b['family'], b['seed'], b['graph'], b['forms'])
    assert a['nodes'] == b['nodes'] and a['noise_std'] == b['noise_std']
    assert all(abs(v - b['coeffs'][n][p]) <= 1e-12
               for n, coeffs in a['coeffs'].items() for p, v in coeffs.items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--new-root', required=True, type=Path)
    p.add_argument('--reference-root', type=Path,
                   default=Path('results/research_pev_shift30_mean_pilot'))
    a = p.parse_args()
    manifest = list(csv.DictReader((a.new_root / 'submitted.tsv').open(), delimiter='\t'))
    expected = {(method, seed) for method in NEW for seed in SEEDS}
    actual = {(Path(row['output']).parent.name, int(Path(row['output']).name.removeprefix('seed_')))
              for row in manifest}
    assert len(manifest) == len(actual) == 6 and actual == expected
    assert len({row['revision'] for row in manifest}) == 1
    records = []
    for seed in SEEDS:
        ref_spec = None
        for method in OLD + NEW:
            root = a.new_root if method in NEW else a.reference_root
            spec, steps = cell(root / 'shift30' / method / f'seed_{seed}')
            if ref_spec is None:
                ref_spec = spec
            else:
                same_system(ref_spec, spec)
            values = [abs(float(row['value'])) for row in steps]
            if method in NEW:
                assert all(value == 5.0 for value in values)
            final = steps[-1]
            records.append({'seed': seed, 'method': method,
                            'query_samples': int(final['query_samples']),
                            'final_feasible_mean_nonroot': float(final['feasible_mean_nonroot_loss']),
                            'final_feasible_noisy_nonroot': float(final['feasible_nonroot_loss']),
                            'final_broad_nonroot': float(final['broad_nonroot_loss']),
                            'mean_abs_action_value': mean(values),
                            'unique_targets': len({row['target'] for row in steps})})
    output = a.new_root / 'comparison.csv'
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    receipt = {'validated_new_cells': 6, 'validated_reference_cells': 12,
               'paired_systems': 3,
               'comparison_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    (a.new_root / 'comparison_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for method in OLD + NEW:
        arm = [row for row in records if row['method'] == method]
        print(f'{method}: final mean={mean(row["final_feasible_mean_nonroot"] for row in arm):.6f}, '
              f'|value|={mean(row["mean_abs_action_value"] for row in arm):.3f}')
    print('validated six new cells, 12 archived reference cells, exact budgets, '
          'graph/forms and coefficient parity')


if __name__ == '__main__':
    main()

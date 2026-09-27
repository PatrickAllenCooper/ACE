#!/usr/bin/env python3
"""Validate and summarize the frozen nine-cell random-DAG development pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from validate_cell import valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=Path(
        'docs/development/guidance/protocol_pev_random_dag_dev_2026-09-27.json'))
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    manifest = list(csv.DictReader((args.root / 'submitted.tsv').open(), delimiter='\t'))
    expected = {(method, seed) for method in protocol['methods'] for seed in protocol['seeds']}
    actual = {(Path(row['output']).parent.name, int(Path(row['output']).name.removeprefix('seed_')))
              for row in manifest}
    assert len(manifest) == len(actual) == 9 and actual == expected
    assert len({row['revision'] for row in manifest}) == 1
    records = []
    system_hashes = []
    for seed in protocol['seeds']:
        hashes = []
        for method in protocol['methods']:
            directory = args.root / protocol['family'] / method / f'seed_{seed}'
            ok, reason = valid(directory, 'persistent')
            assert ok, (directory, reason)
            receipt = json.loads((directory / 'complete.json').read_text())
            assert receipt['schema_version'] == 3
            assert receipt['query_samples'] == receipt['budget'] == 2000
            hashes.append(receipt['system_sha256'])
            with (directory / 'trajectory.csv').open() as stream:
                steps = list(csv.DictReader(stream))
            assert len(steps) == receipt['steps'] == 32
            values = [float(row['value']) for row in steps]
            if method == 'nonleaf_extreme_coverage_ens':
                assert all(abs(v) == 5.0 for v in values)
            outcome = float(steps[-1]['feasible_mean_nonroot_loss'])
            assert math.isfinite(outcome)
            records.append({'seed': seed, 'method': method,
                            'final_feasible_mean_nonroot': outcome,
                            'mean_abs_action_value': mean(map(abs, values)),
                            'unique_targets': len({row['target'] for row in steps})})
        assert len(set(hashes)) == 1
        system_hashes.append(hashes[0])
    assert len(set(system_hashes)) == 3
    output = args.root / 'comparison.csv'
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    receipt = {'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
               'comparison_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'source_revision': manifest[0]['revision'],
               'validated_cells': 9, 'paired_systems': 3,
               'system_hashes': system_hashes}
    (args.root / 'comparison_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for method in protocol['methods']:
        arm = [row['final_feasible_mean_nonroot'] for row in records
               if row['method'] == method]
        print(method, f'mean={mean(arm):.6f}', 'per_seed=', [round(x, 6) for x in arm])


if __name__ == '__main__':
    main()

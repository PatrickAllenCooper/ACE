#!/usr/bin/env python3
"""Validate and summarize fixed-data source-library transfer cells."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

try:
    from .validate_cell import valid
except ImportError:
    from validate_cell import valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    a = p.parse_args()
    manifest = list(csv.DictReader((a.root / 'submitted.tsv').open(), delimiter='\t'))
    expected = {(typ, k, seed) for seed in range(3000, 3020)
                for typ in ('family', 'coefficient') for k in (1, 3, 10)}
    data = defaultdict(dict)
    source_hashes = set()
    invalid = []
    for entry in manifest:
        seed = int(Path(entry['output']).name.removeprefix('seed_'))
        for typ in ('family', 'coefficient'):
            for k in (1, 3, 10):
                d = a.root / f'seed_{seed}' / typ / f'changed_{k}'
                ok, detail = valid(d, 'learned_transfer')
                if not ok:
                    invalid.append((seed, typ, k, detail))
                    continue
                spec = json.loads((d / 'system.json').read_text())
                assert (spec['seed'], spec['change_type'], spec['changed']) == (seed, typ, k)
                source_hashes.add(spec['source_sha256'])
                with (d / 'metrics.csv').open() as stream:
                    for row in csv.DictReader(stream):
                        data[(typ, k, seed, int(row['target_budget']))][row['method']] = row
    print(f'validated {len(data)}/{len(expected)*3} setting/budget cells from {len(manifest)} jobs; '
          f'{len(invalid)} invalid; {len(source_hashes)} source-library hashes')
    if invalid:
        for item in invalid[:12]:
            print('  pending/invalid:', item)
        return
    assert len(data) == len(expected) * 3 and len(source_hashes) == 1
    for typ in ('family', 'coefficient'):
        for k in (1, 3, 10):
            for budget in (120, 200, 400):
                subset = [data[typ, k, s, budget] for s in range(3000, 3020)]
                means = {m: mean(float(arms[m]['changed_mse']) for arms in subset)
                         for m in ('warm', 'source_retrieval', 'source_mixture')}
                paired = [float(arms['source_retrieval']['changed_mse']) - float(arms['warm']['changed_mse'])
                          for arms in subset]
                print(f'{typ} k={k} target={budget}: changed-node MSE '
                      f'warm={means["warm"]:.5g}, retrieval={means["source_retrieval"]:.5g}, '
                      f'mixture={means["source_mixture"]:.5g}; '
                      f'retrieval-warm mean={mean(paired):+.5g}, median={median(paired):+.5g}, '
                      f'better seeds={sum(x<0 for x in paired)}/20')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Audit the actual DAG provenance of the frozen shift30 confirmation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from validate_cell import valid

METHODS = ('nonleaf_random_ens', 'nonleaf_coverage_ens', 'pev', 'pev_var')
SEEDS = tuple(range(5000, 5020))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    cells = []
    for seed in SEEDS:
        specs = []
        for method in METHODS:
            directory = a.root / 'shift30' / method / f'seed_{seed}'
            ok, reason = valid(directory, 'persistent')
            assert ok, (directory, reason)
            spec = json.loads((directory / 'system.json').read_text())
            receipt = json.loads((directory / 'complete.json').read_text())
            assert receipt['schema_version'] == 3
            assert receipt['query_samples'] == receipt['budget'] == 2000
            assert (spec['seed'], spec['family']) == (seed, 'shift30')
            specs.append(spec)
        assert all(spec == specs[0] for spec in specs), seed
        spec = specs[0]
        graph = spec['graph']
        assert len(graph) == 30 and all(all(p in graph for p in parents) for parents in graph.values())
        assert all(int(p[1:]) < int(n[1:]) for n, parents in graph.items() for p in parents)
        eligible = sum(any(n in parents for parents in graph.values()) for n in graph)
        cells.append({'seed': seed, 'graph_sha256': digest(graph),
                      'coefficients_sha256': digest(spec['coeffs']),
                      'system_sha256': hashlib.sha256((a.root / 'shift30' / 'pev' /
                                                        f'seed_{seed}' / 'system.json').read_bytes()).hexdigest(),
                      'roots': sum(not parents for parents in graph.values()),
                      'edges': sum(map(len, graph.values())),
                      'eligible_targets': eligible,
                      'source_revision': spec['source_revision']})
    assert len({row['graph_sha256'] for row in cells}) == len(SEEDS)
    assert len({row['coefficients_sha256'] for row in cells}) == len(SEEDS)
    assert len({row['source_revision'] for row in cells}) == 1
    output = a.root / 'graph_provenance.csv'
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    receipt = {'cells_validated': len(SEEDS) * len(METHODS),
               'systems': len(SEEDS), 'unique_graphs': len(SEEDS),
               'graph_provenance_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    (a.root / 'graph_provenance_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'{len(SEEDS)} unique DAGs; edges {min(x["edges"] for x in cells)}–'
          f'{max(x["edges"] for x in cells)}; eligible targets '
          f'{min(x["eligible_targets"] for x in cells)}–'
          f'{max(x["eligible_targets"] for x in cells)}; 80/80 cells valid')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Audit executed action-value diversity on archived connected-SCM cells."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    cells = sorted(a.root.glob('seed_*/complete.json'))
    if len(cells) != 20:
        raise ValueError('expected 20 archived systems')
    per_seed = []
    for receipt_path in cells:
        cell = receipt_path.parent
        receipt = json.loads(receipt_path.read_text())
        for filename, key in (('actions.csv', 'actions_sha256'),
                              ('metrics.csv', 'metrics_sha256'),
                              ('system.json', 'system_sha256')):
            if sha(cell / filename) != receipt[key]:
                raise ValueError(f'archive hash mismatch: {cell / filename}')
        system = json.loads((cell / 'system.json').read_text())
        seed = int(system['seed'])
        actions = list(csv.DictReader((cell / 'actions.csv').open()))
        metrics = {r['method']: r for r in csv.DictReader((cell / 'metrics.csv').open())}
        for method in ('coverage_single', 'coverage_pair', 'risk_pair'):
            own = [r for r in actions if r['method'] == method]
            if len(own) != int(metrics[method]['steps']):
                raise ValueError('action/metric step mismatch')
            by_motif = defaultdict(list)
            for row in own:
                by_motif[int(row['motif'])].append(tuple(float(v) for v in row['levels'].split(',')))
            if any(len(vectors) > 0 and len(vectors[0]) != (1 if method.endswith('single') else 2)
                   for vectors in by_motif.values()):
                raise ValueError('actuator width mismatch')
            repeated_diverse = sum(len(set(vectors)) > 1 for vectors in by_motif.values())
            motif0 = by_motif.get(0, [])
            if method.endswith('pair') and motif0:
                design = np.array([[1, v[0], v[1], v[0] * v[1]] for v in motif0])
                direct_rank = int(np.linalg.matrix_rank(design))
            else:
                direct_rank = 0
            per_seed.append({'seed': seed, 'method': method,
                             'actions': len(own), 'motifs_visited': len(by_motif),
                             'motifs_with_diverse_repeats': repeated_diverse,
                             'motif0_actions': len(motif0),
                             'motif0_distinct_value_vectors': len(set(motif0)),
                             'motif0_direct_pair_design_rank': direct_rank,
                             'cost_spent': int(metrics[method]['cost_spent'])})
    summary = []
    for method in ('coverage_single', 'coverage_pair', 'risk_pair'):
        group = [r for r in per_seed if r['method'] == method]
        summary.append({'method': method, 'systems': len(group),
                        'mean_actions': float(np.mean([r['actions'] for r in group])),
                        'mean_motifs_visited': float(np.mean([r['motifs_visited'] for r in group])),
                        'systems_with_diverse_repeated_motif': sum(
                            r['motifs_with_diverse_repeats'] > 0 for r in group),
                        'systems_with_motif0_rank_at_least_2': sum(
                            r['motif0_direct_pair_design_rank'] >= 2 for r in group),
                        'mean_cost_spent': float(np.mean([r['cost_spent'] for r in group]))})
    a.output.mkdir(parents=True)
    per_path, summary_path = a.output / 'per_seed.csv', a.output / 'summary.csv'
    write_csv(per_path, sorted(per_seed, key=lambda r: (r['seed'], r['method'])))
    write_csv(summary_path, summary)
    (a.output / 'complete.json').write_text(json.dumps({
        'source_revision': json.loads(cells[0].read_text())['source_revision'],
        'input_cells': len(cells), 'per_seed_rows': len(per_seed),
        'per_seed_sha256': sha(per_path), 'summary_sha256': sha(summary_path),
        'new_queries': 0, 'closed_model_calls': 0}, indent=2) + '\n')
    print('validated', len(cells), 'archived systems; action-value audit complete')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Summarize receipt-validated ACE research cells and paired pilot outcomes."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics

from validate_cell import valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    manifest = root / 'submitted.tsv'
    if not manifest.exists():
        print(f'No submission ledger at {manifest}')
        return
    with manifest.open() as stream:
        records = list(csv.DictReader(stream, delimiter='\t'))
    latest = {row['output']: row for row in records}
    grouped = defaultdict(lambda: [0, 0, 0])  # total, valid, invalid/pending
    numerical = defaultdict(list)
    for output, rec in latest.items():
        folder = Path(output)
        kind = 'pev' if '/pev_canary/' in output else 'agenda'
        parts = folder.relative_to(root).parts if folder.is_relative_to(root) else (kind,)
        depth = 4 if len(parts) > 1 and parts[1] == 'design' else (2 if len(parts) > 1 and parts[1] == 'prior' else 3)
        key = '/'.join(parts[:depth])
        grouped[key][0] += 1
        ok, _ = valid(folder, kind)
        grouped[key][1 if ok else 2] += 1
        if ok and kind == 'agenda':
            with (folder / 'metrics.csv').open() as stream:
                metric_rows = list(csv.DictReader(stream))
            for row in metric_rows:
                # Interpret only the last budget reached by each method within a cell.
                method = row['method']
                group_key = (key, method)
                budget = int(row['budget'])
                previous = numerical[group_key]
                previous.append((str(folder), budget, float(row['mse'])))
    print(f'cells: {len(latest)} distinct outputs, {sum(v[1] for v in grouped.values())} valid, '
          f'{sum(v[2] for v in grouped.values())} pending/invalid')
    for key, (total, ready, missing) in sorted(grouped.items()):
        print(f'  {key}: {ready}/{total} valid')
    print('pilot end-budget MSE by method (descriptive; no confirmatory tests):')
    for (key, method), values in sorted(numerical.items()):
        per_cell = {}
        for folder, budget, error in values:
            if folder not in per_cell or budget >= per_cell[folder][0]:
                per_cell[folder] = (budget, error)
        errors = [item[1] for item in per_cell.values()]
        if errors:
            spread = statistics.stdev(errors) if len(errors) > 1 else 0.0
            print(f'  {key}/{method}: n={len(errors)}, mean={statistics.mean(errors):.6g}, sd={spread:.6g}')


if __name__ == '__main__':
    main()

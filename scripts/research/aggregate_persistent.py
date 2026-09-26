#!/usr/bin/env python3
"""Audit and summarize locally copied persistent SCM campaigns.

Shows paired, final-checkpoint differences only. Development seeds are not a
confirmatory population sample, especially for the fixed legacy five-node SCM.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

try:
    from .validate_cell import valid
except ImportError:  # direct CLI execution
    from validate_cell import valid


def collect(root: Path):
    manifest = root / 'submitted.tsv'
    with manifest.open() as stream:
        submitted = list(csv.DictReader(stream, delimiter='\t'))
    outputs = {}
    for entry in submitted:
        source = Path(entry['output'])
        if root.name not in source.parts:
            raise ValueError(f"Output outside research root: {source}")
        i = source.parts.index(root.name)
        folder = root.joinpath(*source.parts[i + 1:])
        key = (entry['job_name'], str(folder))
        outputs[key] = (folder, entry)
    cells = {}
    invalid = []
    for folder, entry in outputs.values():
        ok, reason = valid(folder, 'persistent')
        if not ok:
            invalid.append((entry['job_id'], entry['job_name'], reason))
            continue
        parts = folder.relative_to(root).parts
        if len(parts) != 3 or not parts[2].startswith('seed_'):
            raise ValueError(f'Unexpected cell path: {folder}')
        family, method, seed_name = parts
        seed = int(seed_name.removeprefix('seed_'))
        receipt = json.loads((folder / 'complete.json').read_text())
        if (receipt['family'], receipt['method'], receipt['seed']) != (family, method, seed):
            raise ValueError(f'Receipt identity mismatch: {folder}')
        with (folder / 'trajectory.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        key = (family, seed, method)
        cells[key] = {'job_id': entry['job_id'], 'revision': entry['revision'],
                      'steps': receipt['steps'], 'samples': receipt['query_samples'],
                      'broad': float(rows[-1]['broad_total_loss']),
                      'observed': float(rows[-1]['observed_total_loss'])}
        if receipt['schema_version'] >= 2:
            cells[key]['broad_nonroot'] = float(rows[-1]['broad_nonroot_loss'])
            cells[key]['feasible_nonroot'] = float(rows[-1]['feasible_nonroot_loss'])
            cells[key]['system_hash'] = receipt['system_sha256']
    return cells, invalid, len(outputs)


def report(cells, invalid, total):
    print(f'validated: {len(cells)}/{total} cells')
    for job_id, name, reason in invalid:
        print(f'  pending/invalid: {job_id} {name}: {reason}')
    grouped = defaultdict(dict)
    for (family, seed, method), cell in cells.items():
        grouped[(family, seed)][method] = cell
    primary_p = {}
    for family in sorted({k[0] for k in grouped}):
        cohort = [(seed, arms) for (f, seed), arms in grouped.items() if f == family]
        print(f'\n{family}: {len(cohort)} paired systems/seeds')
        for seed, arms in sorted(cohort):
            counts = {v['samples'] for v in arms.values()}
            revisions = {v['revision'] for v in arms.values()}
            systems = {v.get('system_hash') for v in arms.values()}
            parity = 'MATCHED' if len(counts) == len(revisions) == len(systems) == 1 else 'MISMATCH'
            print(f'  seed {seed}: {parity}, {len(arms)} arms, samples={sorted(counts)}')
        for method in sorted({m for _, arms in cohort for m in arms}):
            vals = [arms[method] for _, arms in cohort if method in arms]
            print(f'  {method}: n={len(vals)}, final broad={mean(v["broad"] for v in vals):.5g}, '
                  f'final observed={mean(v["observed"] for v in vals):.5g}')
            if all('feasible_nonroot' in v for v in vals):
                print(f'    final feasible nonroot={mean(v["feasible_nonroot"] for v in vals):.5g}, '
                      f'broad nonroot={mean(v["broad_nonroot"] for v in vals):.5g}')
        metric = 'feasible_nonroot' if all(
            'feasible_nonroot' in v for _, arms in cohort for v in arms.values()) else 'broad'
        for method, baseline in (('pev', 'nonleaf_coverage_ens'),
                                 ('pev', 'nonleaf_random_ens'),
                                 ('pev', 'pev_var')):
            paired = [(seed, arms[method][metric] - arms[baseline][metric])
                      for seed, arms in cohort
                      if method in arms and baseline in arms
                      and arms[method]['samples'] == arms[baseline]['samples']
                      and arms[method]['revision'] == arms[baseline]['revision']
                      and arms[method].get('system_hash') == arms[baseline].get('system_hash')]
            if paired:
                print(f'  {method} minus {baseline}, {metric} (negative is better): '
                      + ', '.join(f'{s}:{d:+.5g}' for s, d in sorted(paired)))
                if len(paired) == 20:
                    delta = [d for _, d in paired]
                    se = stdev(delta) / len(delta) ** 0.5
                    margin = float(stats.t.ppf(0.975, len(delta)-1)) * se
                    pval = float(stats.ttest_1samp(delta, 0).pvalue)
                    if method == 'pev' and baseline == 'nonleaf_coverage_ens' and family in ('hom30', 'hetero30'):
                        primary_p[family] = pval
                    print(f'    mean={mean(delta):+.5g}, 95% t CI '
                          f'[{mean(delta)-margin:+.5g}, {mean(delta)+margin:+.5g}], '
                          f'two-sided paired t p={pval:.5g}')
    if len(primary_p) == 2:
        ordered = sorted(primary_p.items(), key=lambda x: x[1])
        adjusted = {}
        running = 0.0
        for rank, (family, pval) in enumerate(ordered):
            running = max(running, min(1.0, (2-rank) * pval))
            adjusted[family] = running
        print('\nPrimary paired t tests, Holm adjusted: '
              + ', '.join(f'{family}={adjusted[family]:.5g}' for family in sorted(adjusted)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    cells, invalid, total = collect(args.root)
    report(cells, invalid, total)


if __name__ == '__main__':
    main()

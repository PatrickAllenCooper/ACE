#!/usr/bin/env python3
"""Post hoc hard-vs-soft prior gate audit on archived development seeds only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

SEEDS = tuple(range(100, 112))
BUDGETS = (16, 32, 64)
THRESHOLDS = (-0.045, 0.0, 0.045, 0.09)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, default=Path('results/local_prior_gate_dev_20260926'))
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    source_hashes = []
    by_cell = {}
    for seed in SEEDS:
        folder = a.archive / f'seed_{seed}'
        receipt = json.loads((folder / 'complete.json').read_text())
        metric_file = folder / 'metrics.csv'
        assert hashlib.sha256(metric_file.read_bytes()).hexdigest() == receipt['metrics_sha256']
        source_hashes.append(receipt['metrics_sha256'])
        with metric_file.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 18
        for row in rows:
            key = (seed, row['condition'], int(row['budget']), row['method'])
            assert key not in by_cell
            by_cell[key] = row
    assert len(by_cell) == 12 * 2 * 3 * 3
    outcomes = []
    for budget in BUDGETS:
        for condition in ('correct', 'wrong'):
            for threshold in THRESHOLDS:
                errors = {'broad': [], 'proposal': [], 'validation_gate': [], 'hard_gate': []}
                selected = 0
                for seed in SEEDS:
                    arms = {m: by_cell[seed, condition, budget, m]
                            for m in ('broad', 'proposal', 'validation_gate')}
                    delta = (float(arms['validation_gate']['validation_proposal_sse']) -
                             float(arms['validation_gate']['validation_base_sse']))
                    choose_proposal = delta < threshold
                    selected += choose_proposal
                    for method in ('broad', 'proposal', 'validation_gate'):
                        errors[method].append(float(arms[method]['mse']))
                    errors['hard_gate'].append(float(arms['proposal' if choose_proposal else 'broad']['mse']))
                outcomes.append({'budget': budget, 'condition': condition,
                                 'sse_threshold': threshold, 'selected_proposals': selected,
                                 'broad_mean_mse': mean(errors['broad']),
                                 'proposal_mean_mse': mean(errors['proposal']),
                                 'soft_gate_mean_mse': mean(errors['validation_gate']),
                                 'hard_gate_mean_mse': mean(errors['hard_gate'])})
    a.output.mkdir(parents=True, exist_ok=True)
    summary = a.output / 'summary.csv'
    with summary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    protocol = {'status': 'post hoc development threshold diagnostic',
                'seeds': list(SEEDS), 'budgets': list(BUDGETS),
                'sse_thresholds': list(THRESHOLDS),
                'selection': 'proposal iff held-out SSE(proposal)-SSE(broad) < threshold',
                'validation_examples_counted': 8,
                'source_metric_hashes': source_hashes}
    (a.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    receipt = {'summary_rows': len(outcomes)}
    for name in ('summary.csv', 'protocol.json'):
        receipt[name + '_sha256'] = hashlib.sha256((a.output / name).read_bytes()).hexdigest()
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'validated 12 archived receipts; {len(outcomes)} summary rows -> {a.output}')


if __name__ == '__main__':
    main()

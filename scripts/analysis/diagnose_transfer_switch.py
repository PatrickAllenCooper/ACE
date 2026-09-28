#!/usr/bin/env python3
"""Evaluation-only audit of the finite-source switch's 200-example misses."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    records = defaultdict(list)
    hashes = {}
    cells = sorted(a.root.glob('seed_*/source_n_*/*/changed_*/node_metrics.csv'))
    if len(cells) != 144:
        raise ValueError(f'expected 144 cells, got {len(cells)}')
    for path in cells:
        receipt = json.loads((path.parent / 'complete.json').read_text())
        if sha(path) != receipt['node_metrics_sha256']:
            raise ValueError(f'hash mismatch: {path}')
        hashes[str(path.relative_to(a.root))] = sha(path)
        for row in csv.DictReader(path.open()):
            if int(row['budget']) == 200:
                records[int(row['source_n']), row['change_type'], int(row['changed'])].append(row)
    summary = []
    for (source_n, change_type, changed), rows in sorted(records.items()):
        for changed_node in (0, 1):
            subset = [r for r in rows if int(r['changed_node']) == changed_node]
            expected = 12 * (changed if changed_node else 30 - changed)
            if len(subset) != expected:
                raise ValueError('node count mismatch')
            missed = [r for r in subset if not int(r['switched'])]
            no_nomination = [r for r in missed if not int(r['nominated'])]
            failed_confirmation = [r for r in missed if int(r['nominated'])]
            later_positive = [r for r in no_nomination
                              if float(r['confirmation_log_bf']) > math.log(4)]
            # If nomination were removed, this is the possible false-switch
            # cost on unchanged nodes. It is a post hoc diagnostic, not a policy.
            late_flagged = [r for r in subset
                            if float(r['confirmation_log_bf']) > math.log(4)]
            summary.append({
                'source_n': source_n, 'change_type': change_type, 'changed': changed,
                'changed_node': changed_node, 'nodes': len(subset),
                'nominated': sum(int(r['nominated']) for r in subset),
                'switched': sum(int(r['switched']) for r in subset),
                'missed': len(missed), 'missed_no_nomination': len(no_nomination),
                'missed_failed_confirmation': len(failed_confirmation),
                'no_nomination_with_later_bf_above_log4': len(later_positive),
                'missed_excess_mse_vs_scratch_sum': sum(
                    float(r['warm_mse']) - float(r['scratch_mse']) for r in missed),
                'later_bf_above_log4_count': len(late_flagged),
                'posthoc_scratch_excess_mse_sum_if_later_only': sum(
                    float(r['scratch_mse']) - float(r['warm_mse']) for r in late_flagged),
            })
    a.output.mkdir(parents=True)
    path = a.output / 'summary.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    source_receipt = json.loads((a.root / 'suite_complete.json').read_text())
    (a.output / 'complete.json').write_text(json.dumps({
        'source_suite_sha256': sha(a.root / 'suite_complete.json'),
        'source_revision': source_receipt['source_revision'],
        'input_cells': len(cells), 'input_cell_sha256': hashes,
        'summary_rows': len(summary), 'summary_sha256': sha(path),
        'new_target_examples': 0, 'closed_model_calls': 0,
    }, indent=2) + '\n')
    print(len(cells), 'validated input cells;', len(summary), 'audit rows')


if __name__ == '__main__':
    main()

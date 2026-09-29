#!/usr/bin/env python3
"""Evaluation-only ranking audit on archived four-response transfer assays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from transfer_finite_source_switch_dev import log_evidence  # noqa: E402
from transfer_safe_switch_dev import target  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    suite = json.loads((a.root / 'suite_complete.json').read_text())
    rows = []
    for seed in range(700, 720):
        for source_n in (16, 64):
            src = a.root / f'seed_{seed}' / f'source_n_{source_n}' / 'source'
            receipt = json.loads((src / 'complete.json').read_text())
            path = src / 'source_posteriors.npz'
            if sha(path) != receipt['source_posteriors_sha256']:
                raise ValueError('source posterior hash mismatch')
            with np.load(path) as z:
                means, covs = z['mean'], z['covariance']
            for change_type in ('family', 'coefficient'):
                _, _, changed_ids, x, y, _ = target(seed, 1, change_type)
                scores = []
                for node in range(30):
                    xi, yi = x[node, :4], y[node, :4]
                    scores.append(log_evidence(xi, yi, np.zeros(6), np.eye(6) / .25, .15) -
                                  log_evidence(xi, yi, means[node], covs[node], .15))
                order = np.argsort(-np.asarray(scores))
                rank = int(np.where(order == changed_ids[0])[0][0]) + 1
                rows.append({'seed': seed, 'source_n': source_n,
                             'change_type': change_type, 'changed_node_rank': rank,
                             'top_8': int(rank <= 8),
                             'changed_score': float(scores[int(changed_ids[0])]),
                             'source_posterior_sha256': sha(path)})
    if len(rows) != 80:
        raise ValueError('incomplete k=1 rank audit')
    a.output.mkdir(parents=True)
    path = a.output / 'ranks.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for n in (16, 64):
        for typ in ('family', 'coefficient'):
            group = [r for r in rows if r['source_n'] == n and r['change_type'] == typ]
            summary.append({'source_n': n, 'change_type': typ,
                            'systems': len(group), 'top_8': sum(r['top_8'] for r in group),
                            'rank_1': sum(r['changed_node_rank'] == 1 for r in group)})
    summary_path = a.output / 'summary.csv'
    with summary_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (a.output / 'complete.json').write_text(json.dumps({
        'source_revision': suite['source_revision'],
        'parent_suite_sha256': sha(a.root / 'suite_complete.json'),
        'rank_rows': len(rows), 'rank_sha256': sha(path),
        'summary_sha256': sha(summary_path),
        'new_source_examples': 0, 'new_target_examples': 0,
        'closed_model_calls': 0}, indent=2) + '\n')
    print('top-eight', sum(r['top_8'] for r in rows), '/', len(rows))


if __name__ == '__main__':
    main()

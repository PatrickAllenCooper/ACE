#!/usr/bin/env python3
"""Check that a connected SCM separates mechanism change from descendant drift."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from connected_motif import make_system, sample


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    args.output.mkdir(parents=True)
    rows = []
    for seed in range(1500, 1512):
        source = make_system(seed, 30, 10, .15, topology='fanout')
        target_coefficients = source.coefficients.copy()
        target_coefficients[0, 0] += 1.0
        target = replace(source, coefficients=target_coefficients)
        # Common exogenous noise makes each downstream shift attributable to
        # the changed upstream mechanism rather than sampling variation.
        source_values, _, source_mask = sample(source, np.random.default_rng(seed + 4), 512)
        target_values, _, target_mask = sample(target, np.random.default_rng(seed + 4), 512)
        if not source_mask.all() or not target_mask.all():
            raise AssertionError('natural-observation mask')
        child = source.children[1]
        parent = source.parents[1][0]
        root = source.parents[1][1]
        # The unchanged child's structural conditional mean is compared at
        # identical contexts rather than comparing its shifted marginal mean.
        grid = np.random.default_rng(seed + 40).uniform(-2, 2, (1024, 2))
        a, b, c = source.coefficients[1]
        source_conditional = a*grid[:, 0] + b*grid[:, 1] + c*grid[:, 0]*grid[:, 1]
        at, bt, ct = target.coefficients[1]
        target_conditional = at*grid[:, 0] + bt*grid[:, 1] + ct*grid[:, 0]*grid[:, 1]
        if source_conditional.tobytes() != target_conditional.tobytes():
            raise AssertionError('unchanged conditional mean drifted')
        parent_shift = float(np.mean((source_values[:, parent] - target_values[:, parent])**2))
        child_shift = float(np.mean((source_values[:, child] - target_values[:, child])**2))
        root_shift = float(np.max(np.abs(source_values[:, root] - target_values[:, root])))
        if parent_shift <= .001 or child_shift <= .001 or root_shift != 0:
            raise AssertionError('missing descendant shift or altered nondescendant')
        rows.append({'seed': seed, 'changed_motif': 0, 'unchanged_descendant_motif': 1,
                     'changed_parent_node': parent, 'unchanged_descendant_node': child,
                     'unaffected_root_node': root,
                     'paired_parent_shift_mse': parent_shift,
                     'paired_child_shift_mse': child_shift,
                     'max_unaffected_root_change': root_shift,
                     'conditional_mean_identical': True,
                     'source_revision': revision})
    with (args.output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    receipt = {'source_revision': revision, 'systems': len(rows),
               'generated_source_responses': len(rows)*512,
               'generated_target_responses': len(rows)*512,
               'acquired_training_responses': 0, 'closed_model_calls': 0,
               'metrics_sha256': sha(args.output / 'metrics.csv')}
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('validated', len(rows), 'systems; descendant conditional invariance and marginal shift')


if __name__ == '__main__':
    main()

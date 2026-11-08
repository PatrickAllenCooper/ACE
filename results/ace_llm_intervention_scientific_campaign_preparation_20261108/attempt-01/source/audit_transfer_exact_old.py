#!/usr/bin/env python3
"""Evaluator-only audit of exact old mechanisms and floor-six harm."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from transfer_safe_switch_dev import SEEDS, node_mse, target


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--floor', type=Path, required=True)
    p.add_argument('--uniform', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    old_errors = defaultdict(list)
    exact_unchanged = changed_equal = 0
    for seed in SEEDS:
        for change_type in ('family', 'coefficient'):
            for changed in (1, 3, 10):
                old, truth, changed_ids, _, _, test_phi = target(seed, changed, change_type)
                mask = np.zeros(30, dtype=bool)
                mask[changed_ids] = True
                for i in range(30):
                    if mask[i]:
                        changed_equal += int(np.array_equal(old[i], truth[i]))
                    else:
                        exact_unchanged += int(np.array_equal(old[i], truth[i]))
                    old_errors[(change_type, changed, bool(mask[i]))].append(
                        node_mse(old[i], truth[i], test_phi))
    floor = list(csv.DictReader(a.floor.open()))
    uniform = {(int(r['seed']), r['change_type'], int(r['changed']), int(r['node'])):
               float(r['warm_node_mse']) for r in csv.DictReader(a.uniform.open())
               if int(r['target_budget']) == 200 and float(r['odds']) == 4.0}
    if len(floor) != 2160 or len(uniform) != 2160:
        raise ValueError('unexpected archived row count')
    groups = defaultdict(list)
    for row in floor:
        key = (int(row['seed']), row['change_type'], int(row['changed']), int(row['node']))
        if key not in uniform:
            raise ValueError('floor/uniform node mismatch')
        if row['change_type'] == 'family' and int(row['changed']) == 10 and not int(row['changed_node']):
            n = int(row['total_samples'])
            expected_uniform = 7 if int(row['node']) < 20 else 6
            groups[(n, expected_uniform)].append((float(row['adaptive_warm_mse']), uniform[key]))
    group_report = []
    for (adaptive_n, uniform_n), pairs in sorted(groups.items()):
        group_report.append({'adaptive_samples': adaptive_n, 'uniform_samples': uniform_n,
                             'nodes': len(pairs), 'adaptive_mse_sum': sum(x for x, _ in pairs),
                             'uniform_mse_sum': sum(y for _, y in pairs),
                             'mse_ratio': sum(x for x, _ in pairs) / sum(y for _, y in pairs)})
    report = {'development_seeds': list(SEEDS), 'settings': 72,
              'source_revision': subprocess.check_output(
                  ['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'audit_script_sha256': sha(Path(__file__)),
              'nodes_total': 2160, 'unchanged_exact_old_coefficients': exact_unchanged,
              'changed_equal_old_coefficients': changed_equal,
              'old_mean_mse_by_setting': [
                  {'change_type': typ, 'changed': k, 'changed_node': int(changed_node),
                   'nodes': len(values), 'old_mean_mse': float(np.mean(values))}
                  for (typ, k, changed_node), values in sorted(old_errors.items())],
              'family_k10_untouched_floor6_vs_uniform_by_samples': group_report,
              'floor_sha256': sha(a.floor), 'uniform_sha256': sha(a.uniform),
              'new_oracle_actions': 0, 'closed_model_calls': 0,
              'audit_reads_hidden_truth': True,
              'audit_output_is_for_evaluation_only': True}
    if exact_unchanged != 1824 or changed_equal != 0 or len(floor) != 2160 or len(uniform) != 2160:
        raise ValueError('benchmark construction differs from expected exact-old design')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print('exact old on', exact_unchanged, 'unchanged nodes; floor groups', group_report)


if __name__ == '__main__':
    main()

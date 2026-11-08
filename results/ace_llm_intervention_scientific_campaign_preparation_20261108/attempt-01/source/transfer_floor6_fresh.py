#!/usr/bin/env python3
"""Fresh coverage-floor transfer policy on strong and weaker held-out forms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import t

from agenda_runner import features
from transfer_safe_switch_dev import target
from transfer_soft_mixture_fresh import predictions, source_observations
from transfer_top8_allocation_fresh import allocate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def heldout(x: np.ndarray, amplitude: float) -> np.ndarray:
    return amplitude * np.tanh(1.7 * x[..., 0] + .8 * x[..., 1])


def make_target(seed: int, setting: str):
    old, family_truth, changed_ids, x, family_y, test_phi = target(seed, 1, 'family')
    changed = int(changed_ids[0])
    test_x = np.random.default_rng(seed + 91473).uniform(-2, 2, (1024, 2))
    if not np.array_equal(features(test_x), test_phi):
        raise ValueError('test design reconstruction failed')
    if setting not in ('heldout_strong', 'heldout_weak'):
        raise ValueError(setting)
    amplitude = .85 if setting == 'heldout_strong' else .45
    train_phi = features(x[changed])
    y = family_y.copy()
    # Preserve the original independent noise stream while replacing the
    # within-bank family switch by a genuinely held-out conditional form.
    y[changed] -= train_phi @ (family_truth[changed] - old[changed])
    y[changed] += heldout(x[changed], amplitude)
    truth_mean = old @ test_phi.T
    term = heldout(test_x, amplitude)
    truth_mean[changed] += term
    projection = np.linalg.solve(test_phi.T @ test_phi, test_phi.T @ term)
    residual = float(np.mean((term - test_phi @ projection) ** 2))
    if not np.isfinite(residual) or residual <= .002:
        raise ValueError('held-out form lacks a meaningful out-of-bank residual')
    if not np.isfinite(y).all() or not np.isfinite(truth_mean).all():
        raise ValueError('nonfinite held-out conditional mean')
    return old, old, changed, x, y, test_phi, truth_mean, residual


def interval(values: list[float]) -> list[float]:
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    half = float(t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
    return [mean - half, mean + half]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    spec = json.loads(a.protocol.read_text())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    assert len(spec['system_seeds']) == len(set(spec['system_seeds'])) == 20
    assert spec['settings'] == ['heldout_strong', 'heldout_weak']
    assert spec['source_training_examples_per_node'] == [16, 64]
    a.output.mkdir(parents=True)
    protocol_hash = digest(a.protocol)
    source_count = target_arm_count = cell_union_sum = 0
    global_union: dict[tuple[int, str], np.ndarray] = {}
    all_rows = []
    for seed in spec['system_seeds']:
        for source_n in spec['source_training_examples_per_node']:
            source_dir = a.output / f'seed_{seed}' / f'source_n_{source_n}' / 'source'
            source_dir.mkdir(parents=True)
            sx, sy, means, covs = source_observations(seed, source_n, spec)
            np.savez_compressed(source_dir / 'source_data.npz', x=sx, y=sy)
            np.savez_compressed(source_dir / 'source_posteriors.npz',
                                mean=means, covariance=covs)
            (source_dir / 'complete.json').write_text(json.dumps({
                'source_revision': revision, 'protocol_sha256': protocol_hash,
                'source_responses': 30 * source_n,
                'source_data_sha256': digest(source_dir / 'source_data.npz'),
                'source_posteriors_sha256': digest(source_dir / 'source_posteriors.npz')
            }, indent=2) + '\n')
            source_count += 30 * source_n
            for setting in spec['settings']:
                old, truth, changed, x, y, test_phi, truth_mean, residual = make_target(seed, setting)
                adaptive, uniform, top_eight = allocate(x, y, means, covs)
                floor6 = np.full(30, 6, dtype=int)
                floor6[top_eight[:4]] += 5
                assert adaptive.sum() == uniform.sum() == floor6.sum() == 200
                assert min(adaptive) >= 4 and max(adaptive) <= x.shape[1]
                assert min(uniform) >= 4 and max(uniform) <= x.shape[1]
                assert min(floor6) == 6 and max(floor6) == 11
                key = (seed, setting)
                cell_union = np.maximum(np.maximum(adaptive, uniform), floor6)
                global_union[key] = np.maximum(global_union.get(key, np.zeros(30, dtype=int)),
                                               cell_union)
                cell = source_dir.parent / setting
                cell.mkdir()
                action = {'seed': seed, 'source_n': source_n, 'setting': setting,
                          'assay_responses': 120, 'top_eight': top_eight,
                          'adaptive_counts': adaptive.tolist(),
                          'uniform_counts': uniform.tolist(),
                          'floor6_counts': floor6.tolist(),
                          'target_responses_per_policy': 200,
                          'cell_union_acquired_prefix_responses': int(cell_union.sum()),
                          'simulator_generated_potential_responses': int(np.prod(y.shape))}
                (cell / 'actions.json').write_text(json.dumps(action, indent=2) + '\n')
                acquired = []
                for node, n in enumerate(cell_union):
                    for ordinal in range(n):
                        acquired.append({'node': node, 'ordinal': ordinal,
                                         'x1': float(x[node, ordinal, 0]),
                                         'x2': float(x[node, ordinal, 1]),
                                         'response': float(y[node, ordinal]),
                                         'adaptive': int(ordinal < adaptive[node]),
                                         'uniform': int(ordinal < uniform[node]),
                                         'floor6': int(ordinal < floor6[node])})
                write_csv(cell / 'acquired.csv', acquired)
                system = {'seed': seed, 'setting': setting, 'changed_node': changed,
                          'old_coefficients': old.tolist(),
                          'base_target_coefficients': truth.tolist(),
                          'heldout_term': spec['heldout_terms'][setting],
                          'out_of_bank_projection_residual_mse': residual,
                          'test_design_sha256': array_digest(test_phi),
                          'test_conditional_mean_sha256': array_digest(truth_mean)}
                (cell / 'system.json').write_text(json.dumps(system, indent=2) + '\n')
                rows = []
                for allocation, counts in (('adaptive_top8', adaptive), ('uniform', uniform),
                                           ('floor6_top4', floor6)):
                    for node, n in enumerate(counts):
                        fits, weight = predictions(x[node, :n], y[node, :n],
                                                   means[node], covs[node], .15)
                        for method in spec['predictors']:
                            pred = test_phi @ fits[method]
                            mse = float(np.mean((pred - truth_mean[node]) ** 2))
                            if not np.isfinite(mse):
                                raise ValueError('nonfinite score')
                            rows.append({'seed': seed, 'source_n': source_n,
                                         'setting': setting, 'allocation': allocation,
                                         'node': node, 'changed_node': int(node == changed),
                                         'target_examples': int(n), 'method': method,
                                         'scratch_weight': weight, 'mse': mse})
                write_csv(cell / 'node_metrics.csv', rows)
                receipt = {'source_revision': revision, 'protocol_sha256': protocol_hash,
                           'source_posteriors_sha256': digest(source_dir / 'source_posteriors.npz'),
                           'actions_sha256': digest(cell / 'actions.json'),
                           'acquired_sha256': digest(cell / 'acquired.csv'),
                           'system_sha256': digest(cell / 'system.json'),
                           'node_metrics_sha256': digest(cell / 'node_metrics.csv'),
                           'rows': len(rows), 'target_arm_responses': 600,
                           'cell_union_acquired_prefix_responses': len(acquired),
                           'closed_model_calls': 0}
                assert len(rows) == 270 and len(acquired) == action['cell_union_acquired_prefix_responses']
                (cell / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
                all_rows.extend(rows)
                target_arm_count += 600
                cell_union_sum += len(acquired)
    summary = []
    system_mean = {}
    for source_n in spec['source_training_examples_per_node']:
        for setting in spec['settings']:
            for changed_node in (0, 1):
                for allocation in ('adaptive_top8', 'uniform', 'floor6_top4'):
                    for method in spec['predictors']:
                        group = [r for r in all_rows if r['source_n'] == source_n and
                                 r['setting'] == setting and r['changed_node'] == changed_node and
                                 r['allocation'] == allocation and r['method'] == method]
                        expected = 20 * (1 if changed_node else 29)
                        assert len(group) == expected
                        summary.append({'source_n': source_n, 'setting': setting,
                                        'changed_node': changed_node, 'allocation': allocation,
                                        'method': method, 'nodes': len(group),
                                        'mean_mse': float(np.mean([r['mse'] for r in group]))})
                        for seed in spec['system_seeds']:
                            values = [r['mse'] for r in group if r['seed'] == seed]
                            system_mean[(source_n, setting, changed_node,
                                         allocation, method, seed)] = float(np.mean(values))
    write_csv(a.output / 'summary.csv', summary)
    gates = []
    for source_n in spec['source_training_examples_per_node']:
        for setting in spec['settings']:
            contrasts = [
                (1, 'floor6_top4', 'soft_mixture', 'uniform', 'soft_mixture', .8),
                (1, 'floor6_top4', 'soft_mixture', 'floor6_top4', 'scratch', 1.05),
                (0, 'floor6_top4', 'soft_mixture', 'uniform', 'soft_mixture', 1.05),
                (0, 'floor6_top4', 'soft_mixture', 'floor6_top4', 'source_warm', 1.05)]
            for flag, ca, cm, ba, bm, limit in contrasts:
                candidate = [system_mean[(source_n, setting, flag, ca, cm, seed)]
                             for seed in spec['system_seeds']]
                baseline = [system_mean[(source_n, setting, flag, ba, bm, seed)]
                            for seed in spec['system_seeds']]
                ratio = float(np.mean(candidate) / np.mean(baseline))
                gates.append({'source_n': source_n, 'setting': setting,
                              'changed_node': flag,
                              'candidate': f'{ca}/{cm}', 'baseline': f'{ba}/{bm}',
                              'ratio_of_means': ratio, 'limit': limit,
                              'passes_point_gate': ratio <= limit,
                              'paired_difference_95_interval': interval(
                                  [u - v for u, v in zip(candidate, baseline)])})
    (a.output / 'analysis.json').write_text(json.dumps({
        'heldout_gate_pass': all(r['passes_point_gate'] for r in gates),
        'contrasts': gates}, indent=2) + '\n')
    receipt = {'source_revision': revision, 'protocol_sha256': protocol_hash,
               'source_cells': 40, 'target_cells': 80,
               'source_training_responses': source_count,
               'target_arm_response_counts': target_arm_count,
               'sum_cell_union_acquired_prefix_responses': cell_union_sum,
               'global_unique_acquired_prefix_responses': int(sum(
                   counts.sum() for counts in global_union.values())),
               'node_method_rows': len(all_rows),
               'summary_sha256': digest(a.output / 'summary.csv'),
               'analysis_sha256': digest(a.output / 'analysis.json'),
               'closed_model_calls': 0}
    assert source_count == 48000 and target_arm_count == 48000
    assert len(all_rows) == 21600
    (a.output / 'suite_complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('heldout gate pass:', all(r['passes_point_gate'] for r in gates))
    for row in gates:
        print(row['source_n'], row['setting'], row['changed_node'], row['candidate'],
              'vs', row['baseline'], f"{row['ratio_of_means']:.4f}",
              'limit', row['limit'])
    print('validated', source_count, 'source and', target_arm_count,
          'target arm-response counts; global unique',
          int(sum(counts.sum() for counts in global_union.values())))


if __name__ == '__main__':
    main()

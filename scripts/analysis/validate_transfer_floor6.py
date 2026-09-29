#!/usr/bin/env python3
"""Independently validate coverage-floor transfer receipts and stored scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from agenda_runner import features
from transfer_soft_mixture_fresh import predictions


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--protocol', required=True, type=Path)
    a = p.parse_args()
    spec = json.loads(a.protocol.read_text())
    suite = json.loads((a.root / 'suite_complete.json').read_text())
    assert suite['protocol_sha256'] == sha(a.protocol)
    assert suite['summary_sha256'] == sha(a.root / 'summary.csv')
    assert suite['analysis_sha256'] == sha(a.root / 'analysis.json')
    source_total = arm_total = cell_union_total = 0
    global_union = {}
    metric_rows = 0
    for seed in spec['system_seeds']:
        test_x = np.random.default_rng(seed + 91473).uniform(-2, 2, (1024, 2))
        phi = features(test_x)
        for source_n in spec['source_training_examples_per_node']:
            parent = a.root / f'seed_{seed}' / f'source_n_{source_n}'
            src = parent / 'source'
            sr = json.loads((src / 'complete.json').read_text())
            assert sr['protocol_sha256'] == suite['protocol_sha256']
            assert sr['source_revision'] == suite['source_revision']
            assert sr['source_data_sha256'] == sha(src / 'source_data.npz')
            assert sr['source_posteriors_sha256'] == sha(src / 'source_posteriors.npz')
            with np.load(src / 'source_data.npz') as data:
                assert data['x'].shape == (30, source_n, 2)
                assert data['y'].shape == (30, source_n)
            with np.load(src / 'source_posteriors.npz') as data:
                means, covs = data['mean'], data['covariance']
                assert means.shape == (30, 6) and covs.shape == (30, 6, 6)
            assert sr['source_responses'] == 30 * source_n
            source_total += sr['source_responses']
            for setting in spec['settings']:
                cell = parent / setting
                receipt = json.loads((cell / 'complete.json').read_text())
                assert receipt['protocol_sha256'] == suite['protocol_sha256']
                assert receipt['source_revision'] == suite['source_revision']
                assert receipt['source_posteriors_sha256'] == sr['source_posteriors_sha256']
                for name in ('actions', 'acquired', 'system', 'node_metrics'):
                    suffix = 'json' if name in ('actions', 'system') else 'csv'
                    assert receipt[name + '_sha256'] == sha(cell / f'{name}.{suffix}')
                action = json.loads((cell / 'actions.json').read_text())
                system = json.loads((cell / 'system.json').read_text())
                assert system['seed'] == action['seed'] == seed
                assert system['setting'] == action['setting'] == setting
                assert system['test_design_sha256'] == array_sha(phi)
                base = np.asarray(system['base_target_coefficients'])
                truth = np.einsum('nd,td->nt', base, phi)
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    hash_truth = base @ phi.T
                assert setting in ('heldout_strong', 'heldout_weak')
                amplitude = .85 if setting == 'heldout_strong' else .45
                term = amplitude * np.tanh(1.7 * test_x[:, 0] + .8 * test_x[:, 1])
                truth[system['changed_node']] += term
                hash_truth[system['changed_node']] += term
                assert system['out_of_bank_projection_residual_mse'] > .002
                assert np.isfinite(hash_truth).all()
                assert np.max(np.abs(truth - hash_truth)) < 1e-12
                assert system['test_conditional_mean_sha256'] == array_sha(hash_truth)
                adaptive = np.asarray(action['adaptive_counts'])
                uniform = np.asarray(action['uniform_counts'])
                floor6 = np.asarray(action['floor6_counts'])
                assert adaptive.shape == uniform.shape == floor6.shape == (30,)
                assert adaptive.sum() == uniform.sum() == floor6.sum() == 200
                assert action['assay_responses'] == 120
                assert len(action['top_eight']) == len(set(action['top_eight'])) == 8
                expected_adaptive = np.full(30, 4)
                expected_adaptive[action['top_eight']] += 10
                assert np.array_equal(adaptive, expected_adaptive)
                expected_floor6 = np.full(30, 6)
                expected_floor6[action['top_eight'][:4]] += 5
                assert np.array_equal(floor6, expected_floor6)
                cell_union = np.maximum(np.maximum(adaptive, uniform), floor6)
                key = (seed, setting)
                global_union[key] = np.maximum(global_union.get(key, np.zeros(30, dtype=int)),
                                               cell_union)
                acquired = list(csv.DictReader((cell / 'acquired.csv').open()))
                assert len(acquired) == int(cell_union.sum()) == receipt['cell_union_acquired_prefix_responses']
                rows_by_node = [[] for _ in range(30)]
                for row in acquired:
                    node, ordinal = int(row['node']), int(row['ordinal'])
                    assert ordinal == len(rows_by_node[node])
                    assert int(row['adaptive']) == int(ordinal < adaptive[node])
                    assert int(row['uniform']) == int(ordinal < uniform[node])
                    assert int(row['floor6']) == int(ordinal < floor6[node])
                    rows_by_node[node].append(row)
                metrics = list(csv.DictReader((cell / 'node_metrics.csv').open()))
                assert len(metrics) == receipt['rows'] == 270
                lookup = {(r['allocation'], int(r['node']), r['method']): r for r in metrics}
                assert len(lookup) == 270
                for allocation, counts in (('adaptive_top8', adaptive), ('uniform', uniform),
                                           ('floor6_top4', floor6)):
                    for node, n in enumerate(counts):
                        own = rows_by_node[node][:n]
                        x = np.asarray([[float(r['x1']), float(r['x2'])] for r in own])
                        y = np.asarray([float(r['response']) for r in own])
                        fits, weight = predictions(x, y, means[node], covs[node], .15)
                        for method in spec['predictors']:
                            row = lookup[(allocation, node, method)]
                            score = float(np.mean((np.einsum('td,d->t', phi, fits[method]) -
                                                   truth[node]) ** 2))
                            assert np.isclose(score, float(row['mse']), rtol=0, atol=1e-10)
                            assert np.isclose(weight, float(row['scratch_weight']), rtol=0, atol=1e-10)
                            assert int(row['changed_node']) == int(node == system['changed_node'])
                            assert int(row['target_examples']) == n
                assert receipt['target_arm_responses'] == 600
                arm_total += 600
                cell_union_total += len(acquired)
                metric_rows += len(metrics)
    assert suite['source_cells'] == 40 and suite['target_cells'] == 80
    assert source_total == suite['source_training_responses'] == 48000
    assert arm_total == suite['target_arm_response_counts'] == 48000
    assert cell_union_total == suite['sum_cell_union_acquired_prefix_responses']
    assert sum(v.sum() for v in global_union.values()) == suite['global_unique_acquired_prefix_responses']
    assert metric_rows == suite['node_method_rows'] == 21600
    assert suite['closed_model_calls'] == 0
    print('PASS: 40 source receipts, 80 target receipts, 21600 independently recomputed scores')
    print('source', source_total, 'target arm', arm_total,
          'global unique acquired', suite['global_unique_acquired_prefix_responses'])


if __name__ == '__main__':
    main()

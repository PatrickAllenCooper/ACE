"""Read-only acceptance audit and compact custody of the frozen physical study."""
import argparse
import csv
import io
import math
from pathlib import Path
import shutil
import zipfile

from chambers_delivery_validation import validate_protocol, validate_fit_seal, partition, physics_features
from runner_delivery_confirmation import read, write, sha, utc


def audit(out, gate, projection, dest):
    import numpy as np
    out, dest = Path(out), Path(dest)
    p = validate_protocol(out)
    fit, summaries = validate_fit_seal(out, p)
    if p['stage_a_gate_sha256'] != sha(gate) or p['resource_projection_sha256'] != sha(projection):
        raise ValueError('selection/resource binding changed')
    complete = read(out / 'complete.json')
    if complete['scores_sha256'] != sha(out / 'scores.json') or complete['fit_seal_sha256'] != sha(out / 'fit_seal.json'):
        raise ValueError('complete receipt changed')
    if not 0 <= complete['cpu_core_hours'] <= 5 or not 0 < complete['wall_seconds'] <= p['wall_seconds']:
        raise ValueError('resource allowance exceeded')
    for phase in ('fit', 'evaluate'):
        execution = read(out / (phase + '_execution.json'))
        if execution['status'] != 'complete' or execution['exit_code'] != 0 or execution['peak_tree_rss_bytes'] > p['rss_bytes']:
            raise ValueError('incomplete or oversized phase')
    scores = read(out / 'scores.json')
    if set(scores['conditions']) != set(summaries) or scores['new_queries'] != 0:
        raise ValueError('scored condition/query set changed')
    rng = np.random.default_rng(p['bootstrap_seed'])
    rows = {}
    with zipfile.ZipFile(p['archive']) as archive:
        for name in p['conditions']:
            condition = Path(name).stem
            original = list(csv.DictReader(io.TextIOWrapper(archive.open(name))))
            angles = np.array([[float(r['pol_1']), float(r['pol_2'])] for r in original])
            y = np.array([float(r['vis_3']) for r in original])
            test, blocks = partition(angles)
            train = ~test
            summary = summaries[condition]
            if summary['rows'] != len(y) or summary['train_rows'] != int(train.sum()) or summary['test_rows'] != int(test.sum()):
                raise ValueError('row ledger changed')
            variance = float(y[train].var())
            if not math.isclose(variance, summary['train_variance'], rel_tol=1e-12):
                raise ValueError('training-only normalizer changed')
            # Explicitly verify command grouping independently of row order.
            folds = {}
            for angle, fold in zip(map(tuple, angles), test):
                if angle in folds and folds[angle] != bool(fold):
                    raise ValueError('duplicate command leaked across folds')
                folds[angle] = bool(fold)
            with np.load(out / condition / 'predictions.npz') as z:
                for key, value in (('y', y), ('test', test), ('blocks', blocks)):
                    if not np.array_equal(z[key], value):
                        raise ValueError('archive response/split mismatch')
                coefficients = read(out / condition / 'linear_coefficients.json')
                physics = physics_features(angles) @ np.asarray(coefficients['physics'])
                if not np.allclose(physics, z['physics'], rtol=1e-12, atol=1e-12):
                    raise ValueError('relative-angle prediction mismatch')
                model_errors = {}
                for method in ('delivery', 'rolling_buffer', 'physics', 'fourier'):
                    if z[method].shape != y.shape or not np.isfinite(z[method]).all():
                        raise ValueError('invalid prediction array')
                    error = (z[method][test] - y[test]) ** 2
                    model_errors[method] = error
                    metric = float(error.mean() / variance)
                    if not math.isclose(metric, scores['conditions'][condition]['nmse'][method], rel_tol=1e-12):
                        raise ValueError('continuous score mismatch')
                ids = blocks[test, 0] * 6 + blocks[test, 1]
                unique = np.unique(ids)
                for control in ('rolling_buffer', 'physics', 'fourier'):
                    grouped = np.array([[model_errors['delivery'][ids == k].mean(), model_errors[control][ids == k].mean()] for k in unique])
                    ratio = grouped[:, 0].mean() / max(grouped[:, 1].mean(), 1e-12)
                    sampled = grouped[rng.integers(0, len(unique), (p['bootstrap_replicates'], len(unique)))].mean(axis=1)
                    intervals = np.quantile(sampled[:, 0] / np.maximum(sampled[:, 1], 1e-12), [.025, .975])
                    reported = scores['conditions'][condition]['conditional_uncertainty'][control]
                    if reported['n_action_blocks'] != len(unique) or not math.isclose(ratio, reported['block_weighted_ratio'], rel_tol=1e-12) or not np.allclose(intervals, reported['conditional_bootstrap_ci95'], rtol=1e-12):
                        raise ValueError('conditional bootstrap mismatch')
            rows[condition] = {'train_rows': int(train.sum()), 'test_rows': int(test.sum()), 'heldout_action_blocks': len(unique),
                               'nmse': scores['conditions'][condition]['nmse'], 'cost': summary['cost']}
    comparisons = {control: {'conditions_with_lower_row_weighted_nmse': sum(v['nmse']['delivery'] < v['nmse'][control] for v in rows.values()),
                            'conditions': len(rows)} for control in ('rolling_buffer', 'physics', 'fourier')}
    # Descriptive condition counts only; no apparatus-level population inference.
    dest.mkdir(parents=True, exist_ok=True)
    copied = {}
    for file in ('protocol.json', 'launch_authorization.json', 'started.json', 'fit_seal.json', 'fit_complete.json',
                 'fit_execution.json', 'evaluate_execution.json', 'complete.json', 'scores.json'):
        target = dest / file
        if target.exists() and sha(target) != sha(out / file):
            raise ValueError('existing compact custody differs')
        shutil.copyfile(out / file, target)
        copied[file] = sha(target)
    result = {'at': utc(), 'full_acceptance': True, 'source_custody': str(out), 'artifact_hashes_verified': True,
              'archive_rows_splits_normalizers_verified': True, 'scores_bootstrap_recomputed': True,
              'selected_gate_sha256': sha(gate), 'projection_sha256': sha(projection), 'receipt_hashes': copied,
              'cpu_core_hours': complete['cpu_core_hours'], 'scope': scores['scope'], 'new_queries': 0,
              'conditions': rows, 'descriptive_comparisons': comparisons,
              'physical_boundary': 'One measured mechanism; no factorization test or independent-world significance.'}
    target = dest / 'acceptance.json'
    if target.exists():
        previous = read(target)
        if {k: v for k, v in previous.items() if k != 'at'} != {k: v for k, v in result.items() if k != 'at'}:
            raise ValueError('existing acceptance differs')
        return previous
    write(target, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('out', 'gate', 'projection', 'dest'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.out, args.gate, args.projection, args.dest)
    print({'full_acceptance': result['full_acceptance'], 'comparisons': result['descriptive_comparisons'], 'cpu_core_hours': result['cpu_core_hours']})

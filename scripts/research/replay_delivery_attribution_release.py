"""Read-only Stage A checkpoint/metric reconstruction from a relocated bundle.

Uses only the stored exposed grid, paid rows and accepted checkpoints. No fits,
simulator responses or changes to historical scores. Classical pickles require
explicit trust after independent manifest authentication. No partial replay is
reported as full acceptance. This new adapter is not a frozen historical worker.
"""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from verify_delivery_release import verify, sha


def compare(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('replayed keys changed: '+label)
        return max((compare(actual[key], value, label+'/'+key) for key, value in expected.items()), default=0.)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('replayed sequence length changed: '+label)
        return max((compare(actual[i], value, label+'/'+str(i)) for i, value in enumerate(expected)), default=0.)
    elif isinstance(expected, (float, int)):
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('replayed number differs: '+label)
        return abs(actual-expected)
    elif actual != expected:
        raise ValueError('replayed value differs: '+label)
    return 0.


FROZEN_HISTORIES = {'1091699608', '124753321', '1726880744', '27424209',
                    '520888668', '546725957', '585254818', '735595885',
                    '884825602', '921441405', '934168586', '983656467'}


def expected_labels():
    labels = {f'{rows}-{arm}-{epochs}-i{init}' for rows in ('final_buffer', 'online_admitted', 'all_paid')
              for arm in ('scm', 'flat') for epochs in (100, 30000) for init in (0, 1, 2)}
    return labels | {'all_paid-flat-matched-cpu-i0'} | {f'all_paid-{arm}-100-i0' for arm in ('ridge', 'polynomial', 'tree')}


def replay(root, expected, case=None, trusted_pickles=False):
    root = Path(root).resolve()
    integrity = verify(root, expected)
    protocol = json.loads((root/'A/protocol.json').read_text())
    for name, version in protocol['dependencies'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('recorded A dependency mismatch: '+name)
    cases = sorted(p.parent.name for p in (root/'A').glob('*/input.json'))
    if set(cases) != FROZEN_HISTORIES or (case is not None and case != cases[0]):
        raise ValueError('full twelve-history membership required; smoke is first lexical case only')
    if not trusted_pickles:
        raise ValueError('classical controls require explicit trust in authenticated original pickles')
    if any(name == 'ace' or name.startswith('ace.') for name in sys.modules):
        raise ValueError('fresh process required; archived learner imports are already cached')
    sys.path.insert(0, str(root/'source/runner'))
    import numpy as np
    import torch
    from ace.oracle import MLPSurrogate
    from ace.grid_eval import GridTruth, score
    torch.set_num_threads(1)
    cpu0, wall0 = time.process_time(), time.monotonic()
    count, maximum_numeric_difference = 0, 0.
    selected = cases if case is None else [case]
    for history in selected:
        folder = root/'A'/history
        data = json.loads((folder/'input.json').read_text())
        meta = data['meta']
        cached = json.loads((folder/'scores.json').read_text())['results']
        fits = {p.name: p for p in (folder/'fits').iterdir() if p.is_dir()}
        if set(fits) != expected_labels() or set(cached) != set(fits):
            raise ValueError('full forty-configuration history required')
        truth = GridTruth.load(root/'A/grid.npz', target=meta['target_name'], env_id=meta['env_id'])
        roots, by_id = meta['feature_names'], {r['query_index']: r for r in data['rows']}
        def predict(model, x):
            with torch.no_grad():
                return np.concatenate([model(torch.tensor(x[i:i+8192], dtype=torch.float32)).numpy()
                                       for i in range(0, len(x), 8192)])
        for label, fit in sorted(fits.items()):
            receipt = json.loads((fit/'receipt.json').read_text())
            if sha(fit/'receipt.json') != cached[label]['receipt_sha256']:
                raise ValueError('cached score receipt changed')
            diagnostics = {}
            if (fit/'models.pt').exists():
                states = torch.load(fit/'models.pt', weights_only=True, map_location='cpu')
                models = {n: MLPSurrogate.from_state_dict(s).eval() for n, s in states.items()}
                if receipt['arm'] == 'scm':
                    values = {n: truth.nodes[n] for n in roots}
                    for node, parents in meta['causal_dag'].items():
                        if not parents:
                            continue
                        true_x = np.column_stack([truth.nodes[n] for n in parents])
                        local = predict(models[node], true_x)
                        free = predict(models[node], np.column_stack([values[n] for n in parents]))
                        values[node] = free
                        retained = []
                        for i in data['selection'][receipt['row_set']]:
                            row = by_id[i]
                            if node in row['interventions']:
                                continue
                            observed = {**row['params'], **row['intermediates'], meta['target_name']: row['outcome']}
                            retained.append([float(observed[n]) for n in parents])
                        if not retained:
                            raise ValueError('empty eligible mechanism support')
                        bounds = np.asarray(retained)
                        outside = np.any((true_x < bounds.min(axis=0)) | (true_x > bounds.max(axis=0)), axis=1)
                        diagnostics[node] = {'observed_parent_mse': float(np.mean((local-truth.nodes[node])**2)),
                                             'free_running_mse': float(np.mean((free-truth.nodes[node])**2)),
                                             'propagated_prediction_shift_mse': float(np.mean((local-free)**2)),
                                             'outside_training_parent_box_fraction': float(outside.mean())}
                    prediction = values[meta['target_name']]
                else:
                    prediction = predict(models['flat'], truth.inputs(roots))
            else:
                # Same authenticated sklearn artifacts and versions as original A.
                import pickle
                with (fit/'regressor.pkl').open('rb') as stream:
                    model = pickle.load(stream)
                prediction = model['model'].predict((truth.inputs(roots)-model['lo'])/(model['hi']-model['lo']))
            levels, margins = truth.levels(), None
            if levels is not None:
                boundaries = (levels[1:]+levels[:-1])/2
                margin = np.min(np.abs(truth.y[:, None]-boundaries), axis=1)
                margins = {'fraction_prediction_error_below_true_margin': float((np.abs(prediction-truth.y) < margin).mean()),
                           'margin_min': float(margin.min()),
                           'absolute_error_quantiles': np.quantile(np.abs(prediction-truth.y), [.5, .9, .99]).tolist()}
            computed = {'receipt_sha256': sha(fit/'receipt.json'), 'score': score(prediction, truth.y, levels),
                        'node_diagnostics': diagnostics, 'margins': margins}
            maximum_numeric_difference = max(maximum_numeric_difference,
                                             compare(computed, cached[label], history+'/'+label))
            count += 1
        print(json.dumps({'event': 'history_replayed', 'history': history, 'fits': 40}), file=sys.stderr, flush=True)
    return {'integrity': integrity, 'histories_replayed': len(selected), 'fits_replayed': count,
            'full_matrix_replayed': case is None and count == 480,
            'maximum_absolute_numeric_difference': maximum_numeric_difference,
            'discrepancy_scope': 'all numeric scores, mechanism diagnostics and quantization margins',
            'comparison_tolerance': {'rtol': 1e-10, 'atol': 1e-12},
            'fit_cpu_seconds': 0, 'replay_cpu_seconds': time.process_time()-cpu0,
            'replay_wall_seconds': time.monotonic()-wall0,
            'new_optimizer_updates': 0, 'new_responses': 0,
            'scope': 'accepted A checkpoints/scores/diagnostics only; not refitting, online-comparator replay or prospective acceptance'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--smoke-first-history', action='store_true')
    parser.add_argument('--trust-original-classical-pickles', action='store_true')
    args = parser.parse_args()
    first = sorted(p.parent.name for p in (args.root/'A').glob('*/input.json'))[0] if args.smoke_first_history else None
    print(json.dumps(replay(args.root, args.expected_manifest_sha256, first,
                            args.trust_original_classical_pickles), indent=2))

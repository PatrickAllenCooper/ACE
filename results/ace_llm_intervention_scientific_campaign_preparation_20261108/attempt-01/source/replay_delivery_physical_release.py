"""Offline checkpoint inference and cached score reconstruction for Stage C.

Read-only: no optimization, new observations or output files. This is a new
release adapter, not the original frozen worker or a claim about Stage B replay.
Requires the exact C dependency versions recorded in the derived protocol.
"""
import argparse
import csv
import importlib.metadata
import io
import json
import math
from pathlib import Path
import sys
import zipfile

# Keep the verified bundle read-only, including Python's import cache writes.
sys.dont_write_bytecode = True
from verify_delivery_release import verify


def replay(root, expected):
    root = Path(root).resolve()
    integrity = verify(root, expected)
    p = json.loads((root/'C/protocol.json').read_text())
    for name, version in p['dependencies'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('recorded C dependency mismatch: '+name)
    sys.path.insert(0, str(root/'source/runner'))
    import numpy as np
    import torch
    from ace.oracle import MLPSurrogate
    torch.set_num_threads(1)
    scores = json.loads((root/'C/scores.json').read_text())
    rng = np.random.default_rng(p['bootstrap_seed'])
    maximum_shift = 0.
    with zipfile.ZipFile(root/'C/archive.zip') as archive:
        for name in p['conditions']:
            condition = Path(name).stem
            rows = list(csv.DictReader(io.TextIOWrapper(archive.open(name))))
            angles = np.array([[float(r['pol_1']), float(r['pol_2'])] for r in rows])
            y = np.array([float(r['vis_3']) for r in rows])
            blocks = np.floor((angles+90)/30).astype(int)
            test = (blocks[:, 0]+2*blocks[:, 1]) % 5 == 0
            train = ~test
            if not np.isfinite(y).all() or y[train].var() <= 0:
                raise ValueError('invalid physical response/normalizer')
            states = torch.load(root/'C'/condition/'models.pt', weights_only=True, map_location='cpu')
            coef = json.loads((root/'C'/condition/'linear_coefficients.json').read_text())
            predicted = {}
            x = torch.tensor(angles/90, dtype=torch.float32)
            # Match the frozen worker's Python scalar casts. NumPy scalar
            # mean/sd would instead promote its float32 predictions to float64.
            mean, sd = float(y[train].mean()), float(y[train].std())
            for state, method in (('delivery_mlp', 'delivery'), ('rolling_buffer', 'rolling_buffer')):
                model = MLPSurrogate.from_state_dict(states[state])
                model.eval()
                with torch.no_grad():
                    predicted[method] = model(x).numpy()*sd+mean
            radians = np.deg2rad(angles)
            predicted['physics'] = np.column_stack([np.ones(len(y)), np.cos(radians[:, 0]-radians[:, 1])**2]) @ coef['physics']
            def basis(v):
                return np.column_stack([np.ones(len(v)), np.sin(2*v), np.cos(2*v), np.sin(4*v), np.cos(4*v)])
            fourier = np.einsum('ni,nj->nij', basis(radians[:, 0]), basis(radians[:, 1])).reshape(len(y), -1)
            predicted['fourier'] = fourier @ coef['fourier']
            with np.load(root/'C'/condition/'predictions.npz', allow_pickle=False) as saved:
                for field, actual in (('y', y), ('test', test), ('blocks', blocks)):
                    if not np.array_equal(saved[field], actual):
                        raise ValueError('archive/split mismatch')
                errors = {}
                for method, actual in predicted.items():
                    if not np.isfinite(actual).all() or not np.allclose(actual, saved[method], rtol=1e-10, atol=1e-10):
                        raise ValueError('checkpoint/linear prediction replay mismatch: '+condition+'/'+method)
                    maximum_shift = max(maximum_shift, float(np.max(np.abs(actual-saved[method]))))
                    errors[method] = (actual[test]-y[test])**2
                    nmse = float(errors[method].mean()/y[train].var())
                    if not math.isclose(nmse, scores['conditions'][condition]['nmse'][method], rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError('replayed NMSE mismatch')
                ids = blocks[test, 0]*6+blocks[test, 1]
                unique = np.unique(ids)
                for control in ('rolling_buffer', 'physics', 'fourier'):
                    grouped = np.array([[errors['delivery'][ids == k].mean(), errors[control][ids == k].mean()] for k in unique])
                    sampled = grouped[rng.integers(0, len(unique), (p['bootstrap_replicates'], len(unique)))].mean(axis=1)
                    ci = np.quantile(sampled[:, 0]/np.maximum(sampled[:, 1], 1e-12), [.025, .975])
                    ratio = grouped[:, 0].mean()/max(grouped[:, 1].mean(), 1e-12)
                    reported = scores['conditions'][condition]['conditional_uncertainty'][control]
                    if (reported['n_action_blocks'] != len(unique) or
                            not math.isclose(ratio, reported['block_weighted_ratio'], rel_tol=1e-10) or
                            not np.allclose(ci, reported['conditional_bootstrap_ci95'], rtol=1e-10, atol=1e-12)):
                        raise ValueError('replayed conditional bootstrap mismatch')
    return {'integrity': integrity, 'physical_conditions_replayed': len(p['conditions']),
            'neural_checkpoints_replayed': 2*len(p['conditions']),
            'linear_coefficient_predictions_replayed': 2*len(p['conditions']),
            'scores_and_conditional_bootstrap_recomputed': True,
            'maximum_cached_prediction_difference': maximum_shift,
            'new_optimization_updates': 0, 'new_responses': 0,
            'scope': 'C checkpoint/linear inference and conditional score reconstruction only; not refitting or Stage B acceptance'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.root, args.expected_manifest_sha256), indent=2))

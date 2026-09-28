#!/usr/bin/env python3
"""Score a frozen forecast in a process with private NeuronBench targets."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from neuronbench_custody_smoke import SOURCE_FILES, sha


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--source-hashes', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    receipt_path = a.output / 'oracle_complete.json'
    receipt = json.loads(receipt_path.read_text())
    hashes = json.loads(a.source_hashes.read_text())['upstream_source_sha256']
    if receipt['upstream_source_sha256'] != hashes or receipt['total_cost'] != 4:
        raise ValueError('oracle receipt mismatch')
    for name in SOURCE_FILES:
        if sha(a.upstream / 'neuronbench' / name) != hashes[name]:
            raise ValueError(f'upstream source mismatch: {name}')
    public = a.output / 'public'
    for name, digest in receipt['public_files'].items():
        if sha(public / name) != digest:
            raise ValueError(f'public file changed: {name}')
    problem = json.loads((public / 'problem.json').read_text())
    prediction_path = a.output / 'predictions.json'
    predictions = json.loads(prediction_path.read_text())
    labels = problem['test_protocol_labels']
    values = predictions['predicted_spike_counts']
    if len(labels) != 6 or set(values) != set(labels) or any(
            not math.isfinite(float(v)) or float(v) < 0 for v in values.values()):
        raise ValueError('invalid or incomplete predictions')
    if predictions['problem_sha256'] != sha(public / 'problem.json') or predictions[
            'observations_sha256'] != sha(public / 'observations.json') or predictions[
            'environment_samples_used'] != 4 or predictions['observations_used'] != 4:
        raise ValueError('prediction provenance or budget mismatch')
    sys.path.insert(0, str(a.upstream))
    import neuronbench as nb
    targets = nb.evaluator.held_out_targets(receipt['world'], stochastic=False,
                                            seed=receipt['seed'])
    if set(targets) != set(labels):
        raise ValueError('private target label mismatch')
    mse = nb.evaluator.forecast_mse(values, targets)
    private = a.output / 'private'
    private.mkdir(parents=True, exist_ok=True)
    (private / 'targets.json').write_text(json.dumps(targets, indent=2) + '\n')
    (private / 'score.json').write_text(json.dumps({
        'floored_spike_forecast_mse': mse, 'heldout_labels': len(labels),
        'floor': float(nb.evaluator.MSE_FLOOR), 'budget': 4}, indent=2) + '\n')
    complete = dict(receipt)
    complete['oracle_receipt_sha256'] = sha(receipt_path)
    complete['predictions_sha256'] = sha(prediction_path)
    complete['private_files'] = {path.name: sha(path) for path in sorted(private.iterdir())}
    (a.output / 'complete.json').write_text(json.dumps(complete, indent=2) + '\n')
    print('floored spike forecast MSE', mse)


if __name__ == '__main__':
    main()

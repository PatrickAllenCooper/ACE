#!/usr/bin/env python3
"""Private scorer for frozen public-only BoxingGym signal forecasts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--methods', nargs='+', default=[
        'rbf_kernel', 'gaussian_three_source', 'privileged_inverse_quadratic'])
    a = p.parse_args()
    receipt = json.loads((a.fit / 'complete.json').read_text())
    pred_path = a.fit / 'predictions.csv'
    forecast_count = receipt.get('forecast_questions', receipt.get('heldout_questions'))
    if receipt['private_files_read'] != 0 or receipt['training_queries'] != 16 or receipt[
            'closed_model_calls'] != 0 or forecast_count != 32 or sha(pred_path) != receipt[
            'predictions_sha256']:
        raise ValueError('fit receipt mismatch')
    if sha(a.public / 'observations.csv') != receipt['observations_sha256'] or sha(
            a.public / 'forecast_questions.csv') != receipt['forecast_questions_sha256']:
        raise ValueError('public source changed')
    q = read(a.public / 'forecast_questions.csv')
    h = read(a.private / 'holdout.csv')
    predictions = read(pred_path)
    if len(q) != len(h) or len(h) != len(predictions) or len(h) != 32:
        raise ValueError('forecast count mismatch')
    methods = tuple(a.methods)
    y = np.array([float(row['response']) for row in h])
    if not np.isfinite(y).all():
        raise ValueError('nonfinite targets')
    for i, (question, target, predicted) in enumerate(zip(q, h, predictions)):
        if int(predicted['index']) != i or any(not math.isclose(float(question[key]),
                float(row[key]), rel_tol=0, abs_tol=1e-12) for key in ('x1', 'x2')
                for row in (target, predicted)):
            raise ValueError('question/target/prediction coordinate mismatch')
    rows = []
    for name in methods:
        values = np.array([float(row[name]) for row in predictions])
        if not np.isfinite(values).all():
            raise ValueError('nonfinite prediction')
        error = np.abs(values - y)
        rows.append({'method': name, 'mae': float(error.mean()),
                     'rmse': float(np.sqrt(np.mean(error ** 2))),
                     'median_ae': float(np.median(error)), 'heldout_queries': len(y)})
    score_path = a.fit / 'scores.csv'
    with score_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scored = {'fit_receipt_sha256': sha(a.fit / 'complete.json'),
              'predictions_sha256': sha(pred_path),
              'private_holdout_sha256': sha(a.private / 'holdout.csv'),
              'scores_sha256': sha(score_path), 'heldout_queries': 32}
    (a.fit / 'scored_complete.json').write_text(json.dumps(scored, indent=2) + '\n')
    print([(row['method'], round(row['mae'], 4)) for row in rows])


if __name__ == '__main__':
    main()

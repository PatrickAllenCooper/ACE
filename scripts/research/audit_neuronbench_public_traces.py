#!/usr/bin/env python3
"""Check whether archived public traces support a dynamics-based forecaster."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from neuronbench_public_timing import current_and_test_start


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--v1', type=Path, required=True)
    parser.add_argument('--v2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = [args.v1 / 'z_rebound', *(args.v2 / w for w in
              ('h_sag', 'na_fatigue', 'ca_rebound', 'd_type', 'textbook_M'))]
    cells = []
    for root in roots:
        for arm in ('random', 'coverage'):
            public = root / arm / 'public'
            observations = public / 'observations.json'
            rows = json.loads(observations.read_text())
            problem = public / 'problem.json'
            lookup = dict(json.loads(problem.read_text())['protocols'])
            if len(rows) != 4:
                raise ValueError(f'expected four public observations: {public}')
            traces = []
            for row in rows:
                path = public / row['trace_file']
                with np.load(path) as data:
                    voltage = data['voltage']
                    indices = data['obs_idx']
                if len(voltage) != row['trace_points'] or len(indices) != len(voltage):
                    raise ValueError(f'trace length mismatch: {path}')
                if not np.isfinite(voltage).all() or not np.array_equal(
                        np.diff(indices), np.full(len(indices) - 1, 10)):
                    raise ValueError(f'nonfinite or nonuniform trace: {path}')
                start = int(row['test_start'])
                if row['protocol_label'] not in lookup:
                    raise ValueError(f'unknown public protocol: {path}')
                current, expected_start = current_and_test_start(lookup[row['protocol_label']])
                if len(current) != len(voltage) or expected_start != start:
                    raise ValueError(f'public stimulus reconstruction mismatch: {path}')
                if not 0 <= start < len(voltage) - 1:
                    raise ValueError(f'invalid test start: {path}')
                crossing_count = int(np.sum((voltage[start:-1] < 0) &
                                            (voltage[start + 1:] >= 0)))
                if crossing_count != int(row['spike_count']):
                    raise ValueError(f'public spike-count mismatch: {path}')
                traces.append({'protocol_label': row['protocol_label'],
                               'trace_file': path.name, 'trace_sha256': sha(path),
                               'trace_points': len(voltage), 'index_stride': 10,
                               'test_start': start, 'zero_crossings_after_start': crossing_count,
                               'stimulus_points': len(current)})
            cells.append({'world': root.name, 'arm': arm,
                          'problem_sha256': sha(problem),
                          'observations_sha256': sha(observations), 'traces': traces})
    # Same acquired actions and counts can still carry distinct subthreshold
    # dynamics. This checks information availability, not forecast accuracy.
    roots_by_world = {root.name: root for root in roots}
    a, b = (roots_by_world[w] / 'random' / 'public' for w in ('h_sag', 'ca_rebound'))
    ra, rb = (json.loads((p / 'observations.json').read_text()) for p in (a, b))
    va = {row['protocol_label']: np.load(a / row['trace_file'])['voltage'] for row in ra}
    vb = {row['protocol_label']: np.load(b / row['trace_file'])['voltage'] for row in rb}
    ca = {row['protocol_label']: row['spike_count'] for row in ra}
    cb = {row['protocol_label']: row['spike_count'] for row in rb}
    if set(va) != set(vb) or ca != cb:
        raise ValueError('paired trace comparison requires identical actions and counts')
    pair = {label: float(np.sqrt(np.mean((va[label] - vb[label]) ** 2)))
            for label in sorted(va)}
    result = {'public_trace_cells': len(cells), 'public_traces': sum(len(c['traces']) for c in cells),
              'private_files_read': 0, 'closed_model_calls': 0, 'cells': cells,
              'same_count_world_pair': ['h_sag', 'ca_rebound'],
              'same_count_trace_rmse_by_protocol': pair}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('validated', result['public_traces'], 'public voltage traces')


if __name__ == '__main__':
    main()

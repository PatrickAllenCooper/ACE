#!/usr/bin/env python3
"""Oracle-only custody smoke for pinned NeuronBench, without solver access."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


SOURCE_FILES = ('__init__.py', 'worlds.py', 'protocols.py', 'evaluator.py',
                'features.py', 'stochastic.py')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', required=True, type=Path)
    p.add_argument('--protocol', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--source-revision', required=True)
    a = p.parse_args()
    spec = json.loads(a.protocol.read_text())
    root = a.upstream / 'neuronbench'
    assert set(spec['upstream_source_sha256']) == set(SOURCE_FILES)
    for name in SOURCE_FILES:
        if sha(root / name) != spec['upstream_source_sha256'][name]:
            raise ValueError(f'upstream source mismatch: {name}')
    if spec['stochastic'] or spec['budget'] != 2:
        raise ValueError('only frozen deterministic two-action smoke supported')
    sys.path.insert(0, str(a.upstream))
    import neuronbench as nb
    world = nb.load_world(spec['world'], stochastic=False, seed=spec['seed'])
    problem = world.problem()
    if set(problem) != {'text_prior', 'reference_model', 'protocols',
                        'test_protocol_labels', 'budget_rule'}:
        raise ValueError('public problem contract changed')
    pool = problem['protocols']
    if len(pool) != 9 or len({lab for lab, _ in pool}) != 9:
        raise ValueError('unexpected public action pool')
    public = a.output / 'public'
    public.mkdir(parents=True, exist_ok=True)
    (public / 'problem.json').write_text(json.dumps(problem, indent=2) + '\n')
    observations = []
    spent = 0
    used = set()
    for index, action in enumerate(pool[:spec['budget']]):
        label = action[0]
        if label in used or action not in pool:
            raise ValueError('illegal or repeated action')
        obs = world.run(action, reps=1)
        if obs.protocol_label != label or obs.cost != 1 or obs.reps != 1:
            raise ValueError('unexpected oracle cost or response')
        trace = np.asarray(obs.voltage, dtype=float)
        indices = np.asarray(obs.obs_idx, dtype=int)
        if not len(trace) or len(trace) != len(indices) or not np.isfinite(trace).all():
            raise ValueError('invalid partial trace')
        filename = f'observation_{index}.npz'
        np.savez_compressed(public / filename, voltage=trace, obs_idx=indices)
        observations.append({'protocol_label': label, 'spike_count': obs.spike_count,
                             'cost': obs.cost, 'reps': obs.reps,
                             'test_start': obs.test_start, 'trace_file': filename,
                             'trace_points': len(trace)})
        spent += obs.cost
        used.add(label)
    if spent != spec['budget'] or len(used) != spec['budget']:
        raise ValueError('budget accounting failed')
    (public / 'observations.json').write_text(json.dumps(observations, indent=2) + '\n')
    receipt = {'source_revision': a.source_revision,
               'upstream_revision': spec['upstream_revision'],
               'upstream_source_sha256': spec['upstream_source_sha256'],
               'action_count': len(observations), 'unique_actions': len(used),
               'total_cost': spent, 'budget': spec['budget'],
               'closed_model_calls': 0, 'public_files': {}}
    for path in sorted(public.iterdir()):
        receipt['public_files'][path.name] = sha(path)
    (a.output / 'oracle_complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('validated actions', len(observations), 'cost', spent, 'API calls', 0)


if __name__ == '__main__':
    main()

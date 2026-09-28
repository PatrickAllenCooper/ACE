#!/usr/bin/env python3
"""Execute a frozen public plan behind the NeuronBench oracle boundary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from neuronbench_custody_smoke import SOURCE_FILES, sha


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--source-hashes', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--world', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-revision', required=True)
    a = p.parse_args()
    hashes = json.loads(a.source_hashes.read_text())['upstream_source_sha256']
    for name in SOURCE_FILES:
        if sha(a.upstream / 'neuronbench' / name) != hashes[name]:
            raise ValueError(f'upstream source mismatch: {name}')
    plan = json.loads(a.plan.read_text())
    sys.path.insert(0, str(a.upstream))
    import neuronbench as nb
    world = nb.load_world(a.world, stochastic=False, seed=a.seed)
    problem = world.problem()
    if [lab for lab, _ in world.test_protocols] != problem['test_protocol_labels']:
        raise ValueError('heldout public specification mismatch')
    problem['forecast_protocols'] = world.test_protocols
    public = a.output / 'public'
    public.mkdir(parents=True, exist_ok=True)
    problem_path = public / 'problem.json'
    problem_path.write_text(json.dumps(problem, indent=2) + '\n')
    if sha(problem_path) != plan['problem_sha256']:
        raise ValueError('frozen public problem hash mismatch')
    budget = plan['budget']
    indices = plan['indices']
    if len(indices) != budget or len(set(indices)) != budget or len(plan['actions']) != budget:
        raise ValueError('invalid plan length or repeated action')
    pool = problem['protocols']
    if any(not isinstance(i, int) or i < 0 or i >= len(pool) or pool[i] != action
           for i, action in zip(indices, plan['actions'])):
        raise ValueError('frozen plan does not match public action pool')
    observations = []
    for step, action in enumerate(plan['actions']):
        obs = world.run(action, reps=1)
        if obs.protocol_label != action[0] or obs.cost != 1 or obs.reps != 1:
            raise ValueError('unexpected oracle cost or response')
        voltage = np.asarray(obs.voltage, dtype=float)
        obs_idx = np.asarray(obs.obs_idx, dtype=int)
        if not len(voltage) or len(voltage) != len(obs_idx) or not np.isfinite(voltage).all():
            raise ValueError('invalid partial trace')
        trace_file = f'observation_{step}.npz'
        np.savez_compressed(public / trace_file, voltage=voltage, obs_idx=obs_idx)
        observations.append({'protocol_label': action[0], 'spike_count': obs.spike_count,
                             'cost': obs.cost, 'reps': obs.reps, 'test_start': obs.test_start,
                             'trace_file': trace_file, 'trace_points': len(voltage)})
    if sum(row['cost'] for row in observations) != budget:
        raise ValueError('budget exceeded or incomplete')
    (public / 'observations.json').write_text(json.dumps(observations, indent=2) + '\n')
    receipt = {'source_revision': a.source_revision, 'world': a.world, 'seed': a.seed,
               'upstream_revision': 'c354622458c460b419cab821d482c879f0578377',
               'upstream_source_sha256': hashes, 'plan_sha256': sha(a.plan),
               'action_count': budget, 'unique_actions': budget, 'total_cost': budget,
               'closed_model_calls': 0,
               'public_files': {path.name: sha(path) for path in sorted(public.iterdir())}}
    (a.output / 'oracle_complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('counted oracle actions', budget)


if __name__ == '__main__':
    main()

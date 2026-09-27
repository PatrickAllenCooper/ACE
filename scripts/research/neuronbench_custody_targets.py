#!/usr/bin/env python3
"""Separate private target writer for the NeuronBench custody smoke."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neuronbench_custody_smoke import SOURCE_FILES, sha


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', required=True, type=Path)
    p.add_argument('--protocol', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    spec = json.loads(a.protocol.read_text())
    oracle = json.loads((a.output / 'oracle_complete.json').read_text())
    if oracle['total_cost'] != spec['budget'] or oracle['unique_actions'] != spec['budget']:
        raise ValueError('oracle budget invalid')
    for name in SOURCE_FILES:
        if sha(a.upstream / 'neuronbench' / name) != spec['upstream_source_sha256'][name]:
            raise ValueError(f'upstream source mismatch: {name}')
    for name, digest in oracle['public_files'].items():
        if sha(a.output / 'public' / name) != digest:
            raise ValueError(f'public artifact mismatch: {name}')
    sys.path.insert(0, str(a.upstream))
    import neuronbench as nb
    targets = nb.evaluator.held_out_targets(spec['world'], stochastic=False,
                                            seed=spec['seed'])
    if set(targets) != set(json.loads((a.output / 'public/problem.json').read_text())[
            'test_protocol_labels']):
        raise ValueError('held-out label mismatch')
    private = a.output / 'private'
    private.mkdir(parents=True, exist_ok=True)
    target_path = private / 'targets.json'
    target_path.write_text(json.dumps({'world': spec['world'], 'seed': spec['seed'],
                                       'spike_targets': targets}, indent=2) + '\n')
    complete = dict(oracle)
    complete['heldout_labels'] = len(targets)
    complete['private_files'] = {'targets.json': sha(target_path)}
    complete['oracle_receipt_sha256'] = sha(a.output / 'oracle_complete.json')
    (a.output / 'complete.json').write_text(json.dumps(complete, indent=2) + '\n')
    print('private held-out labels', len(targets))


if __name__ == '__main__':
    main()

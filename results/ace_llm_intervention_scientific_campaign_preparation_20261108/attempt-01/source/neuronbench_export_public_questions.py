#!/usr/bin/env python3
"""Export only public acquisition and forecast questions from pinned NeuronBench."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neuronbench_custody_smoke import SOURCE_FILES, sha


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--source-hashes', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-revision', required=True)
    a = p.parse_args()
    frozen = json.loads(a.protocol.read_text())
    if frozen['mode'] != 'deterministic' or frozen['world_seed'] != 42:
        raise ValueError('unsupported frozen protocol')
    hashes = json.loads(a.source_hashes.read_text())['upstream_source_sha256']
    for name in SOURCE_FILES:
        if sha(a.upstream / 'neuronbench' / name) != hashes[name]:
            raise ValueError(f'upstream source mismatch: {name}')
    sys.path.insert(0, str(a.upstream))
    import neuronbench as nb
    manifest = {'source_revision': a.source_revision,
                'upstream_revision': frozen['upstream_revision'],
                'protocol_sha256': sha(a.protocol),
                'upstream_source_sha256': hashes, 'worlds': {},
                'oracle_actions': 0, 'closed_model_calls': 0}
    for name in frozen['worlds']:
        world = nb.load_world(name, stochastic=False, seed=frozen['world_seed'])
        problem = world.problem()
        if set(problem) != {'text_prior', 'reference_model', 'protocols',
                            'test_protocol_labels', 'budget_rule'}:
            raise ValueError('public problem contract changed')
        forecast = world.test_protocols
        if [lab for lab, _ in forecast] != problem['test_protocol_labels']:
            raise ValueError('forecast label mismatch')
        if len(problem['protocols']) != 9 or len(forecast) != 6:
            raise ValueError('unexpected action or forecast pool')
        overlap = sorted(set(lab for lab, _ in problem['protocols']) &
                         set(problem['test_protocol_labels']))
        problem['forecast_protocols'] = forecast
        path = a.output / name / 'problem.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(problem, indent=2) + '\n')
        manifest['worlds'][name] = {'problem_sha256': sha(path),
                                    'acquisition_actions': 9, 'forecast_labels': 6,
                                    'overlap_labels': overlap}
    (a.output / 'export_complete.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('public question sets', len(manifest['worlds']), 'oracle actions', 0)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Numerical-only custody smoke for pinned BoxingGym signal localization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_environment(upstream: Path):
    source = upstream / 'src/boxing_gym/envs/location_finding.py'
    if not source.is_file():
        raise FileNotFoundError(source)
    # Neither optional dependency is used by Signal generation/querying. Fail
    # loudly if the unused Bayesian design or reporting paths are invoked.
    if importlib.util.find_spec('pymc') is None:
        module = ModuleType('pymc')
        sys.modules['pymc'] = module
    helper = ModuleType('boxing_gym.agents.box_loop_helper')
    def unused_reporting_helper(*_args, **_kwargs):
        raise RuntimeError('unused BoxingGym reporting helper invoked')
    helper.construct_dataframe = unused_reporting_helper
    sys.modules[helper.__name__] = helper
    sys.path.insert(0, str(upstream / 'src'))
    from boxing_gym.envs.location_finding import Signal
    return Signal, source


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-revision', required=True)
    a = p.parse_args()
    spec = json.loads(a.protocol.read_text())
    if a.seed not in spec['world_seeds']:
        raise ValueError('seed outside frozen protocol')
    cls, source = load_environment(a.upstream)
    if sha(source) != spec['upstream_simulator_sha256']:
        raise ValueError('upstream simulator hash mismatch')
    np.random.seed(a.seed)
    env = cls()
    if env.num_sources != 3 or env.dim != 2:
        raise ValueError('unexpected upstream world dimensions')
    descriptive = env.generate_system_message(True, 'Predict the signal intensity.')
    anonymous = env.generate_system_message(False, 'Predict the numeric response.')
    query_rng = np.random.default_rng(a.seed + 700001)
    test_rng = np.random.default_rng(a.seed + 800001)
    query_points = query_rng.uniform(-2, 2, (16, 2))
    test_points = test_rng.normal(0, 1, (32, 2))
    observed = []
    for x in query_points:
        response, ok = env.run_experiment(json.dumps(x.tolist()))
        if not ok or not np.isfinite(response):
            raise ValueError('invalid upstream query')
        observed.append({'x1': float(x[0]), 'x2': float(x[1]), 'response': response})
    if len(env.get_data()) != 16:
        raise ValueError('public query count mismatch')
    heldout = []
    questions = []
    for x in test_points:
        response = float(env.step(x))
        if not np.isfinite(response):
            raise ValueError('nonfinite heldout response')
        questions.append({'x1': float(x[0]), 'x2': float(x[1])})
        heldout.append({'x1': float(x[0]), 'x2': float(x[1]), 'response': response})
    if len(env.get_data()) != 16:
        raise ValueError('private calls entered public query history')
    public = a.output / 'public'
    private = a.output / 'private'
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    write_csv(public / 'observations.csv', observed)
    write_csv(public / 'forecast_questions.csv', questions)
    (public / 'descriptive_message.txt').write_text(descriptive)
    (public / 'anonymous_message.txt').write_text(anonymous)
    write_csv(private / 'holdout.csv', heldout)
    (private / 'sources.json').write_text(json.dumps(np.asarray(env.true_theta).tolist(), indent=2) + '\n')
    receipt = {'kind': 'boxing_signal_custody_smoke', 'seed': a.seed,
               'source_revision': a.source_revision,
               'upstream_revision': spec['upstream_revision'],
               'upstream_simulator_sha256': sha(source),
               'protocol_sha256': sha(a.protocol),
               'query_count': 16, 'heldout_count': 32,
               'closed_model_calls': 0,
               'public_files': {path.name: sha(path) for path in sorted(public.iterdir())},
               'private_files': {path.name: sha(path) for path in sorted(private.iterdir())}}
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('BoxingGym signal custody: 16 public queries, 32 heldout, API calls 0')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Numerical-only custody smoke for pinned upstream BoxingGym Lotka–Volterra.

The optional Box's Loop reporting helper imports ArviZ at module load time.
When ArviZ is unavailable, replace only that unused helper with a function that
raises if called. The upstream simulator, step function, and query API are not
modified. No language-model or network API is imported or invoked.
"""
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
    source = upstream / 'src' / 'boxing_gym' / 'envs' / 'lotka_volterra.py'
    if not source.is_file():
        raise FileNotFoundError(source)
    if importlib.util.find_spec('arviz') is None:
        helper = ModuleType('boxing_gym.agents.box_loop_helper')

        def unused_reporting_helper(*_args, **_kwargs):
            raise RuntimeError('BoxingGym reporting helper is unavailable in this minimal runtime')

        helper.construct_dataframe = unused_reporting_helper
        sys.modules[helper.__name__] = helper
    sys.path.insert(0, str(upstream / 'src'))
    from boxing_gym.envs.lotka_volterra import LotkaVolterra
    return LotkaVolterra, source


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', required=True, type=Path)
    p.add_argument('--upstream-revision', required=True)
    p.add_argument('--expected-source-sha256', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--observations', type=int, default=8)
    p.add_argument('--holdout', type=int, default=16)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if (a.observations < 1 or a.holdout < 1 or len(a.upstream_revision) != 40
            or len(a.expected_source_sha256) != 64):
        p.error('positive sample counts, full upstream revision, and source SHA-256 required')
    cls, source = load_environment(a.upstream)
    if sha(source) != a.expected_source_sha256:
        raise ValueError('Upstream simulator hash differs from frozen protocol')
    np.random.seed(a.seed)
    env = cls()
    public_message = env.generate_system_message(True, 'Predict both observed populations.')
    anonymous_message = env.generate_system_message(False, 'Predict both observed responses.')
    low, high = env.lower_limit, env.upper_limit
    query_rng = np.random.default_rng(a.seed + 700001)
    test_rng = np.random.default_rng(a.seed + 800001)
    query_times = query_rng.uniform(low, high, a.observations)
    test_times = test_rng.uniform(low, high, a.holdout)
    observed = []
    for time in query_times:
        response, ok = env.run_experiment(str(float(time)))
        assert ok and len(response) == 2
        observed.append({'time': float(time), 'response_1': response[0],
                         'response_2': response[1]})
    assert len(env.get_data()) == a.observations
    heldout = []
    for time in test_times:
        response = env.step(float(time))
        assert len(response) == 2
        heldout.append({'time': float(time), 'response_1': response[0],
                        'response_2': response[1]})
    assert len(env.get_data()) == a.observations  # held-out calls do not enter training history
    assert not set(query_times).intersection(test_times)
    public = a.output / 'public'
    private = a.output / 'private'
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    write_csv(public / 'observations.csv', observed)
    write_csv(private / 'holdout.csv', heldout)
    (public / 'descriptive_message.txt').write_text(public_message)
    (public / 'anonymous_message.txt').write_text(anonymous_message)
    parameter_bytes = json.dumps([env.alpha, env.beta, env.gamma, env.delta],
                                 separators=(',', ':')).encode()
    receipt = {'kind': 'boxing_lotka_numerical_smoke', 'seed': a.seed,
               'upstream_revision': a.upstream_revision,
               'upstream_simulator_sha256': a.expected_source_sha256,
               'query_count': len(observed), 'heldout_count': len(heldout),
               'parameter_sha256': hashlib.sha256(parameter_bytes).hexdigest(),
               'public_files': {}, 'private_files': {},
               'closed_model_calls': 0}
    for name in ('observations.csv', 'descriptive_message.txt', 'anonymous_message.txt'):
        receipt['public_files'][name] = sha(public / name)
    receipt['private_files']['holdout.csv'] = sha(private / 'holdout.csv')
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('validated upstream simulator', a.upstream_revision,
          'queries', len(observed), 'heldout', len(heldout),
          'API calls', receipt['closed_model_calls'])


if __name__ == '__main__':
    main()

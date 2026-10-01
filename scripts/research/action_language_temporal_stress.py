#!/usr/bin/env python3
"""Later-authored action-description stress fixture for the frozen rule parser."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from action_language_fixture import ACTUATORS, COMMON, legal_actions


CASES = (
    ('joint_forbidden', 1, 4, ['danger'], (
        'The controller can actuate one channel per trial. The danger lead is disconnected. Maximum total charge: 4 credits.',
        'A trial may set exactly one of x, z, or proxy_p. Danger is locked out. Spend no more than 4 credits.',
    )),
    ('cost_limited', 2, 3, ['danger'], (
        'Pairs are allowed, but a trial has a budget of three credits. The danger lead is disabled.',
        'No more than two controls may fire. Combined expense cannot exceed 3; exclude danger.',
    )),
    ('hazard_excluded', 2, 4, ['danger'], (
        'Two controls may be fired together at a total cost of four or less. The danger channel is locked out.',
        'The danger actuator is out of service. A run may include at most two other channels and use no more than 4 credits.',
    )),
    ('proxy_only', 2, 4, ['x', 'z', 'danger'], (
        'Only the proxy_p line is connected. Direct x and z lines plus danger are disconnected. Maximum expense is four credits.',
        'The bridge channel proxy_p alone is operable; all others are isolated. Spend no more than 4 per trial.',
    )),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    public, answers = [], []
    for name, joint, cap, excluded, descriptions in CASES:
        schema = {'actuators': ACTUATORS,
                  'allowed_values': {key: [-1, 1] for key in ACTUATORS},
                  'excluded_actuators': excluded,
                  'max_joint_targets': joint, 'cost_cap': cap}
        legal = legal_actions(schema)
        for index, text in enumerate(descriptions):
            task_id = f'{name}_stress_{index}'
            public.append({'id': task_id, 'description': COMMON + ' ' + text,
                           'requested_output': 'JSON list of legal actions with targets and values'})
            answers.append({'id': task_id, 'schema': schema,
                            'legal_actions': legal, 'legal_action_count': len(legal)})
    args.output.mkdir(parents=True)
    for filename, rows in (('prompts.jsonl', public), ('answer_key.jsonl', answers)):
        (args.output / filename).write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    hashes = {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
              for name in ('prompts.jsonl', 'answer_key.jsonl')}
    (args.output / 'complete.json').write_text(json.dumps({
        'source_revision': revision, 'tasks': len(public), 'distinct_schemas': len(CASES),
        'legal_action_counts': [len(legal_actions({
            'actuators': ACTUATORS, 'allowed_values': {key: [-1, 1] for key in ACTUATORS},
            'excluded_actuators': excluded, 'max_joint_targets': joint, 'cost_cap': cap}))
            for _, joint, cap, excluded, _ in CASES],
        'file_sha256': hashes, 'simulator_queries': 0, 'model_calls': 0,
    }, indent=2, sort_keys=True) + '\n')
    print(f'validated {len(public)} later-authored descriptions across {len(CASES)} schemas')


if __name__ == '__main__':
    main()

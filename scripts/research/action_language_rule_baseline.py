#!/usr/bin/env python3
"""Hand-written text-rule control for the tiny action-language fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from action_language_fixture import ACTUATORS, legal_actions


def parse_schema(description: str) -> dict:
    lower=description.lower()
    if 'only proxy_p' in lower or 'sole usable actuator' in lower:
        excluded=['x','z','danger']
    elif 'danger' in lower and any(word in lower for word in ('prohibited','excluded','do not touch','never use')):
        excluded=['danger']
    else:
        excluded=[]
    joint=1 if ('at most one actuator' in lower or 'single-actuator experiments only' in lower) else 2
    patterns=(r'cost limit is (\d+)',r'spend no more than (\d+)',
              r'single experiment may spend at most (\d+)',
              r'cost ceiling of (\d+)',r'cost is capped at (\d+)',r'cost limit: (\d+)')
    matches=[int(m.group(1)) for pattern in patterns for m in re.finditer(pattern,lower)]
    if len(matches)!=1:
        raise ValueError('ambiguous or absent action cost limit')
    return {'actuators':ACTUATORS,'allowed_values':{key:[-1,1] for key in ACTUATORS},
            'excluded_actuators':excluded,'max_joint_targets':joint,'cost_cap':matches[0]}


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION')!=revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    prompts=[json.loads(line) for line in (a.fixture/'prompts.jsonl').read_text().splitlines()]
    key={row['id']:row for row in (json.loads(line) for line in (a.fixture/'answer_key.jsonl').read_text().splitlines())}
    if len(prompts)!=8 or len(key)!=8:
        raise ValueError('unexpected fixture')
    rows=[]
    for task in prompts:
        schema=parse_schema(task['description'])
        legal=legal_actions(schema)
        correct=key[task['id']]
        rows.append({'id':task['id'],'parsed_schema':schema,
                     'predicted_actions':legal,
                     'exact_schema':schema==correct['schema'],
                     'exact_legal_menu':legal==correct['legal_actions']})
    a.output.mkdir(parents=True)
    payload=''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows)
    (a.output/'predictions.jsonl').write_text(payload)
    (a.output/'complete.json').write_text(json.dumps({
        'source_revision':revision,'fixture_complete_sha256':hashlib.sha256((a.fixture/'complete.json').read_bytes()).hexdigest(),
        'tasks':len(rows),'exact_schemas':sum(row['exact_schema'] for row in rows),
        'exact_legal_menus':sum(row['exact_legal_menu'] for row in rows),
        'predictions_sha256':hashlib.sha256(payload.encode()).hexdigest(),
        'closed_model_calls':0,'simulator_queries':0},sort_keys=True,indent=2)+'\n')
    print('rule baseline exact legal menus',sum(row['exact_legal_menu'] for row in rows),'/',len(rows))


if __name__=='__main__':
    main()

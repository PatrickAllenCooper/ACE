#!/usr/bin/env python3
"""Small paired text/schema actuator-menu fixture; no model is evaluated."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path

from action_menu_validator_smoke import validate


ACTUATORS = {
    "x": {"cost": 2, "affects": ["x"]},
    "z": {"cost": 2, "affects": ["z"]},
    "proxy_p": {"cost": 3, "affects": ["x", "z"]},
    "danger": {"cost": 1, "affects": ["x"]},
}
COMMON = "Actuators x and z each cost 2 per use. Proxy p (proxy_p) costs 3 and jointly affects x and z. Danger costs 1 and affects x. Each selected actuator must be set to either -1 or +1."
FIXTURES = (
    ("joint_forbidden",1,4,["danger"],(
        "At most one actuator may be operated at a time. Never use danger. The action cost limit is 4.",
        "Single-actuator experiments only; danger is prohibited. Spend no more than 4 on one experiment.",
    ),6),
    ("cost_limited",2,3,["danger"],(
        "Up to two actuators may be operated together, but the action cost limit is 3. Never use danger.",
        "Pairs are permitted in principle. A single experiment may spend at most 3, and danger is excluded.",
    ),6),
    ("hazard_excluded",2,4,["danger"],(
        "At most two actuators may be combined. Spend no more than 4. Danger is unsafe and prohibited.",
        "Joint interventions of size two are allowed under a cost ceiling of 4; do not touch danger.",
    ),10),
    ("proxy_only",2,4,["x","z","danger"],(
        "Only proxy_p can be operated; x, z, and danger are locked. Two-actuator actions are otherwise permitted and cost is capped at 4.",
        "The direct x and z actuators and danger are unavailable. Proxy p is the sole usable actuator. Cost limit: 4.",
    ),2),
)


def legal_actions(schema: dict) -> list[dict]:
    names = list(schema["actuators"])
    actions=[]
    for size in (1,2):
        for targets in itertools.combinations(names,size):
            for values in itertools.product((-1,1),repeat=size):
                proposal={"targets":list(targets),"values":list(values)}
                if validate(schema,proposal)==(True,"valid"):
                    actions.append(proposal)
    return actions


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION')!=revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    public=[];answer=[]
    for name,joint,cap,excluded,descriptions,expected in FIXTURES:
        schema={"actuators":ACTUATORS,"allowed_values":{key:[-1,1] for key in ACTUATORS},
                "excluded_actuators":excluded,"max_joint_targets":joint,"cost_cap":cap}
        legal=legal_actions(schema)
        assert len(legal)==expected,(name,len(legal))
        for variant,description in enumerate(descriptions):
            task_id=f"{name}_{variant}"
            public.append({"id":task_id,"description":COMMON+" "+description,
                           "requested_output":"JSON list of legal actions with targets and values"})
            answer.append({"id":task_id,"schema":schema,"legal_actions":legal,
                           "legal_action_count":expected})
    a.output.mkdir(parents=True)
    for name,rows in (("prompts.jsonl",public),("answer_key.jsonl",answer)):
        (a.output/name).write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
    sha=lambda name:hashlib.sha256((a.output/name).read_bytes()).hexdigest()
    (a.output/'complete.json').write_text(json.dumps({
        "source_revision":revision,"tasks":len(public),"distinct_schemas":len(FIXTURES),
        "legal_action_counts":[row[5] for row in FIXTURES],
        "prompts_sha256":sha('prompts.jsonl'),"answer_key_sha256":sha('answer_key.jsonl'),
        "closed_model_calls":0,"simulator_queries":0},sort_keys=True,indent=2)+'\n')
    print('validated',len(public),'texts across',len(FIXTURES),'formal schemas')


if __name__=='__main__':
    main()

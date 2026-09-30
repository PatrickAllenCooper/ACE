#!/usr/bin/env python3
"""Frozen public-text arithmetic control for the paired binary-SCM suite."""
from __future__ import annotations

import argparse
import json
import re
from fractions import Fraction
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row['id']: row for row in rows}


def parse_public(text: str) -> tuple[Fraction, list[list[int]]]:
    p = re.search(r'Bernoulli\((\d+/\d+|\d+)\)', text)
    tables = re.findall(r'Candidate [AB] table: (\[[01], [01], [01], [01]\])', text)
    if p is None or len(tables) != 2:
        raise ValueError('public description outside frozen grammar')
    return Fraction(p.group(1)), [json.loads(table) for table in tables]


def probability(table: list[int], p: Fraction, action: str) -> Fraction:
    if action == 'doZ1':
        return (1-p)*table[0]+p*table[3]
    if action not in ('doX0','doX1'):
        raise ValueError(action)
    x = int(action[-1])
    return (1-p)*table[2*x]+p*table[2*x+1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--reveals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    public, reveals = load(args.public), load(args.reveals)
    if set(public) != set(reveals):
        raise ValueError('task IDs differ')
    result = []
    for task_id, task in public.items():
        p, tables = parse_public(task['description'])
        q = [probability(table,p,'doX1') for table in tables]
        # For a binary outcome, equal-prior information is positive iff response laws differ.
        action = next((action for action in task['legal_actions']
                       if probability(tables[0],p,action) != probability(tables[1],p,action)),
                      'abstain_unresolved')
        reveal = reveals[task_id]
        likelihood = [probability(table,p,reveal['reference_action']) for table in tables]
        if reveal['observed_Y'] == 0:
            likelihood = [1-value for value in likelihood]
        denominator = sum(likelihood)
        if denominator <= 0:
            raise ValueError('impossible reference observation')
        posterior = [value/denominator for value in likelihood]
        result.append({'id':task_id,
                       'stage1':{'candidate_set_doX1_mean_interval':[
                           str(min(q)),str(max(q))],
                           'observations_identify_candidate':False,'action':action},
                       'stage2':{'posterior_A_B':[str(v) for v in posterior],
                                 'posterior_supported_candidates':[
                                     name for name,value in zip(('A','B'),posterior) if value > 0]}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in result))
    print(f'rule control: {len(result)} public tasks')


if __name__ == '__main__':
    main()

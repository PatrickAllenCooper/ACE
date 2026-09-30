#!/usr/bin/env python3
"""Generate exact two-candidate binary-SCM tasks with discriminating/null menus."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np


def binary_entropy(p: Fraction) -> float:
    x=float(p)
    return -sum(v*math.log2(v) for v in (x,1-x) if v)


def y_probability(table: list[int], p: Fraction, action: str) -> Fraction:
    if action=='doZ1':
        return (1-p)*table[0] + p*table[3]
    x=0 if action=='doX0' else 1
    return (1-p)*table[2*x] + p*table[2*x+1]


def information_gain(tables: list[list[int]], p: Fraction, action: str) -> float:
    q=[y_probability(table,p,action) for table in tables]
    return binary_entropy((q[0]+q[1])/2) - sum(binary_entropy(v) for v in q)/2


def obs_law(table: list[int], p: Fraction) -> dict[str,str]:
    return {f'X0Y{table[0]}':str(1-p),f'X1Y{table[3]}':str(p)}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION')!=revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    public=[]; reveal=[]; answers=[]
    for index,seed in enumerate(range(2100,2120)):
        rng=np.random.default_rng(seed+991)
        p=Fraction(int(rng.choice((1,2,3))),4)
        diagonal=[int(rng.integers(2)),int(rng.integers(2))]
        first=[diagonal[0],int(rng.integers(2)),int(rng.integers(2)),diagonal[1]]
        second=first.copy()
        flip=int(rng.choice((1,2,3)))
        if flip in (1,3):second[1]^=1
        if flip in (2,3):second[2]^=1
        tables=[first,second]
        if rng.integers(2):tables.reverse()
        active=index%2==0
        menu=['doX0','doX1','doZ1'] if active else ['doZ1']
        assert obs_law(tables[0],p)==obs_law(tables[1],p)
        gains={action:information_gain(tables,p,action) for action in menu}
        informative=[action for action in menu if gains[action]>1e-12]
        assert bool(informative)==active
        best=max(menu,key=lambda action:gains[action]) if active else 'abstain_unresolved'
        query=[y_probability(table,p,'doX1') for table in tables]
        reference_action=best if active else 'doZ1'
        true_index=int(rng.integers(2))
        u=int(rng.random()<float(p))
        x=u if reference_action=='doZ1' else (0 if reference_action=='doX0' else 1)
        outcome=tables[true_index][2*x+u]
        likelihoods=[y_probability(table,p,reference_action) for table in tables]
        likelihoods=[v if outcome==1 else 1-v for v in likelihoods]
        denominator=sum(likelihoods)
        assert denominator>0
        posterior=[v/denominator for v in likelihoods]
        task_id=f'pair_{seed}'
        description=(f'Only two candidate SCMs are under consideration. U is hidden and Bernoulli({p}); X=U observationally. '
                     'Y=f(X,U), with each table ordered as f(0,0), f(0,1), f(1,0), f(1,1). '
                     'Z is an irrelevant actuator: do(Z=1) changes neither X nor Y. '
                     f'Candidate A table: {tables[0]}. Candidate B table: {tables[1]}. '
                     'The candidates have equal prior weight. All listed actions have cost 1.')
        public.append({'id':task_id,'description':description,
                       'observational_joint_law':obs_law(tables[0],p),
                       'legal_actions':menu,
                       'question':'Return JSON with candidate_set_doX1_mean_interval, observations_identify_candidate, action (a legal action or abstain_unresolved). If actions tie, choose the first listed.'})
        reveal.append({'id':task_id,'reference_action':reference_action,'observed_Y':outcome,
                       'question':'Given the same two candidates and equal pre-intervention prior, return posterior probabilities for A and B as JSON.'})
        answers.append({'id':task_id,'seed':seed,'p':str(p),'tables':tables,
                        'candidate_set_doX1_mean_interval':[str(min(query)),str(max(query))],
                        'observations_identify_candidate':False,
                        'action_information_gain_bits':gains,
                        'best_action':best,'reference_true_candidate':true_index,
                        'reference_U':u,'posterior_A_B':[str(v) for v in posterior]})
    args.output.mkdir(parents=True)
    for name,rows in (('stage1_prompts.jsonl',public),('stage2_reveals.jsonl',reveal),('answer_key.jsonl',answers)):
        (args.output/name).write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
    hashes={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest()
            for name in ('stage1_prompts.jsonl','stage2_reveals.jsonl','answer_key.jsonl')}
    (args.output/'complete.json').write_text(json.dumps({
        'source_revision':revision,'protocol':'protocol_partial_id_open_model_dev_2026-09-30.md',
        'tasks':20,'discriminating_menus':10,'null_menus':10,
        'simulator_queries':0,'closed_model_calls':0,'file_sha256':hashes},indent=2,sort_keys=True)+'\n')
    print('validated 20 exact paired-SCM tasks (10 discriminating, 10 null)')


if __name__=='__main__':
    main()

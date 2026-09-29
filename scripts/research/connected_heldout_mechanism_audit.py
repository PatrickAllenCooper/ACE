#!/usr/bin/env python3
"""Validate an out-of-bank conditional form in the connected SCM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from connected_motif import make_system, sample


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    a.output.mkdir(parents=True)
    rows=[]
    for seed in range(1900,1912):
        system = make_system(seed,30,10,.15,topology='fanout')
        old,phi,old_mask = sample(system,np.random.default_rng(seed+77),512)
        new,phi_new,new_mask = sample(system,np.random.default_rng(seed+77),512,
                                      heldout_terms={7:.85})
        if not np.array_equal(old_mask,new_mask) or not old_mask.all():
            raise AssertionError('natural mechanism masks drifted')
        parent1,parent2 = system.parents[7]
        if not np.array_equal(old[:,parent1],new[:,parent1]) or not np.array_equal(old[:,parent2],new[:,parent2]):
            raise AssertionError('held-out term changed its parents')
        expected=.85*np.tanh(1.7*old[:,parent1]+.8*old[:,parent2])
        actual=new[:,system.children[7]]-old[:,system.children[7]]
        if not np.allclose(actual,expected,rtol=0,atol=1e-12):
            raise AssertionError('incorrect held-out response')
        if not np.array_equal(phi[:,7],phi_new[:,7]):
            raise AssertionError('held-out parent features drifted')
        for j,child in enumerate(system.children):
            if j != 7 and not np.array_equal(old[:,child],new[:,child]):
                raise AssertionError('unmodified motif response changed')
        rng=np.random.default_rng(seed+303)
        grid=rng.uniform(-2,2,(2048,2))
        design=np.column_stack((grid[:,0],grid[:,1],grid[:,0]*grid[:,1]))
        term=.85*np.tanh(1.7*grid[:,0]+.8*grid[:,1])
        projection=np.linalg.lstsq(design,term,rcond=None)[0]
        residual=float(np.mean((term-design@projection)**2))
        if residual <= .002:
            raise AssertionError('held-out form nearly contained in fitted bank')
        rows.append({'seed':seed,'heldout_motif':7,'amplitude':.85,
                     'generated_paired_trajectories_per_domain':512,
                     'max_term_identity_error':float(np.max(np.abs(actual-expected))),
                     'three_feature_projection_residual_mse':residual,
                     'source_revision':revision})
    with (a.output/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    (a.output/'complete.json').write_text(json.dumps({
        'source_revision':revision,'systems':12,'heldout_motif':7,
        'paired_diagnostic_trajectories_per_domain':12*512,
        'acquired_training_trajectories':0,
        'metrics_sha256':sha(a.output/'metrics.csv'),'closed_model_calls':0},indent=2)+'\n')
    print('validated',len(rows),'systems; residual range',
          min(r['three_feature_projection_residual_mse'] for r in rows),
          max(r['three_feature_projection_residual_mse'] for r in rows))


if __name__=='__main__':
    main()

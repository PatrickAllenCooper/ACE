#!/usr/bin/env python3
"""Smoothly taper selected nonlinear repairs by acquired parent-context support."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from connected_motif import action_menu, make_system, sample
from connected_transfer_bridge_dev import posterior
from heldout_rbf_residual_screen import WIDTHS, PRECISIONS, fit, predict


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cv_losses(z, y, prior_mean, prior_cov):
    fold = np.arange(len(y)) % 4
    linear_errors = []
    candidates = {(w,p):[] for w in WIDTHS for p in PRECISIONS}
    for heldout in range(4):
        train = fold != heldout
        linear_mean,_ = posterior(z[train],y[train],prior_mean,prior_cov,.15)
        linear_errors.extend((np.sum(z[~train]*linear_mean,axis=1)-y[~train])**2)
        for width,precision in candidates:
            coef = fit(z[train],y[train],prior_mean,prior_cov,width,precision)
            candidates[width,precision].extend((predict(z[~train],coef,width)-y[~train])**2)
    linear_cv = float(np.mean(linear_errors))
    rbf_cv,width,precision = min((float(np.mean(errors)),w,p)
                                 for (w,p),errors in candidates.items())
    # Prespecified development gate. The absolute margin is ~11% of known
    # observation-noise variance; both criteria must pass.
    repair = rbf_cv < .9*linear_cv and linear_cv-rbf_cv > .0025
    return linear_cv,rbf_cv,width,precision,repair


def support_weight(q, acquired_z, width):
    # Uses acquired parent inputs and the query's parent inputs only; no
    # sealed labels, simulator truth, or panel-level tuning.
    squared=np.sum((q[:,None,:2]-acquired_z[None,:,:2])**2,axis=2)
    nearest=np.min(squared,axis=1)
    return np.exp(-nearest/(2*width**2))


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION')!=revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    parent=json.loads((a.input/'complete.json').read_text())
    if parent['systems']!=12 or parent['target_campaigns']!=48:
        raise ValueError('unexpected input')
    rows=[]
    for seed in range(2000,2012):
        sd=a.input/f'seed_{seed}'
        spec=json.loads((sd/'system.json').read_text())
        source=make_system(seed,30,10,.15,topology='fanout')
        target=np.asarray(spec['target_coefficients'])
        inputs=np.load(sd/'source_and_assay.npz')
        rng=np.random.default_rng(seed+77153)
        padding=np.random.default_rng(seed+77154)
        panel=[]
        for action in action_menu(source,True):
            _,phi,mask=sample(source,rng,64,action,coefficients=target,
                               padding_rng=padding,heldout_terms={7:.85})
            panel.append((phi,mask))
        for source_n in (16,64):
            for policy in ('factorial_hub_pair','risk_pair'):
                cell=sd/f'source_n_{source_n}'/policy
                receipt=json.loads((cell/'complete.json').read_text())
                if sha(cell/'acquired.npz')!=receipt['acquired.npz_sha256']:
                    raise AssertionError('acquired-data receipt mismatch')
                acquired=np.load(cell/'acquired.npz')
                for j,child in enumerate(source.children):
                    natural=acquired['natural_mask'][:,j]
                    z=acquired['motif_features'][natural,j]
                    y=acquired['values'][natural,child]
                    if len(y)<16:
                        raise AssertionError('too few natural labels')
                    prior_mean,prior_cov=posterior(inputs['source_features'][:source_n,j],
                                                   inputs['source_values'][:source_n,child],
                                                   np.zeros(3),np.eye(3),.15)
                    linear_cv,rbf_cv,width,precision,repair=cv_losses(z,y,prior_mean,prior_cov)
                    linear_mean,_=posterior(z,y,prior_mean,prior_cov,.15)
                    nonlinear_coef=fit(z,y,prior_mean,prior_cov,width,precision) if repair else None
                    linear_errors=[]; gated_errors=[]; tapered_errors=[]
                    support_values=[]
                    for phi,mask in panel:
                        q=phi[mask[:,j],j]
                        truth=np.sum(q*target[j],axis=1)
                        if j==7:
                            truth+=.85*np.tanh(1.7*q[:,0]+.8*q[:,1])
                        linear=np.sum(q*linear_mean,axis=1)
                        gated=predict(q,nonlinear_coef,width) if repair else linear
                        if repair:
                            weight=support_weight(q,z,width)
                            tapered=linear+weight*(gated-linear)
                            support_values.extend(weight.tolist())
                        else:
                            tapered=linear
                        linear_errors.extend((linear-truth)**2)
                        gated_errors.extend((gated-truth)**2)
                        tapered_errors.extend((tapered-truth)**2)
                    linear_mse=float(np.mean(linear_errors))
                    gated_mse=float(np.mean(gated_errors))
                    tapered_mse=float(np.mean(tapered_errors))
                    if not all(np.isfinite(x) for x in (linear_mse,gated_mse,tapered_mse)):
                        raise AssertionError('invalid sealed score')
                    rows.append({'seed':seed,'source_n':source_n,'policy':policy,
                                 'motif':j,'changed':int(j in (0,5,7)),
                                 'out_of_bank':int(j==7),'training_n':len(y),
                                 'linear_cv_mse':linear_cv,'rbf_cv_mse':rbf_cv,
                                 'rbf_width':width,'rbf_precision':precision,
                                 'repair_selected':int(repair),
                                 'mean_support_weight':float(np.mean(support_values)) if repair else '',
                                 'linear_feasible_mse':linear_mse,
                                 'untapered_feasible_mse':gated_mse,
                                 'tapered_feasible_mse':tapered_mse})
    a.output.mkdir(parents=True)
    with (a.output/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    (a.output/'complete.json').write_text(json.dumps({
        'source_revision':revision,'input_complete_sha256':sha(a.input/'complete.json'),
        'input_source_revision':parent['source_revision'],
        'rows':len(rows),'training_queries_added':0,'closed_model_calls':0,
        'gate':{'relative_cv_factor':.9,'absolute_cv_gain':.0025},
        'taper':'exp(-nearest_acquired_parent_squared_distance/(2*selected_rbf_width^2))',
        'metrics_sha256':sha(a.output/'metrics.csv')},indent=2)+'\n')
    print('screened',len(rows),'motif cells; selected',sum(r['repair_selected'] for r in rows),'repairs')


if __name__=='__main__':
    main()

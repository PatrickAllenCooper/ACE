#!/usr/bin/env python3
"""Post hoc, oracle-localized RBF residual capacity screen on saved target data."""
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


CENTERS = np.array([(x,y) for x in (-2.,0.,2.) for y in (-2.,0.,2.)])
WIDTHS = (.75,1.5,3.)
PRECISIONS = (1.,10.,100.,1000.)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def basis(z: np.ndarray, width: float) -> np.ndarray:
    delta = z[:,None,:2] - CENTERS[None,:,:]
    return np.exp(-np.sum(delta**2,axis=2)/(2*width**2))


def fit(z: np.ndarray, y: np.ndarray, prior_mean: np.ndarray,
        prior_cov: np.ndarray, width: float, precision: float) -> np.ndarray:
    design = np.column_stack((z,basis(z,width)))
    ridge = np.zeros((12,12))
    ridge[:3,:3] = np.linalg.inv(prior_cov)
    ridge[3:,3:] = precision*np.eye(9)
    prior = np.r_[prior_mean,np.zeros(9)]
    return np.linalg.solve(ridge+design.T@design/.15**2,
                           ridge@prior+design.T@y/.15**2)


def predict(z: np.ndarray, coef: np.ndarray, width: float) -> np.ndarray:
    return np.column_stack((z,basis(z,width)))@coef


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    source_receipt = json.loads((a.input/'complete.json').read_text())
    if source_receipt['systems'] != 12:
        raise ValueError('wrong input suite')
    rows = []
    for seed in range(2000,2012):
        sd = a.input/f'seed_{seed}'
        spec = json.loads((sd/'system.json').read_text())
        source = make_system(seed,30,10,.15,topology='fanout')
        target_coef = np.asarray(spec['target_coefficients'])
        if not np.array_equal(source.coefficients,np.asarray(spec['source_coefficients'])):
            raise AssertionError('source mismatch')
        inputs = np.load(sd/'source_and_assay.npz')
        rng = np.random.default_rng(seed+77153)
        padding = np.random.default_rng(seed+77154)
        test_z = []
        for action in action_menu(source,True):
            _,phi,mask = sample(source,rng,64,action,coefficients=target_coef,
                                 padding_rng=padding,heldout_terms={7:.85})
            test_z.append(phi[mask[:,7],7])
        test_z = np.concatenate(test_z)
        truth = np.sum(test_z*target_coef[7],axis=1)+.85*np.tanh(1.7*test_z[:,0]+.8*test_z[:,1])
        for source_n in (16,64):
            prior_mean,prior_cov = posterior(inputs['source_features'][:source_n,7],
                                             inputs['source_values'][:source_n,source.children[7]],
                                             np.zeros(3),np.eye(3),.15)
            for policy in ('factorial_hub_pair','risk_pair'):
                cell = sd/f'source_n_{source_n}'/policy
                receipt = json.loads((cell/'complete.json').read_text())
                if sha(cell/'acquired.npz') != receipt['acquired.npz_sha256']:
                    raise AssertionError('acquired-data hash mismatch')
                acquired = np.load(cell/'acquired.npz')
                natural = acquired['natural_mask'][:,7]
                z = acquired['motif_features'][natural,7]
                y = acquired['values'][natural,source.children[7]]
                if len(y)!=44:
                    raise AssertionError('unexpected training-label count')
                baseline_mean,_ = posterior(z,y,prior_mean,prior_cov,.15)
                baseline = float(np.mean((np.sum(test_z*baseline_mean,axis=1)-truth)**2))
                choices=[]
                fold=np.arange(len(y))%4
                for width in WIDTHS:
                    for precision in PRECISIONS:
                        losses=[]
                        for heldout in range(4):
                            train=fold!=heldout
                            coef=fit(z[train],y[train],prior_mean,prior_cov,width,precision)
                            losses.extend(((predict(z[~train],coef,width)-y[~train])**2).tolist())
                        choices.append((float(np.mean(losses)),width,precision))
                cv,width,precision=min(choices)
                coef=fit(z,y,prior_mean,prior_cov,width,precision)
                rbf=float(np.mean((predict(test_z,coef,width)-truth)**2))
                if not np.isfinite(rbf) or not np.isfinite(baseline):
                    raise AssertionError('nonfinite heldout result')
                rows.append({'seed':seed,'source_n':source_n,'policy':policy,
                             'training_n':len(y),'width':width,'precision':precision,
                             'training_only_cv_mse':cv,'linear_warm_mse':baseline,
                             'rbf_residual_mse':rbf,'rbf_minus_linear':rbf-baseline})
    a.output.mkdir(parents=True)
    with (a.output/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    (a.output/'complete.json').write_text(json.dumps({
        'source_revision':revision,'input_source_revision':source_receipt['source_revision'],
        'input_complete_sha256':sha(a.input/'complete.json'),
        'cells':len(rows),'new_training_queries':0,'closed_model_calls':0,
        'metrics_sha256':sha(a.output/'metrics.csv')},indent=2)+'\n')
    print('screened',len(rows),'saved-data cells')


if __name__=='__main__':
    main()

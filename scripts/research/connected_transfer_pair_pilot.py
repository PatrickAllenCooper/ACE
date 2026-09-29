#!/usr/bin/env python3
"""Small connected-SCM source-transfer and pair-acquisition development pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from connected_acquisition import SealedEvaluator, choose
from connected_motif import action_menu, make_system, sample
from connected_transfer_bridge_dev import posterior


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def update(values, features, natural, children, mean, cov):
    for j, child in enumerate(children):
        z = features[natural[:,j],j]
        if len(z):
            mean[j], cov[j] = posterior(z, values[natural[:,j],child],
                                        mean[j], cov[j], .15)


def score(evaluator, truth, mean):
    out = []
    for j in range(len(mean)):
        errors = []
        for phi, mask in evaluator.panel:
            z = phi[mask[:,j],j]
            if len(z):
                errors.extend((z @ (mean[j]-truth[j]))**2)
        out.append(float(np.mean(errors)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    a.output.mkdir(parents=True)
    all_metrics = []
    receipts = {}
    broad_mean, broad_cov = np.zeros(3), np.eye(3)
    for seed in range(1800,1806):
        source = make_system(seed,30,10,.15,topology='fanout')
        coefficients = source.coefficients.copy()
        coefficients[0,0] += 1.0
        coefficients[5,2] += .5
        coefficients[7,0] -= .35
        target = replace(source,coefficients=coefficients)
        public = replace(target,coefficients=np.zeros_like(coefficients))
        evaluator = SealedEvaluator(target,seed)
        source_values, source_phi, natural = sample(source,np.random.default_rng(seed+101),64)
        assert natural.all()
        assay_values, assay_phi, assay_mask = sample(target,np.random.default_rng(seed+601),4)
        assert assay_mask.all()
        seed_dir = a.output/f'seed_{seed}'
        seed_dir.mkdir()
        np.savez_compressed(seed_dir/'source_and_assay.npz',
                            source_values=source_values,source_features=source_phi,
                            assay_values=assay_values,assay_features=assay_phi)
        (seed_dir/'system.json').write_text(json.dumps({
            'seed':seed,'nodes':source.nodes,'motifs':source.motifs,'topology':'fanout',
            'parents':source.parents,'children':source.children,'edges':source.edges,
            'source_coefficients':source.coefficients.tolist(),
            'target_coefficients':target.coefficients.tolist(),
            'source_revision':revision},indent=2)+'\n')
        for source_n in (16,64):
            source_mean = np.empty((10,3)); source_cov = np.empty((10,3,3))
            for j, child in enumerate(source.children):
                source_mean[j],source_cov[j] = posterior(
                    source_phi[:source_n,j],source_values[:source_n,child],
                    broad_mean,broad_cov,.15)
            for method in ('factorial_hub_pair','risk_pair'):
                warm_mean,warm_cov = source_mean.copy(),source_cov.copy()
                scratch_mean = np.zeros((10,3))
                scratch_cov = np.repeat(np.eye(3)[None,:,:],10,axis=0)
                update(assay_values,assay_phi,assay_mask,source.children,warm_mean,warm_cov)
                update(assay_values,assay_phi,assay_mask,source.children,scratch_mean,scratch_cov)
                rows = [{'step':0,'motif':'','targets':'','levels':'',
                         'trajectories':4,'actuator_uses':0,'natural_motif_labels':40,
                         'masked_motif_labels':0,'step_cost':4,'cumulative_cost':4}]
                chunks = [(assay_values,assay_phi,assay_mask)]
                schedule = [(0,(-2.,-2.)),(0,(-2.,2.)),(0,(2.,-2.)),
                            (0,(2.,2.)),(1,(-2.,-2.))]
                menu = action_menu(target,True)
                spent = 4
                decision_rng = np.random.default_rng(seed+9001)
                padding_rng = np.random.default_rng(seed+9002)
                for step in range(5):
                    if method == 'risk_pair':
                        action = choose('risk_pair',public,decision_rng,padding_rng,
                                        warm_mean,warm_cov,step,8)
                    else:
                        motif,levels = schedule[step]
                        action = next(x for x in menu if x[0]==motif and x[2]==levels)
                    # Couple environment randomness by batch, independent of
                    # how many proposal simulations the selector performed.
                    values,phi,mask = sample(target,np.random.default_rng(seed+701+step),8,
                                             action,padding_rng=np.random.default_rng(seed+1701+step))
                    update(values,phi,mask,source.children,warm_mean,warm_cov)
                    update(values,phi,mask,source.children,scratch_mean,scratch_cov)
                    chunks.append((values,phi,mask))
                    spent += 72
                    rows.append({'step':step+1,'motif':action[0],
                                 'targets':','.join(map(str,action[1])),
                                 'levels':','.join(map(str,action[2])),
                                 'trajectories':8,'actuator_uses':16,
                                 'natural_motif_labels':int(mask.sum()),
                                 'masked_motif_labels':int(mask.size-mask.sum()),
                                 'step_cost':72,'cumulative_cost':spent})
                assert spent==364 and sum(r['trajectories'] for r in rows)==44
                assert sum(r['actuator_uses'] for r in rows)==80
                assert sum(r['natural_motif_labels']+r['masked_motif_labels']
                           for r in rows)==440
                cell = seed_dir/f'source_n_{source_n}'/method
                cell.mkdir(parents=True)
                write_csv(cell/'actions.csv',rows)
                np.savez_compressed(cell/'acquired.npz',
                                    values=np.concatenate([x[0] for x in chunks]),
                                    motif_features=np.concatenate([x[1] for x in chunks]),
                                    natural_mask=np.concatenate([x[2] for x in chunks]))
                metrics = []
                for estimator,mean in (('source_warm',warm_mean),('scratch',scratch_mean)):
                    errors = score(evaluator,coefficients,mean)
                    for j,error in enumerate(errors):
                        metrics.append({'seed':seed,'source_n':source_n,'policy':method,
                                        'estimator':estimator,'motif':j,
                                        'changed':int(j in (0,5,7)),
                                        'feasible_motif_mse':error})
                write_csv(cell/'metrics.csv',metrics)
                all_metrics.extend(metrics)
                receipt = {'seed':seed,'source_n':source_n,'policy':method,
                           'source_revision':revision,'target_trajectories':44,
                           'target_cost':364,'actuator_uses':80,
                           'natural_motif_labels':sum(r['natural_motif_labels'] for r in rows),
                           'masked_motif_labels':sum(r['masked_motif_labels'] for r in rows),
                           **{name+'_sha256':sha(cell/name)
                              for name in ('actions.csv','acquired.npz','metrics.csv')},
                           'closed_model_calls':0}
                (cell/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
                receipts[f'{seed}/{source_n}/{method}'] = sha(cell/'complete.json')
    write_csv(a.output/'metrics.csv',all_metrics)
    summary=[]
    for n in (16,64):
        for policy in ('factorial_hub_pair','risk_pair'):
            for estimator in ('source_warm','scratch'):
                for changed in (0,1):
                    group=[r['feasible_motif_mse'] for r in all_metrics
                           if r['source_n']==n and r['policy']==policy and
                           r['estimator']==estimator and r['changed']==changed]
                    summary.append({'source_n':n,'policy':policy,'estimator':estimator,
                                    'changed':changed,'mean_feasible_motif_mse':float(np.mean(group))})
    write_csv(a.output/'summary.csv',summary)
    (a.output/'complete.json').write_text(json.dumps({
        'source_revision':revision,'systems':6,'source_sizes':[16,64],
        'pair_policies':2,'estimators_per_policy':2,'target_campaigns':24,
        'target_arm_trajectories':24*44,'target_arm_cost':24*364,
        'target_arm_actuator_uses':24*80,'metric_rows':len(all_metrics),
        'source_generated_trajectories':6*64,
        'target_assay_generated_trajectories':6*4,
        'cell_receipt_hashes':receipts,
        'metrics_sha256':sha(a.output/'metrics.csv'),
        'summary_sha256':sha(a.output/'summary.csv'),
        'closed_model_calls':0},indent=2)+'\n')
    print('validated',len(all_metrics),'metric rows, 24 target campaigns')
    for row in summary: print(row)


if __name__=='__main__':
    main()

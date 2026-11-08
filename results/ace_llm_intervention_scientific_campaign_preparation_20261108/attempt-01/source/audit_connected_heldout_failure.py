#!/usr/bin/env python3
"""Post hoc decomposition of the frozen connected held-out transfer result."""
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    suite = json.loads((a.input/'complete.json').read_text())
    if suite['systems'] != 12 or suite['target_campaigns'] != 48:
        raise ValueError('unexpected input suite')
    rows = []
    for seed in range(2000,2012):
        seed_dir = a.input/f'seed_{seed}'
        spec = json.loads((seed_dir/'system.json').read_text())
        if sha(seed_dir/'system.json') != suite['seed_input_hashes'][str(seed)]['system_sha256']:
            raise AssertionError('system receipt mismatch')
        system = make_system(seed,30,10,.15,topology='fanout')
        target = np.asarray(spec['target_coefficients'])
        rng = np.random.default_rng(seed+77153)
        padding_rng = np.random.default_rng(seed+77154)
        features = []
        for action in action_menu(system,True):
            _, phi, mask = sample(system,rng,64,action,coefficients=target,
                                  padding_rng=padding_rng,heldout_terms={7:.85})
            features.append(phi[mask[:,7],7])
        z = np.concatenate(features)
        truth = np.sum(z*target[7],axis=1) + .85*np.tanh(1.7*z[:,0]+.8*z[:,1])
        projection = np.linalg.lstsq(z,truth,rcond=None)[0]
        projected = np.sum(z*projection,axis=1)
        if not np.isfinite(z).all() or not np.isfinite(truth).all() or not np.isfinite(projected).all():
            raise AssertionError('nonfinite oracle projection')
        floor = float(np.mean((projected-truth)**2))
        if not np.isfinite(floor) or floor <= 0:
            raise AssertionError('invalid approximation floor')
        for source_n in (16,64):
            for policy in ('factorial_hub_pair','risk_pair'):
                cell = seed_dir/f'source_n_{source_n}'/policy
                receipt = json.loads((cell/'complete.json').read_text())
                for name in ('actions.csv','acquired.npz','metrics.csv'):
                    if sha(cell/name) != receipt[name+'_sha256']:
                        raise AssertionError('cell receipt mismatch')
                actions = list(csv.DictReader((cell/'actions.csv').open()))
                visited = [int(row['motif']) for row in actions[1:]]
                acquired = np.load(cell/'acquired.npz')
                mask = acquired['natural_mask']
                metrics = list(csv.DictReader((cell/'metrics.csv').open()))
                for estimator in ('source_warm','scratch'):
                    metric = next(row for row in metrics if row['estimator']==estimator and
                                  int(row['motif'])==7)
                    rows.append({
                        'seed':seed,'source_n':source_n,'policy':policy,
                        'estimator':estimator,'motif_7_oracle_linear_floor':floor,
                        'motif_7_learned_mse':float(metric['feasible_motif_mse']),
                        'motif_7_excess_above_linear_floor':float(metric['feasible_motif_mse'])-floor,
                        'motif_7_direct_pair_visits':visited.count(7),
                        'motif_7_natural_labels':int(mask[:,7].sum()),
                        'visited_motifs':','.join(map(str,visited)),
                    })
    a.output.mkdir(parents=True)
    write_csv(a.output/'decomposition.csv',rows)
    (a.output/'complete.json').write_text(json.dumps({
        'source_revision':revision,'input_source_revision':suite['source_revision'],
        'input_complete_sha256':sha(a.input/'complete.json'),
        'rows':len(rows),'new_training_queries':0,'closed_model_calls':0,
        'decomposition_sha256':sha(a.output/'decomposition.csv'),
    },indent=2)+'\n')
    print('validated',len(rows),'post hoc motif-7 decomposition rows')


if __name__=='__main__':
    main()

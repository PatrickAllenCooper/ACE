#!/usr/bin/env python3
"""Validate full-trajectory accounting for a connected transfer campaign."""
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

from connected_motif import action_menu, make_system, sample


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    revision = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    if os.environ.get('ACE_SOURCE_REVISION') != revision:
        raise ValueError('ACE_SOURCE_REVISION must equal HEAD')
    a.output.mkdir(parents=True)
    receipts = {}
    for seed in range(1700, 1703):
        source = make_system(seed, 30, 10, .15, topology='fanout')
        coeff = source.coefficients.copy()
        coeff[0,0] += 1.0
        coeff[5,2] += .5
        coeff[7,0] -= .35
        target = replace(source, coefficients=coeff)
        rng = np.random.default_rng(seed+601)
        menu = action_menu(target, True)
        schedule = [(0,(-2.,-2.)), (0,(-2.,2.)), (0,(2.,-2.)),
                    (0,(2.,2.)), (1,(-2.,-2.))]
        cell = a.output/f'seed_{seed}'
        cell.mkdir()
        rows = []
        chunks = []
        spent = trajectories = natural_labels = masked_labels = actuator_uses = 0
        for step, item in enumerate([(None,None)] + schedule):
            if step == 0:
                action, batch = None, 4
            else:
                motif, levels = item
                action = next(x for x in menu if x[0] == motif and x[2] == levels)
                batch = 8
            values, phi, natural = sample(target, rng, batch, action)
            cost = batch * (1 + 4*(0 if action is None else len(action[1])))
            spent += cost
            trajectories += batch
            actuator_uses += batch*(0 if action is None else len(action[1]))
            natural_labels += int(natural.sum())
            masked_labels += int(natural.size-natural.sum())
            chunks.append((values,phi,natural))
            rows.append({'step': step, 'motif': '' if action is None else action[0],
                         'targets': '' if action is None else ','.join(map(str,action[1])),
                         'levels': '' if action is None else ','.join(map(str,action[2])),
                         'trajectories': batch, 'actuator_uses': batch*(0 if action is None else 2),
                         'natural_motif_labels': int(natural.sum()),
                         'masked_motif_labels': int(natural.size-natural.sum()),
                         'step_cost': cost, 'cumulative_cost': spent})
        if trajectories != 44 or actuator_uses != 80 or spent != 364:
            raise AssertionError('campaign cost mismatch')
        if natural_labels + masked_labels != 44*10:
            raise AssertionError('motif label accounting mismatch')
        # Only motif 1's first parent is a manipulated child on the final
        # action; its natural conditional label must then be excluded.
        if masked_labels != 8 or natural_labels != 432:
            raise AssertionError('intervention mask mismatch')
        with (cell/'actions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
        np.savez_compressed(cell/'acquired.npz',
                            values=np.concatenate([x[0] for x in chunks]),
                            motif_features=np.concatenate([x[1] for x in chunks]),
                            natural_mask=np.concatenate([x[2] for x in chunks]))
        (cell/'system.json').write_text(json.dumps({
            'seed':seed, 'nodes':source.nodes, 'motifs':source.motifs,
            'topology':'fanout', 'parents':source.parents, 'children':source.children,
            'edges':source.edges, 'source_coefficients':source.coefficients.tolist(),
            'target_coefficients':target.coefficients.tolist(),
            'source_revision':revision},indent=2)+'\n')
        receipt = {'source_revision':revision,'seed':seed,'trajectories':trajectories,
                   'actuator_uses':actuator_uses,'cost':spent,
                   'natural_motif_labels':natural_labels,'masked_motif_labels':masked_labels,
                   'actions_sha256':sha(cell/'actions.csv'),
                   'acquired_sha256':sha(cell/'acquired.npz'),
                   'system_sha256':sha(cell/'system.json'),
                   'closed_model_calls':0}
        (cell/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
        receipts[str(seed)] = sha(cell/'complete.json')
    (a.output/'complete.json').write_text(json.dumps({
        'source_revision':revision,'systems':3,'target_trajectories':132,
        'actuator_uses':240,'cost_units':1092,
        'natural_motif_labels':1296,'masked_motif_labels':24,
        'cell_receipt_hashes':receipts,'closed_model_calls':0},indent=2)+'\n')
    print('validated 3 systems, 132 trajectories, 240 actuator uses, 1296 natural labels, 24 masked')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Matched exact-posterior acquisition on the connected-motif SCM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from connected_motif import action_menu, make_system, sample


METHODS = ('random_single', 'coverage_single', 'random_pair', 'coverage_pair', 'risk_pair')
Q = np.diag((4/3, 4/3, 16/9))


class SealedEvaluator:
    def __init__(self, system, seed):
        rng = np.random.default_rng(seed + 77153)
        padding_rng = np.random.default_rng(seed + 77154)
        # An independent, fixed, feasible panel; never passed to a policy.
        self.panel = []
        for action in action_menu(system, True):
            _, phi, mask = sample(system, rng, 64, action, padding_rng=padding_rng)
            self.panel.append((phi, mask))
        self.truth = system.coefficients.copy()

    def evaluate(self, mean):
        error = mean - self.truth
        broad = float(np.mean(np.einsum('mi,ij,mj->m', error, Q, error, optimize=False)))
        feasible = []
        for phi, mask in self.panel:
            # Only natural child outputs enter the feasible comparison.
            pred_error = np.einsum('bmd,md->bm', phi, error, optimize=False)
            feasible.extend((pred_error[mask] ** 2).tolist())
        return broad, float(np.mean(feasible))


def choose(method, public_system, rng, padding_rng, mean, cov, step, batch,
           coverage_offset=0, coverage_order=None, motif_visits=None,
           hub_order=None):
    menu = action_menu(public_system, method.endswith('pair'))
    if method.startswith('random'):
        return menu[int(rng.integers(len(menu)))]
    if method.startswith('coverage'):
        per_motif = len(menu) // public_system.motifs
        motif_position = (step + coverage_offset) % public_system.motifs
        motif = (coverage_order[motif_position] if coverage_order is not None
                 else motif_position)
        return menu[motif * per_motif +
                    (step // public_system.motifs) % per_motif]
    if method.startswith('hub_'):
        if public_system.motifs < 3 or hub_order is None:
            raise ValueError('Hub controls require at least three motifs and a frozen order')
        if step < 3:
            motif = 0
            levels = ((-2., -2.), (-2., 2.), (2., 2.))[step]
        else:
            motif = hub_order[step - 3]
            levels = ((2., 2.), (-2., -2.))[step - 3]
        return next(action for action in menu
                    if action[0] == motif and action[2] == levels)
    if method == 'factorial_hub_pair':
        if public_system.motifs < 2:
            raise ValueError('Factorial hub control requires two motifs')
        motif = 0 if step < 4 else 1
        levels = ((-2., -2.), (-2., 2.), (2., -2.), (2., 2.),
                  (-2., -2.))[step]
        return next(action for action in menu
                    if action[0] == motif and action[2] == levels)
    scores = []
    for action in menu:
        # Student-predictive contexts only: public_system contains zero truth.
        _, phi, mask = sample(public_system, rng, 64, action, coefficients=mean,
                              padding_rng=padding_rng)
        score = 0.
        for j in range(public_system.motifs):
            z = phi[mask[:, j], j]
            if not len(z):
                continue
            moment = z.T @ z / len(z)
            next_cov = np.linalg.inv(np.linalg.inv(cov[j]) + batch * moment / .15**2)
            score += float(np.trace(Q @ (cov[j] - next_cov)))
        scores.append(score)
    if method == 'balanced_risk_pair':
        if motif_visits is None or len(motif_visits) != public_system.motifs:
            raise ValueError('Balanced selection requires motif visit counts')
        minimum = min(motif_visits)
        scores = [score if motif_visits[action[0]] == minimum else -np.inf
                  for action, score in zip(menu, scores)]
    if method == 'risk_motif_fixed_value_pair':
        if motif_visits is None or len(motif_visits) != public_system.motifs:
            raise ValueError('Fixed-value motif selection requires visit counts')
        motif_score = [max(score for action, score in zip(menu, scores)
                           if action[0] == motif)
                       for motif in range(public_system.motifs)]
        motif = int(np.argmax(motif_score))
        levels = ((-2., -2.), (-2., 2.), (2., -2.), (2., 2.))[
            motif_visits[motif] % 4]
        return next(action for action in menu
                    if action[0] == motif and action[2] == levels)
    return menu[int(np.argmax(scores))]


def experiment(seed, nodes, motifs, root_sd, penalty, budget=400, batch=8,
               coverage_offset=0, coverage_order=None, include_balanced=False,
               topology='chain', include_hub=False, include_factorial=False,
               include_fixed_value=False):
    if penalty < 0 or budget < batch:
        raise ValueError('Invalid cost parameters')
    if not 0 <= coverage_offset < motifs:
        raise ValueError('Invalid coverage rotation')
    if coverage_order is not None and sorted(coverage_order) != list(range(motifs)):
        raise ValueError('Coverage order must be a motif permutation')
    if include_hub and motifs < 3:
        raise ValueError('Hub controls require at least three motifs')
    if include_hub and budget // (batch * (1 + 2 * penalty)) != 5:
        raise ValueError('Frozen hub controls require exactly five pair actions')
    if include_factorial and (motifs < 2 or budget // (batch * (1 + 2 * penalty)) != 5):
        raise ValueError('Factorial control requires two motifs and five pair actions')
    system = make_system(seed, nodes, motifs, root_sd, topology=topology)
    public = replace(system, coefficients=np.zeros_like(system.coefficients))
    evaluator = SealedEvaluator(system, seed)
    rows, actions = [], []
    methods = METHODS + (('balanced_risk_pair',) if include_balanced else ()) + (
        ('hub_coverage_pair', 'hub_random_pair') if include_hub else ()) + (
        ('factorial_hub_pair',) if include_factorial else ()) + (
        ('risk_motif_fixed_value_pair',) if include_fixed_value else ())
    hub_random_order = tuple(int(j) for j in
                             np.random.default_rng(seed + 991337).permutation(
                                 np.arange(1, motifs))) if include_hub else ()
    for mi, method in enumerate(methods):
        # Couple controls to the risk arm's initial stream; action paths can diverge.
        rng_index = METHODS.index('risk_pair') if method in (
            'balanced_risk_pair', 'hub_coverage_pair', 'hub_random_pair',
            'factorial_hub_pair', 'risk_motif_fixed_value_pair') else mi
        rng = np.random.default_rng(seed * 113 + rng_index + 31)
        padding_rng = np.random.default_rng(seed * 113 + rng_index + 80031)
        mean = np.zeros((motifs, 3))
        cov = np.repeat(np.eye(3)[None, :, :], motifs, axis=0)
        spent = samples = actuators = masked = step = 0
        motif_visits = [0] * motifs
        hub_order = ((1, motifs - 1) if method == 'hub_coverage_pair'
                     else hub_random_order[:2] if method == 'hub_random_pair' else None)
        unit_cost = 1 + penalty * (2 if method.endswith('pair') else 1)
        while spent + batch * unit_cost <= budget:
            action = choose(method, public, rng, padding_rng, mean, cov, step, batch,
                            coverage_offset=coverage_offset,
                            coverage_order=coverage_order, motif_visits=motif_visits,
                            hub_order=hub_order)
            values, phi, natural = sample(system, rng, batch, action,
                                          padding_rng=padding_rng)
            for j, child in enumerate(system.children):
                z = phi[natural[:, j], j]
                y = values[natural[:, j], child]
                if not len(z):
                    continue
                precision = np.linalg.inv(cov[j])
                cov[j] = np.linalg.inv(precision + z.T @ z / system.child_sd**2)
                mean[j] = cov[j] @ (precision @ mean[j] + z.T @ y / system.child_sd**2)
            masked += int(np.size(natural) - natural.sum())
            spent += batch * unit_cost
            samples += batch
            actuators += batch * len(action[1])
            actions.append({'method': method, 'step': step, 'motif': action[0],
                            'targets': ','.join(map(str, action[1])),
                            'levels': ','.join(map(str, action[2])),
                            'natural_child_labels': int(natural.sum()),
                            'masked_child_labels': int(np.size(natural) - natural.sum()),
                            'cumulative_cost': spent, 'cumulative_samples': samples})
            motif_visits[action[0]] += 1
            step += 1
        broad, feasible = evaluator.evaluate(mean)
        rows.append({'seed': seed, 'nodes': nodes, 'motifs': motifs, 'method': method,
                     'root_sd': root_sd, 'penalty': penalty, 'budget': budget,
                     'cost_spent': spent, 'samples': samples, 'actuator_uses': actuators,
                     'masked_child_labels': masked, 'steps': step,
                     'broad_motif_mse': broad, 'feasible_motif_mse': feasible,
                     'posterior_risk': float(np.mean([np.trace(Q @ c) for c in cov]))})
    spec = {'seed': seed, 'nodes': nodes, 'motifs': motifs, 'root_sd': root_sd,
            'penalty': penalty, 'budget': budget, 'batch': batch,
            'rng_schema': 'separate_padding_v1',
            'parents': system.parents, 'children': system.children,
            'edges': system.edges, 'coefficients': system.coefficients.tolist(),
            'source_revision': os.environ.get('ACE_SOURCE_REVISION', 'local')}
    if topology != 'chain':
        spec['topology'] = topology
    return rows, actions, spec


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--nodes', required=True, type=int)
    p.add_argument('--motifs', required=True, type=int)
    p.add_argument('--root-sd', required=True, type=float)
    p.add_argument('--penalty', required=True, type=int, choices=(0, 1, 4))
    p.add_argument('--budget', type=int, default=400)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--include-balanced', action='store_true')
    p.add_argument('--include-hub', action='store_true')
    p.add_argument('--topology', choices=('chain', 'fanout', 'binary_tree'), default='chain')
    a = p.parse_args()
    rows, actions, spec = experiment(a.seed, a.nodes, a.motifs, a.root_sd,
                                     a.penalty, a.budget, include_balanced=a.include_balanced,
                                     topology=a.topology, include_hub=a.include_hub)
    a.output.mkdir(parents=True, exist_ok=True)
    metric_file, action_file, system_file = (a.output / x for x in
                                              ('metrics.csv', 'actions.csv', 'system.json'))
    write_csv(metric_file, rows)
    write_csv(action_file, actions)
    system_file.write_text(json.dumps(spec, indent=2, sort_keys=True) + '\n')
    receipt = {'schema_version': 2, 'kind': 'connected_acquisition', 'rows': len(rows),
               'actions': len(actions), 'source_revision': spec['source_revision'],
               'metrics_sha256': hashlib.sha256(metric_file.read_bytes()).hexdigest(),
               'actions_sha256': hashlib.sha256(action_file.read_bytes()).hexdigest(),
               'system_sha256': hashlib.sha256(system_file.read_bytes()).hexdigest()}
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'{a.seed} N={a.nodes} k={a.motifs}: {len(rows)} arms, {len(actions)} actions')


if __name__ == '__main__':
    main()

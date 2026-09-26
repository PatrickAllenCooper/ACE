#!/usr/bin/env python3
"""One persistent SCM/learner campaign with fixed, inaccessible evaluation data.

This is a research runner, separate from the historical episode-reset baselines.
No external model APIs are used. The intervention policy only sees the student.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from baselines import (EnsembleLearner, EnsembleStudentSCM, GroundTruthSCM,
                       InstrumentedOracle, NonLeafCoveragePolicy,
                       NonLeafRandomPolicy, PropagatedVariancePolicy,
                       StudentSCM)
from experiments.large_scale_scm import LargeScaleSCM
from experiments.heterogeneous_scm import HeterogeneousSCM


class SealedEvaluator:
    """A fixed observational and mechanism-domain holdout; never passed to policy."""

    def __init__(self, scm, seed: int, n: int = 500):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 800000)
            self.observations = scm.generate(n)
            self.mechanisms = {}
            for node in scm.nodes:
                parents = scm.get_parents(node)
                if parents:
                    contexts = {p: torch.rand(n) * 8 - 4 for p in parents}
                    truth = scm.mechanisms(contexts, node, n_samples=n)
                    self.mechanisms[node] = (contexts, truth)

    def evaluate(self, student) -> dict[str, float]:
        student.eval()
        with torch.no_grad():
            pred = student(self.observations)
            observed = sum(float(((pred[n] - y) ** 2).mean())
                           for n, y in self.observations.items())
            broad = 0.0
            for node in student.nodes:
                parents = student.get_parents(node)
                if not parents:
                    mu = student.mechanisms[node]['mu']
                    broad += 0.2 * float(((mu - self.observations[node]) ** 2).mean())
                    continue
                contexts, truth = self.mechanisms[node]
                matrix = torch.stack([contexts[p] for p in parents], dim=1)
                estimate = student.mechanisms[node](matrix).reshape(-1)
                broad += float(((estimate - truth.reshape(-1)) ** 2).mean())
        student.train()
        return {'observed_total_loss': observed, 'broad_total_loss': broad}


def campaign(scm, method: str, seed: int, budget: int, epochs: int,
             ensemble_size: int, batch: int, obs_batch: int, obs_interval: int,
             pev_values: int, pev_sim: int) -> tuple[list[dict], dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    evaluator = SealedEvaluator(scm, seed)
    oracle = InstrumentedOracle(scm)
    student = EnsembleStudentSCM(oracle, n_members=ensemble_size,
                                 hidden_dims=StudentSCM.ARCHS['ace'])
    learner = EnsembleLearner(student, oracle=oracle)
    if method == 'nonleaf_random_ens':
        policy = NonLeafRandomPolicy(scm.nodes, scm.graph)
    elif method == 'nonleaf_coverage_ens':
        policy = NonLeafCoveragePolicy(scm.nodes, scm.graph)
    elif method in ('pev', 'pev_var'):
        policy = PropagatedVariancePolicy(scm.nodes, n_values=pev_values,
                                          n_sim=pev_sim,
                                          scoring='ivr' if method == 'pev' else 'var')
    else:
        raise ValueError(method)
    records = []
    while True:
        step = len(records)
        refresh = obs_interval > 0 and step > 0 and step % obs_interval == 0
        required = batch + (obs_batch if refresh else 0)
        if oracle.total_samples() + required > budget:
            break
        target, value = policy.select_intervention(student)
        data = oracle.generate(batch, interventions={target: value}, tag='executed')
        learner.train_step(data, intervened=target, n_epochs=epochs)
        if refresh:
            learner.observational_train(oracle, n_samples=obs_batch, n_epochs=epochs)
        metric = evaluator.evaluate(student)
        records.append({'step': step, 'target': target, 'value': value,
                        'query_samples': oracle.total_samples(), **metric})
    summary = oracle.query_summary()
    assert summary['total']['samples'] <= budget
    assert summary.get('candidate_probe', {}).get('samples', 0) == 0
    return records, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', choices=['legacy5', 'hom30', 'hetero30'], required=True)
    p.add_argument('--method', choices=['nonleaf_random_ens', 'nonleaf_coverage_ens',
                                         'pev', 'pev_var'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--budget', type=int, default=2000)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--ensemble-size', type=int, default=3)
    p.add_argument('--batch', type=int, default=50)
    p.add_argument('--obs-batch', type=int, default=40)
    p.add_argument('--obs-interval', type=int, default=3)
    p.add_argument('--pev-values', type=int, default=5)
    p.add_argument('--pev-sim', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.budget, a.epochs, a.ensemble_size, a.batch) <= 0:
        p.error('budget, epochs, ensemble size, and batch must be positive')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.family == 'legacy5':
        scm = GroundTruthSCM()
    elif a.family == 'hom30':
        scm = LargeScaleSCM(30, coeff_seed=a.seed)
    else:
        scm = HeterogeneousSCM(30, coeff_seed=a.seed)
    rows, queries = campaign(scm, a.method, a.seed, a.budget, a.epochs,
                             a.ensemble_size, a.batch, a.obs_batch,
                             a.obs_interval, a.pev_values, a.pev_sim)
    if not rows:
        raise ValueError('budget too small to execute one intervention')
    a.output.mkdir(parents=True, exist_ok=True)
    metrics = a.output / 'trajectory.csv'
    with metrics.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output / 'query_budget.json').write_text(json.dumps(queries, indent=2) + '\n')
    receipt = {'schema_version': 1, 'family': a.family, 'method': a.method,
               'seed': a.seed, 'steps': len(rows), 'budget': a.budget,
               'query_samples': queries['total']['samples'],
               'metrics_sha256': hashlib.sha256(metrics.read_bytes()).hexdigest()}
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f"{a.family}/{a.method}/seed_{a.seed}: {len(rows)} steps, "
          f"{queries['total']['samples']}/{a.budget} samples; "
          f"final broad MSE {rows[-1]['broad_total_loss']:.4f}")


if __name__ == '__main__':
    main()

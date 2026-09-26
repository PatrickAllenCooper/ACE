#!/usr/bin/env python3
"""Small, closed-model-free experiments for the 2026 SCM research portfolio.

These are deliberately separate from historical ACE runs. The prior experiment
tests numerical mechanism priors; design tests single versus joint interventions;
transfer tests local assays after sparse mechanism changes. Every acquisition
rule uses data/posteriors, never hidden test labels. One process runs one seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def features(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[:, 0], x[:, 1]
    return np.column_stack((x1, x2, x1 * x2, x1**2, np.sin(1.4 * x1), np.tanh(1.3 * x2)))


def posterior(x: np.ndarray, y: np.ndarray, mean: np.ndarray, precision: float,
              sigma: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    phi = features(x)
    d = phi.shape[1]
    covariance = np.linalg.inv(precision * np.eye(d) + phi.T @ phi / sigma**2)
    return covariance @ (precision * mean + phi.T @ y / sigma**2), covariance


def predictive_log_evidence(x: np.ndarray, y: np.ndarray, mean: np.ndarray,
                            precision: float, sigma: float = 0.15) -> float:
    phi = features(x)
    cov = sigma**2 * np.eye(len(x)) + phi @ phi.T / precision
    residual = y - phi @ mean
    chol = np.linalg.cholesky(cov)
    logdet = 2 * np.log(np.diag(chol)).sum()
    whitened = np.linalg.solve(chol, residual)
    return float(-0.5 * (len(x) * np.log(2 * np.pi) + logdet + whitened @ whitened))


def truths(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1287)
    form = seed % 4
    theta = np.zeros(6)
    theta[:2] = rng.normal(0.6, 0.1, 2)
    theta[2 + form] = rng.choice((-1.0, 1.0)) * rng.uniform(0.7, 1.1)
    return theta, rng


def prior_experiment(seed: int) -> list[dict]:
    truth, rng = truths(seed)
    x = rng.uniform(-2, 2, (64, 2))
    y = features(x) @ truth + rng.normal(0, 0.15, len(x))
    test_x = np.random.default_rng(seed + 843).uniform(-2, 2, (2048, 2))
    test_features = features(test_x)
    test_truth = np.einsum('ij,j->i', test_features, truth, optimize=False)
    correct = truth.copy()
    correct[:2] = 0.6
    correct[2:] = 0.0
    correct[2 + seed % 4] = np.sign(truth[2 + seed % 4]) * 0.9
    wrong = correct.copy()
    wrong[2:] = 0.0
    wrong[2 + (seed + 1) % 4] = 0.9
    base = np.zeros(6)
    candidates = {'broad': (base, 0.25), 'correct': (correct, 8.0),
                  'wrong': (wrong, 8.0)}
    out = []
    for n in (8, 16, 32, 64):
        fitted = {name: posterior(x[:n], y[:n], mu, lam)[0] for name, (mu, lam) in candidates.items()}
        logs = np.array([predictive_log_evidence(x[:n], y[:n], mu, lam)
                         for mu, lam in (candidates['wrong'], candidates['broad'])])
        weights = np.exp(logs - np.max(logs))
        weights /= weights.sum()
        fitted['wrong_with_fallback'] = weights[0] * fitted['wrong'] + weights[1] * fitted['broad']
        for name, estimate in fitted.items():
            out.append(dict(track='prior', seed=seed, method=name, budget=n,
                            mse=float(np.mean((np.einsum('ij,j->i', test_features, estimate, optimize=False) - test_truth)**2)),
                            fallback_weight=float(weights[1]) if name == 'wrong_with_fallback' else ''))
    return out


def design_candidates(kind: str) -> list[tuple[tuple[int, ...], tuple[float, ...]]]:
    single = [((i,), (v,)) for i in range(2) for v in (-2.0, 2.0)]
    pair = [((0, 1), (v1, v2)) for v1 in (-2.0, 2.0) for v2 in (-2.0, 2.0)]
    return single if kind == 'single' else single + pair


def make_context(action: tuple[tuple[int, ...], tuple[float, ...]], rng: np.random.Generator,
                 n: int, background_sd: float) -> np.ndarray:
    x = rng.normal(0, background_sd, (n, 2))
    for i, value in zip(*action):
        x[:, i] = value
    return x


def design_experiment(seed: int, budget: int = 400, batch: int = 8,
                      background_sd: float = 0.15, actuator_penalty: int = 0) -> list[dict]:
    if budget < batch or batch < 1 or actuator_penalty < 0:
        raise ValueError('Budget must fit at least one batch; penalty must be nonnegative')
    truth = np.array([0.45, -0.3, 0.9])
    q = np.diag([4 / 3, 4 / 3, 16 / 9])
    out = []
    for method in ('random_single', 'design_single', 'random_joint', 'design_joint'):
        rng = np.random.default_rng(seed + {'random_single': 11, 'design_single': 23,
                                             'random_joint': 37, 'design_joint': 49}[method])
        covariance = np.eye(3)
        estimate = np.zeros(3)
        spent = 0
        actions = 0
        while True:
            options = design_candidates('single' if method.endswith('single') else 'joint')
            options = [a for a in options if spent + batch * (1 + actuator_penalty * (len(a[0]) - 1)) <= budget]
            if not options:
                break
            if method.startswith('random'):
                action = options[int(rng.integers(len(options)))]
            else:
                best_score = -np.inf
                action = options[0]
                # Model-based expected design; no oracle sample or hidden truth in scoring.
                for candidate in options:
                    contexts = make_context(candidate, rng, 128, background_sd)
                    phi = np.column_stack((contexts[:, 0], contexts[:, 1], contexts[:, 0] * contexts[:, 1]))
                    moment = phi.T @ phi / len(phi)
                    new_cov = np.linalg.inv(np.linalg.inv(covariance) + batch * moment / 0.15**2)
                    cost = batch * (1 + actuator_penalty * (len(candidate[0]) - 1))
                    score = np.trace(q @ (covariance - new_cov)) / cost
                    if score > best_score:
                        best_score, action = score, candidate
            x = make_context(action, rng, batch, background_sd)
            phi = np.column_stack((x[:, 0], x[:, 1], x[:, 0] * x[:, 1]))
            y = phi @ truth + rng.normal(0, 0.15, batch)
            new_cov = np.linalg.inv(np.linalg.inv(covariance) + phi.T @ phi / 0.15**2)
            estimate = new_cov @ (np.linalg.solve(covariance, estimate) + phi.T @ y / 0.15**2)
            covariance = new_cov
            spent += batch * (1 + actuator_penalty * (len(action[0]) - 1))
            actions += 1
            if actions in (1, 5, 10, 20) or spent + batch > budget:
                error = estimate - truth
                out.append(dict(track='design', seed=seed, method=method, budget=spent,
                                requested_budget=budget, actuator_penalty=actuator_penalty,
                                background_sd=background_sd, actions=actions,
                                mse=float(error @ q @ error), interaction_error=float(error[2] ** 2),
                                posterior_risk=float(np.trace(q @ covariance))))
    return out


def transfer_experiment(seed: int, nodes: int = 30, changed: int = 3,
                        budget: int = 400, batch: int = 8) -> list[dict]:
    """Direct local-mechanism assays; a transfer feasibility test, not do(X) on a whole SCM."""
    if nodes < 4 or not 0 < changed < nodes or budget < batch:
        raise ValueError('Invalid nodes, changed, or budget')
    rng = np.random.default_rng(seed + 9061)
    form_ids = rng.integers(0, 4, nodes)
    family = np.zeros((4, 6))
    family[:, :2] = (0.6, -0.3)
    for j in range(4):
        family[j, 2 + j] = 0.85
    old = family[form_ids] + rng.normal(0, 0.06, (nodes, 6))
    truth = old.copy()
    changed_ids = rng.choice(nodes, changed, replace=False)
    truth[changed_ids] = family[(form_ids[changed_ids] + 1) % 4] + rng.normal(0, 0.06, (changed, 6))
    # Same passive assay for all arms. It is included in the sample budget.
    passive_x = rng.uniform(-2, 2, (nodes, 4, 2))
    passive_y = np.einsum('nid,nd->ni', features(passive_x.reshape(-1, 2)).reshape(nodes, 4, 6), truth)
    passive_y += rng.normal(0, 0.15, passive_y.shape)
    passive_spent = nodes * 4
    if budget < passive_spent + batch:
        raise ValueError('Budget must cover shared passive assay plus one batch')
    out = []
    test_rng = np.random.default_rng(seed + 10017)
    test = features(test_rng.uniform(-2, 2, (1024, 2)))
    for method in ('scratch_uniform', 'warm_uniform', 'warm_residual', 'module_residual'):
        arm_rng = np.random.default_rng(seed + {'scratch_uniform': 43, 'warm_uniform': 47,
                                               'warm_residual': 53, 'module_residual': 59}[method])
        mean = np.zeros((nodes, 6)) if method.startswith('scratch') else old.copy()
        prior_mean = mean.copy()
        precision = 0.25 if method.startswith('scratch') else 20.0
        info = np.repeat(np.eye(6)[None, :, :] * precision, nodes, axis=0)
        rhs = prior_mean * precision
        residual = np.zeros(nodes)
        for i in range(nodes):
            phi = features(passive_x[i])
            residual[i] = float(np.mean((passive_y[i] - phi @ mean[i]) ** 2))
            info[i] += phi.T @ phi / 0.15**2
            rhs[i] += phi.T @ passive_y[i] / 0.15**2
        if method == 'module_residual':
            # A predicted mechanism change triggers a wider, family-level prior.
            # Selection uses only common passive data, never changed_ids.
            flagged = np.argsort(residual)[-changed:]
            for i in flagged:
                info[i] -= np.eye(6) * precision
                rhs[i] -= prior_mean[i] * precision
                prior_mean[i] = family[form_ids[i]]
                info[i] += np.eye(6) * 0.25
                rhs[i] += prior_mean[i] * 0.25
        spent = passive_spent
        rounds = 0
        while spent + batch <= budget:
            if method.endswith('uniform'):
                target = rounds % nodes
            else:
                estimate = np.linalg.solve(info, rhs[..., None])[..., 0]
                # Residual ranking is updated with data; use a small exploration term.
                target = int(np.argmax(residual + 0.02 * np.array([np.trace(np.linalg.inv(m)) for m in info])))
            x = arm_rng.uniform(-2, 2, (batch, 2))
            phi = features(x)
            y = phi @ truth[target] + arm_rng.normal(0, 0.15, batch)
            info[target] += phi.T @ phi / 0.15**2
            rhs[target] += phi.T @ y / 0.15**2
            estimate_target = np.linalg.solve(info[target], rhs[target])
            residual[target] = float(np.mean((y - phi @ estimate_target) ** 2))
            spent += batch
            rounds += 1
            if rounds in (1, 5, 10, 20) or spent + batch > budget:
                estimate = np.linalg.solve(info, rhs[..., None])[..., 0]
                errors = np.mean(np.einsum('nd,md->nm', estimate - truth, test, optimize=False) ** 2, axis=1)
                mask = np.zeros(nodes, dtype=bool)
                mask[changed_ids] = True
                out.append(dict(track='transfer', seed=seed, method=method, budget=spent,
                                nodes=nodes, changed=changed, rounds=rounds,
                                mse=float(np.mean(errors)), changed_mse=float(np.mean(errors[mask])),
                                unchanged_mse=float(np.mean(errors[~mask]))))
    return out


def write_results(rows: list[dict], output: Path, config: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    # Distinct files per cell; final receipt is written last, atomically.
    csv_path = output / 'metrics.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {'schema_version': 1, 'config': config, 'rows': len(rows),
               'metrics_file': csv_path.name, 'all_finite': all(
                   np.isfinite(row['mse']) for row in rows), 'source': 'agenda_runner.py',
               'source_revision': os.environ.get('ACE_SOURCE_REVISION', 'local'),
               'metrics_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest()}
    if not receipt['all_finite']:
        raise ValueError('Nonfinite primary metric')
    temporary = output / 'complete.json.tmp'
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    os.replace(temporary, output / 'complete.json')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--track', required=True, choices=('prior', 'design', 'transfer'))
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--budget', type=int, default=400)
    p.add_argument('--nodes', type=int, default=30)
    p.add_argument('--changed', type=int, default=3)
    p.add_argument('--background-sd', type=float, default=0.15)
    p.add_argument('--actuator-penalty', type=int, default=0)
    args = p.parse_args()
    start = time.monotonic()
    if args.track == 'prior':
        rows = prior_experiment(args.seed)
    elif args.track == 'design':
        rows = design_experiment(args.seed, args.budget, background_sd=args.background_sd,
                                 actuator_penalty=args.actuator_penalty)
    else:
        rows = transfer_experiment(args.seed, args.nodes, args.changed, args.budget)
    config = vars(args).copy()
    config['output'] = str(args.output)
    config['elapsed_seconds'] = time.monotonic() - start
    write_results(rows, args.output, config)
    print(json.dumps({'track': args.track, 'seed': args.seed, 'rows': len(rows),
                      'seconds': round(config['elapsed_seconds'], 3), 'output': str(args.output)}))


if __name__ == '__main__':
    main()

"""Target-prediction acquisition from the public linear-Gaussian SCM.

This CPU-only reference uses no environment or private evaluator. Structural
moments are exact for independent local parameter posteriors on a DAG. The
one-response lookahead uses Monte Carlo fantasy receipts and the same eligible
row update as the learner. Its expected-risk estimate is not exact, and its
Monte Carlo standard error is not a scientific performance interval.

The query law must be declared publicly before data collection and supplied to
every policy. It is not a list of private evaluation responses. This module
does not yet implement that shared task descriptor in the LLM interface.
"""
from __future__ import annotations

import hashlib

import numpy as np

from llm_intervention_contract import Action, finite, integer, validate_design
from llm_intervention_numerical_policy import restore


def structural_moments(learner, action: Action):
    """Return exact mean/covariance of the noise-disabled composed mapping.

    A fresh coefficient vector is drawn conceptually once per mechanism.
    Shared-ancestor correlations and within-head coefficient covariance are
    retained. Known clamps have zero variance. No aleatoric noise is added.
    These are posterior uncertainties under the working model, not guarantees
    about an unknown or misspecified environment.
    """
    graph = learner.graph
    index = {node.name: i for i, node in enumerate(graph)}
    clamps = learner._clamps(action)
    means = np.zeros(len(graph))
    second = np.zeros((len(graph), len(graph)))
    for i, node in enumerate(graph):
        if node.name in clamps:
            means[i] = clamps[node.name]
            second[i, :i] = means[i] * means[:i]
            second[:i, i] = second[i, :i]
            second[i, i] = means[i] ** 2
            continue
        head = learner.heads[node.name]
        parents = [index[p] for p in node.parents]
        phi_mean = np.r_[1., means[parents]]
        phi_second = np.empty((len(parents)+1, len(parents)+1))
        phi_second[0, :] = phi_mean
        phi_second[:, 0] = phi_mean
        phi_second[1:, 1:] = second[np.ix_(parents, parents)]
        means[i] = head.mean @ phi_mean
        # Current coefficients are independent of every earlier node.
        second[i, :i] = head.mean[0] * means[:i]
        if parents:
            second[i, :i] += head.mean[1:] @ second[parents, :i]
        second[:i, i] = second[i, :i]
        coef_second = head.covariance + np.outer(head.mean, head.mean)
        second[i, i] = np.sum(coef_second * phi_second)
    covariance = second - np.outer(means, means)
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(means).all() or not np.isfinite(covariance).all():
        raise ValueError("nonfinite composed moments")
    tolerance = 1e-10 * max(1., float(np.max(np.abs(second))))
    if np.linalg.eigvalsh(covariance).min() < -tolerance:
        raise ValueError("invalid composed covariance")
    return means, covariance


def _validate_law(graph, target, queries, weights, scale):
    if target not in {node.name for node in graph}:
        raise ValueError("unknown target")
    if (type(queries) is not tuple or not 1 <= len(queries) <= 64
            or type(weights) is not tuple or len(weights) != len(queries)
            or not all(finite(w) and w > 0 for w in weights)
            or not np.isclose(sum(weights), 1., atol=1e-12, rtol=0)
            or not finite(scale) or scale <= 0):
        raise ValueError("fixed public probability law and positive scale required")
    validate_design(graph, queries)


def integrated_risk(learner, *, target, queries, weights, scale):
    """Posterior squared-error risk of the posterior-mean structural mapping."""
    _validate_law(learner.graph, target, queries, weights, scale)
    k = [node.name for node in learner.graph].index(target)
    variances = []
    for query in queries:
        _, covariance = structural_moments(learner, query)
        # Only numerical roundoff can be clipped here, after the PSD check.
        variances.append(max(0., float(covariance[k, k])) / scale**2)
    risk = float(np.dot(weights, variances))
    if not finite(risk):
        raise ValueError("nonfinite integrated risk")
    return risk


def choose_target_risk(public_state, *, target, queries, weights, scale, seed, draws=32):
    """Choose estimated one-response structural-risk reduction per paid unit.

    Candidate fantasies use stable action-specific RNG streams. Fantasy
    descendants use sampled parents. Only the advertised measurements and
    known hard clamps enter each independent branch update. Failed candidates
    invalidate the call rather than disappearing from the comparison.
    Negative estimated gains are retained. The policy spends its common budget
    and only stops for a terminal state or no affordable action. It does not
    claim a calibrated autonomous stopping rule.
    """
    if not integer(seed) or not integer(draws, 2) or draws > 256:
        raise ValueError("fixed nonnegative seed and 2..256 draws required")
    graph, menu, learner = restore(public_state)
    _validate_law(graph, target, queries, weights, scale)
    if not integer(public_state["remaining_units"]) or type(public_state["terminal"]) is not bool:
        raise ValueError("invalid budget/terminal state")
    feasible = sorted((a for a in menu if a.cost_units <= public_state["remaining_units"]),
                      key=lambda a: a.action_id)
    if public_state["terminal"] or not feasible:
        return {"stop": True}, {"fantasy_responses": 0, "environment_responses": 0}
    before = integrated_risk(learner, target=target, queries=queries, weights=weights, scale=scale)
    scores, afters, errors = {}, {}, {}
    for action in feasible:
        action_seed = int.from_bytes(hashlib.sha256(action.action_id.encode()).digest()[:16], "big")
        rng = np.random.default_rng(np.random.SeedSequence([seed, action_seed]))
        risks = []
        for _ in range(draws):
            fantasy = learner.simulate(action, rng)
            branch = learner.fork()
            branch.update({"request_id": 0, "targets": dict(action.targets),
                           "values": {name: fantasy[name] for name in action.measure}})
            risks.append(integrated_risk(branch, target=target, queries=queries,
                                         weights=weights, scale=scale))
        afters[action.action_id] = float(np.mean(risks))
        scores[action.action_id] = (before - afters[action.action_id]) / action.cost_units
        errors[action.action_id] = float(np.std(risks, ddof=1) / np.sqrt(draws) / action.cost_units)
    if not all(finite(v) for mapping in (scores, afters, errors) for v in mapping.values()):
        raise ValueError("nonfinite candidate risk estimate")
    best = min(feasible, key=lambda a: (-scores[a.action_id], a.action_id))
    return {"action_id": best.action_id}, {
        "target": target, "risk_before": before, "risk_after": afters,
        "risk_reduction_per_cost": scores, "monte_carlo_standard_errors": errors,
        "fantasy_responses": draws*len(feasible), "environment_responses": 0,
        "structural_queries": len(queries)*(1+draws*len(feasible)),
        "scope": "working-model one-step estimate, not an empirical utility result"}

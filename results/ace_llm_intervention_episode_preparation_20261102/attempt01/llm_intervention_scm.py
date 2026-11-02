"""Exact linear-Gaussian mechanism learner for contract/teacher qualification.

This is the tractable reference in the agenda, not the final nonlinear learner
or a benchmark chosen to favor language. It receives paid observations only.
No environment truth, world generator or private evaluator is imported here.
Conjugate inference is exact for retained local regression factors with known
independent noise and error-free covariates. Dropping genuinely missing-parent
rows is not full latent-variable inference over partially observed receipts.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from llm_intervention_contract import Action, MechanismSummary, Node, finite, integer


@dataclass
class Head:
    mean: np.ndarray
    covariance: np.ndarray
    noise_variance: float
    eligible_rows: int = 0


class LinearGaussianSCM:
    def __init__(self, graph: tuple[Node, ...], noise_variances: dict[str, float],
                 prior_variance: float = 1.):
        if (type(graph) is not tuple or not graph or not finite(prior_variance)
                or prior_variance <= 0 or set(noise_variances) != {n.name for n in graph}):
            raise ValueError("graph, noise specification and positive prior required")
        seen = set()
        self.graph = graph
        self.heads = {}
        self._requests = set()
        for node in graph:
            if node.name in seen or not set(node.parents) <= seen:
                raise ValueError("graph must have unique nodes in topological order")
            variance = noise_variances[node.name]
            if not finite(variance) or variance <= 0:
                raise ValueError("strictly positive noise variance required")
            seen.add(node.name)
            d = 1 + len(node.parents)
            self.heads[node.name] = Head(np.zeros(d), np.eye(d)*prior_variance, float(variance))

    def fork(self):
        """Independent learner state for candidate branches, including row identity."""
        return copy.deepcopy(self)

    def update(self, receipt: dict) -> tuple[str, ...]:
        request = receipt["request_id"]
        values, targets = receipt["values"], receipt["targets"]
        if not integer(request) or request in self._requests:
            raise ValueError("invalid or duplicate observation identity")
        if (not isinstance(values, dict) or not isinstance(targets, dict)
                or not set(values) <= set(self.heads) or not set(targets) <= set(self.heads)
                or not all(finite(x) for x in values.values())
                or not all(finite(x) for x in targets.values())):
            raise ValueError("invalid measured values or target mask")
        for node in self.graph:
            if node.name in targets:
                bounds = node.intervention_bounds
                value = targets[node.name]
                if bounds is None or not bounds[0] <= value <= bounds[1]:
                    raise ValueError("invalid hard intervention")
                if node.name in values and values[node.name] != value:
                    raise ValueError("measured clamp contradicts hard-intervention semantics")
        # A hard clamp is known even when that variable was not separately read.
        available = dict(values, **targets)
        updates = {}
        for node in self.graph:
            if (node.name in targets or node.name not in values
                    or not set(node.parents) <= set(available)):
                continue
            head = self.heads[node.name]
            phi = np.array([1.] + [available[p] for p in node.parents])
            projected = head.covariance @ phi
            denominator = head.noise_variance + phi @ projected
            mean = head.mean + projected * ((values[node.name] - phi @ head.mean) / denominator)
            covariance = head.covariance - np.outer(projected, projected) / denominator
            covariance = (covariance + covariance.T) / 2
            if (not np.isfinite(mean).all() or not np.isfinite(covariance).all()
                    or denominator <= 0 or np.linalg.eigvalsh(covariance).min() < -1e-10):
                raise ValueError("invalid posterior update")
            updates[node.name] = Head(mean, covariance, head.noise_variance, head.eligible_rows+1)
        # Commit only after every eligible head update has passed validation.
        self.heads.update(updates)
        self._requests.add(request)
        return tuple(updates)

    def summaries(self) -> tuple[MechanismSummary, ...]:
        result = []
        for node in self.graph:
            head = self.heads[node.name]
            result.append(MechanismSummary(node.name, ("intercept",) + node.parents,
                          tuple(float(x) for x in head.mean),
                          tuple(tuple(float(x) for x in row) for row in head.covariance),
                          head.noise_variance, head.eligible_rows))
        return tuple(result)

    def _clamps(self, action: Action) -> dict:
        targets = dict(action.targets)
        nodes = {n.name: n for n in self.graph}
        for name, value in targets.items():
            bounds = nodes[name].intervention_bounds if name in nodes else None
            if bounds is None or not bounds[0] <= value <= bounds[1]:
                raise ValueError("invalid model-query intervention")
        return targets

    def predict_mean(self, action: Action) -> dict[str, float]:
        """Compose posterior-mean mechanisms using predicted parents."""
        values = self._clamps(action)
        for node in self.graph:
            if node.name not in values:
                phi = np.array([1.] + [values[p] for p in node.parents])
                values[node.name] = float(phi @ self.heads[node.name].mean)
        if not all(map(finite, values.values())):
            raise ValueError("non-finite composed prediction")
        return values

    def simulate(self, action: Action, rng: np.random.Generator, *, observation_noise=True):
        """One student-predictive draw, not a paid environment response.

        Each draw samples the complete set of mechanism parameters once.
        This preserves within-mechanism covariance and predicted-parent flow.
        The caller separately counts these computation-only planning draws.
        """
        values = self._clamps(action)
        for node in self.graph:
            if node.name not in values:
                head = self.heads[node.name]
                coef = rng.multivariate_normal(head.mean, head.covariance, check_valid="raise")
                phi = np.array([1.] + [values[p] for p in node.parents])
                noise = rng.normal(0, np.sqrt(head.noise_variance)) if observation_noise else 0.
                values[node.name] = float(phi @ coef + noise)
        if not all(map(finite, values.values())):
            raise ValueError("non-finite student-predictive draw")
        return values

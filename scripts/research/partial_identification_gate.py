#!/usr/bin/env python3
"""Exact finite SCM gate for observational ambiguity and local belief repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path


def distribution(model: str, action: int | None) -> dict[str, Fraction]:
    out: dict[str, Fraction] = {}
    for u in (0, 1):
        x = u if action is None else action
        y = 0 if model == "constant" else x ^ u
        key = f"x{x}y{y}"
        out[key] = out.get(key, Fraction(0)) + Fraction(1, 2)
    return out


def mean_y(dist: dict[str, Fraction]) -> Fraction:
    return sum((p for key, p in dist.items() if key.endswith("y1")), Fraction(0))


def serialize(dist: dict[str, Fraction]) -> dict[str, str]:
    return {key: str(value) for key, value in sorted(dist.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if os.environ.get("ACE_SOURCE_REVISION") != revision:
        raise ValueError("ACE_SOURCE_REVISION must equal HEAD")

    models = ("constant", "xor_latent")
    obs = {model: distribution(model, None) for model in models}
    interventions = {str(action): {model: distribution(model, action) for model in models}
                     for action in (0, 1)}
    assert obs[models[0]] == obs[models[1]]
    assert mean_y(interventions["1"]["constant"]) == 0
    assert mean_y(interventions["1"]["xor_latent"]) == Fraction(1, 2)
    # One do(X=1), Y=1 observation is impossible under the constant model.
    likelihoods = {model: mean_y(interventions["1"][model]) for model in models}
    posterior = {model: likelihoods[model] / sum(likelihoods.values()) for model in models}
    assert posterior == {"constant": 0, "xor_latent": 1}

    result = {
        "source_revision": revision,
        "candidate_models": {
            "constant": "U~Bernoulli(1/2); X=U; Y=0",
            "xor_latent": "U~Bernoulli(1/2); X=U; Y=X XOR U",
        },
        "observational_joint_distribution": serialize(obs["constant"]),
        "observational_distributions_equal": True,
        "candidate_set_mean_y_by_action": {
            action: {model: str(mean_y(dist)) for model, dist in by_model.items()}
            for action, by_model in interventions.items()
        },
        "do_x1_y1_event_probability": {model: str(value) for model, value in likelihoods.items()},
        "posterior_after_do_x1_y1_equal_prior": {model: str(value) for model, value in posterior.items()},
        "candidate_set_identified_before_intervention": False,
        "candidate_set_identified_after_event": True,
        "closed_model_calls": 0,
        "acquired_training_queries": 0,
    }
    args.output.mkdir(parents=True)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (args.output / "result.json").write_text(payload)
    (args.output / "complete.json").write_text(json.dumps({
        "source_revision": revision,
        "result_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "exact_enumeration": True,
        "closed_model_calls": 0,
    }, indent=2, sort_keys=True) + "\n")
    print("PASS: observational equivalence, intervention disagreement, and Bayesian repair")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score discriminating and null menus for two observationally equal SCMs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path


def entropy(probabilities: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def information_gain(response_laws: list[list[float]]) -> float:
    """Model prior is uniform; rows are outcome probabilities by candidate SCM."""
    mean_law = [sum(row[i] for row in response_laws) / len(response_laws)
                for i in range(len(response_laws[0]))]
    return entropy(mean_law) - sum(entropy(row) for row in response_laws) / len(response_laws)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if os.environ.get("ACE_SOURCE_REVISION") != revision:
        raise ValueError("ACE_SOURCE_REVISION must equal HEAD")

    # Under do(X=1), Y is always zero in M0 and Bernoulli(1/2) in M1.
    # Under do(Z=1), Z is causally irrelevant and observed (X,Y) laws agree.
    laws = {
        "do_x1": [[1.0, 0.0], [0.5, 0.5]],
        "do_z1": [[1.0, 0.0], [1.0, 0.0]],
    }
    gains = {action: information_gain(rows) for action, rows in laws.items()}
    expected = entropy([0.75, 0.25]) - 0.5
    assert math.isclose(gains["do_x1"], expected, abs_tol=1e-12)
    assert gains["do_z1"] == 0
    menus = {"discriminating": ["do_x1", "do_z1"], "null": ["do_z1"]}
    decisions = {}
    for menu, actions in menus.items():
        best = max(actions, key=lambda action: gains[action])
        decisions[menu] = best if gains[best] > 0 else "abstain_unresolved"
    assert decisions == {"discriminating": "do_x1", "null": "abstain_unresolved"}

    result = {"source_revision": revision, "candidate_models": ["constant", "xor_latent"],
              "action_costs": {"do_x1": 1, "do_z1": 1},
              "information_gain_bits": gains, "menus": menus, "oracle_decisions": decisions,
              "remaining_candidate_count_in_null_menu": 2,
              "simulator_queries": 0, "closed_model_calls": 0}
    args.output.mkdir(parents=True)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    (args.output / "result.json").write_text(payload)
    (args.output / "complete.json").write_text(json.dumps({
        "source_revision": revision, "result_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "menus_validated": 2, "simulator_queries": 0, "closed_model_calls": 0,
    }, sort_keys=True, indent=2) + "\n")
    print("PASS: discriminating menu selects do(X=1); null menu abstains")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministic safety boundary for future language-to-action proposals."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def validate(schema: dict, proposal: dict) -> tuple[bool, str]:
    if set(proposal) != {"targets", "values"}:
        return False, "unknown_or_missing_field"
    targets, values = proposal["targets"], proposal["values"]
    if not isinstance(targets, list) or not isinstance(values, list) or len(targets) != len(values):
        return False, "malformed_action"
    if not targets or len(targets) != len(set(targets)):
        return False, "empty_or_duplicate_targets"
    if any(not isinstance(target, str) or target not in schema["actuators"] for target in targets):
        return False, "unknown_actuator"
    if len(targets) > schema["max_joint_targets"]:
        return False, "joint_limit"
    if any(target in schema["excluded_actuators"] for target in targets):
        return False, "excluded_actuator"
    if any(type(value) not in (int, float) or value not in schema["allowed_values"][target]
           for target, value in zip(targets, values)):
        return False, "invalid_value"
    if sum(schema["actuators"][target]["cost"] for target in targets) > schema["cost_cap"]:
        return False, "cost_cap"
    return True, "valid"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if os.environ.get("ACE_SOURCE_REVISION") != revision:
        raise ValueError("ACE_SOURCE_REVISION must equal HEAD")
    schema = {
        "actuators": {
            "x": {"cost": 2, "affects": ["x"]},
            "z": {"cost": 2, "affects": ["z"]},
            "proxy_p": {"cost": 3, "affects": ["x", "z"]},
            "danger": {"cost": 1, "affects": ["x"]},
        },
        "allowed_values": {key: [-1, 1] for key in ("x", "z", "proxy_p", "danger")},
        "excluded_actuators": ["danger"],
        "max_joint_targets": 2,
        "cost_cap": 4,
    }
    cases = [
        ({"targets": ["x", "z"], "values": [1, -1]}, True, "valid"),
        ({"targets": ["proxy_p"], "values": [1]}, True, "valid"),
        ({"targets": ["x", "z", "proxy_p"], "values": [1, 1, 1]}, False, "joint_limit"),
        ({"targets": ["danger"], "values": [1]}, False, "excluded_actuator"),
        ({"targets": ["x", "proxy_p"], "values": [1, 1]}, False, "cost_cap"),
        ({"targets": ["x"], "values": [0]}, False, "invalid_value"),
        ({"targets": ["x", "x"], "values": [1, -1]}, False, "empty_or_duplicate_targets"),
        ({"targets": ["unknown"], "values": [1]}, False, "unknown_actuator"),
        ({"targets": ["x"], "values": [True]}, False, "invalid_value"),
        ({"targets": ["x"], "values": [1], "hidden": 1}, False, "unknown_or_missing_field"),
    ]
    rows = []
    for proposal, expected_valid, expected_reason in cases:
        valid, reason = validate(schema, proposal)
        assert (valid, reason) == (expected_valid, expected_reason)
        rows.append({"proposal": proposal, "valid": valid, "reason": reason})
    args.output.mkdir(parents=True)
    payload = json.dumps({"source_revision": revision, "schema": schema, "cases": rows,
                          "cases_passed": len(rows), "executed_invalid_actions": 0,
                          "simulator_queries": 0, "closed_model_calls": 0},
                         sort_keys=True, indent=2) + "\n"
    (args.output / "result.json").write_text(payload)
    (args.output / "complete.json").write_text(json.dumps({
        "source_revision": revision, "result_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "cases_passed": len(rows), "executed_invalid_actions": 0,
    }, sort_keys=True, indent=2) + "\n")
    print(f"PASS: {len(rows)} deterministic action validation cases")


if __name__ == "__main__":
    main()

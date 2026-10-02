#!/usr/bin/env python3
"""Audit an independently authored action-language fixture and strict proposals.

Input is JSONL with one object per task. Gold menus are enumerated from the
formal schema; they are never supplied to a policy. A proposal file, when
provided, contains one JSON object per task with exactly ``id`` and ``actions``.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from action_menu_validator_smoke import validate


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every row must be an object")
    return rows


def action_key(action: dict) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def audit_fixture(rows: list[dict], minimum: int) -> dict[str, set[str]]:
    if len(rows) < minimum:
        raise ValueError(f"need at least {minimum} independently authored tasks")
    ids: set[str] = set()
    gold: dict[str, set[str]] = {}
    for row in rows:
        if set(row) != {"id", "description", "provenance", "schema", "gold_actions"}:
            raise ValueError("fixture requires id, description, provenance, schema, gold_actions")
        task_id = row["id"]
        if not isinstance(task_id, str) or not task_id or task_id in ids:
            raise ValueError(f"invalid or duplicate task ID: {task_id!r}")
        ids.add(task_id)
        if not isinstance(row["description"], str) or not row["description"].strip():
            raise ValueError(f"{task_id}: empty description")
        provenance = row["provenance"]
        if not isinstance(provenance, dict) or not all(
            isinstance(provenance.get(field), str) and provenance[field].strip()
            for field in ("author", "source", "authored_at", "gold_adjudicator")
        ):
            raise ValueError(f"{task_id}: incomplete independent provenance")
        if provenance["author"] == provenance["gold_adjudicator"]:
            raise ValueError(f"{task_id}: author and gold adjudicator must differ")
        schema = row["schema"]
        if not isinstance(schema, dict) or not isinstance(schema.get("actuators"), dict):
            raise ValueError(f"{task_id}: malformed schema")
        if not isinstance(row["gold_actions"], list):
            raise ValueError(f"{task_id}: gold_actions must be a list")
        keys = []
        for action in row["gold_actions"]:
            accepted, reason = validate(schema, action)
            if not accepted:
                raise ValueError(f"{task_id}: illegal gold action: {reason}")
            keys.append(action_key(action))
        if len(keys) != len(set(keys)):
            raise ValueError(f"{task_id}: duplicate gold action")
        names = list(schema["actuators"])
        expected = set()
        for size in range(1, min(len(names), schema["max_joint_targets"]) + 1):
            for targets in itertools.combinations(names, size):
                for values in itertools.product(*(schema["allowed_values"][name] for name in targets)):
                    action = {"targets": list(targets), "values": list(values)}
                    if validate(schema, action)[0]:
                        expected.add(action_key(action))
        if set(keys) != expected:
            raise ValueError(f"{task_id}: gold menu does not enumerate all legal actions")
        gold[task_id] = set(keys)
    return gold


def score(rows: list[dict], gold: dict[str, set[str]], proposals: list[dict]) -> list[dict]:
    by_id = {}
    for proposal in proposals:
        if set(proposal) != {"id", "actions"} or proposal["id"] in by_id:
            raise ValueError("proposals require unique id and actions fields only")
        by_id[proposal["id"]] = proposal
    if set(by_id) != set(gold):
        raise ValueError("proposal IDs do not match fixture IDs")
    output = []
    for row in rows:
        task_id = row["id"]
        actions = by_id[task_id]["actions"]
        if not isinstance(actions, list):
            raise ValueError(f"{task_id}: actions must be a list")
        keys = []
        invalid = 0
        for action in actions:
            accepted, _ = validate(row["schema"], action)
            if not accepted:
                invalid += 1
            else:
                keys.append(action_key(action))
        valid_set = set(keys)
        output.append({"id": task_id, "gold_count": len(gold[task_id]),
                       "predicted_count": len(actions), "invalid_proposals": invalid,
                       "duplicate_proposals": len(keys) - len(valid_set),
                       "false_legal": len(valid_set - gold[task_id]),
                       "recalled": len(valid_set & gold[task_id]),
                       "exact_menu": invalid == 0 and len(keys) == len(valid_set)
                       and valid_set == gold[task_id],
                       "abstained": len(actions) == 0})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--minimum", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_jsonl(args.fixture)
    gold = audit_fixture(rows, args.minimum)
    report = {"tasks": len(rows), "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              "source_count": len({row["provenance"]["source"] for row in rows}),
              "author_count": len({row["provenance"]["author"] for row in rows}),
              "gold_action_count": sum(map(len, gold.values())),
              "simulator_queries": 0, "model_calls": 0, "executed_invalid_actions": 0}
    if args.proposals:
        results = score(rows, gold, read_jsonl(args.proposals))
        report["proposals_sha256"] = hashlib.sha256(args.proposals.read_bytes()).hexdigest()
        report["scores"] = results
        report["exact_menus"] = sum(item["exact_menu"] for item in results)
        report["invalid_proposals"] = sum(item["invalid_proposals"] for item in results)
        report["false_legal"] = sum(item["false_legal"] for item in results)
        report["recalled"] = sum(item["recalled"] for item in results)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(f"audited {len(rows)} tasks; proposal scoring: {bool(args.proposals)}")


if __name__ == "__main__":
    main()

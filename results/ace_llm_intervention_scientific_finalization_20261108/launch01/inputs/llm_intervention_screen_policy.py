"""Prospective numerical controls over the existing public task envelope.

This is a new CLI. It does not change the historical numerical CLI seed law.
Bindings are declarations, not scientific admission or a malicious-code sandbox.
Training behavior and snapshot teachers are deliberately unsupported here.
"""
from __future__ import annotations

import argparse
import json
import sys

from llm_intervention_contract import _unique_object, canonical, digest, integer
from llm_intervention_numerical_policy import choose as choose_numerical
from llm_intervention_screen_protocol import (
    DEVELOPMENT_START, VERSION as PROPOSAL_VERSION,
)
from llm_intervention_target_risk import choose_target_risk
from llm_intervention_task import MAX_HISTORY, unpack_envelope
from llm_intervention_world import VERSION as WORLD_VERSION, keyed_seed


VERSION = "ace-screen-public-numerical-policy-20261107-v1"
METHODS = ("uniform_random", "menu_coverage", "local_information", "target_risk")
PLANNING_DRAWS = 32
MAX_INPUT_BYTES = 2 * 1024 * 1024


def _name(value):
    return type(value) is str and 0 < len(value) <= 256


def _pin(value):
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def bind_policy(*, seed_prefix, scope, world_id, request_namespace, arm, role,
                initial_actions, initial_cost_units, total_budget, planning_draws):
    """Pure detached binding from fixed public/control declarations, without draws.

    Development uses the proposal's exact arm-specific trajectory namespace.
    Artificial fixtures use the same address construction with artificial scope.
    No private world descriptor, coefficient, outcome or score is accepted.
    Initial values are absent: only the declared schedule and cost are bound.
    """
    if (type(scope) is not str or scope not in ("artificial", "development")
            or not integer(world_id) or world_id >= 2**63
            or type(arm) is not str or arm not in METHODS or role != "behavior"
            or type(role) is not str):
        raise ValueError("only artificial/development numerical behavior bindings supported")
    if scope == "development" and not DEVELOPMENT_START <= world_id < DEVELOPMENT_START + 24:
        raise ValueError("world ID outside the current proposed development range")
    namespace = digest({"protocol": PROPOSAL_VERSION, "scope": scope,
                        "world_id": world_id, "purpose": "trajectory", "arm": arm})
    prefix = [scope, world_id, "policy", namespace, "behavior"]
    if (type(seed_prefix) not in (tuple, list) or not _pin(request_namespace)
            or request_namespace != namespace
            or canonical(seed_prefix) != canonical(prefix)):
        raise ValueError("seed prefix, scope, identity, arm and proposal namespace disagree")
    if (type(initial_actions) not in (tuple, list) or len(initial_actions) > MAX_HISTORY
            or any(not _name(a) for a in initial_actions)
            or not integer(initial_cost_units) or not integer(total_budget, 1)
            or not len(initial_actions) <= initial_cost_units <= total_budget <= 10**6
            or (not initial_actions and initial_cost_units != 0)
            or type(planning_draws) is not int or planning_draws != PLANNING_DRAWS):
        raise ValueError("fixed bounded initial schedule, budget and exactly 32 planning draws required")
    if scope == "development" and (list(initial_actions) != ["observe_all"] * 8
            or initial_cost_units != 8 or total_budget != 32):
        raise ValueError("development requires the proposed eight initial rows and 32-unit budget")
    return {"version": VERSION, "proposal_version": PROPOSAL_VERSION,
            "seed_prefix": prefix, "scope": scope, "world_id": world_id,
            "request_namespace": namespace, "arm": arm, "role": role,
            "initial_actions": list(initial_actions), "initial_cost_units": initial_cost_units,
            "total_budget": total_budget, "planning_draws": planning_draws}


def validate_binding(binding, *, expected_sha256):
    """Require an independently supplied pin and the exact binding schema."""
    fields = {"version", "proposal_version", "seed_prefix", "scope", "world_id",
              "request_namespace", "arm", "role", "initial_actions",
              "initial_cost_units", "total_budget", "planning_draws"}
    if (type(binding) is not dict or set(binding) != fields or not _pin(expected_sha256)
            or digest(binding) != expected_sha256):
        raise ValueError("binding schema or independent SHA256 differs")
    rebuilt = bind_policy(**{k: binding[k] for k in fields - {"version", "proposal_version"}})
    if canonical(rebuilt) != canonical(binding):
        raise ValueError("binding differs from this proposal/interface version")
    return rebuilt


def decision_rng(envelope, *, binding, expected_binding_sha256, expected_decision_index=None):
    """Derive a policy-call index from a normal, complete public history.

    The trusted episode loop must stop on any failed, rejected or ambiguous call.
    Each preceding successful acquisition call then has exactly one receipt.
    This stateless helper cannot prove the absence of hidden retries. The caller
    may cross-check its own call counter with expected_decision_index, and the
    outer supervisor must reconcile ProcessPolicy filenames/journals/receipts.
    """
    b = validate_binding(binding, expected_sha256=expected_binding_sha256)
    state, _, _, menu, _ = unpack_envelope(envelope)
    if state["terminal"]:
        raise ValueError("no policy call is allowed after a terminal state")
    count = len(b["initial_actions"])
    if (state["requests"] < count
            or [r["action_id"] for r in state["history"][:count]] != b["initial_actions"]):
        raise ValueError("fixed initial history is incomplete or differs")
    actions = {a.action_id: a for a in menu}
    initial_cost = sum(actions[a].cost_units for a in b["initial_actions"])
    if (initial_cost != b["initial_cost_units"]
            or sum(r["cost_units"] for r in state["history"][:count]) != initial_cost
            or state["spent_units"] + state["remaining_units"] != b["total_budget"]
            or count + (b["total_budget"] - initial_cost) // min(a.cost_units for a in menu) > MAX_HISTORY):
        raise ValueError("initial charges, total budget or response bound differs")
    if not any(a.cost_units <= state["remaining_units"] for a in menu):
        raise ValueError("budget termination belongs to the executor, not another policy call")
    index = state["requests"] - count
    if expected_decision_index is not None and (
            not integer(expected_decision_index) or expected_decision_index != index):
        raise ValueError("policy call counter disagrees with completed acquisition history")
    # Exact original world rule, with no truncation, extra arm component,
    # SeedSequence remix, cost index, initial response offset or global RNG use.
    seed = keyed_seed(*b["seed_prefix"], index)
    key = [WORLD_VERSION, b["scope"], b["world_id"], "policy",
           [b["request_namespace"], "behavior", index]]
    receipt = {"version": VERSION, "proposal_version": PROPOSAL_VERSION,
        "binding_sha256": expected_binding_sha256, "public_envelope_sha256": digest(envelope),
        "arm": b["arm"], "role": "behavior", "decision_index": index,
        "request_count": state["requests"], "initial_response_count": count,
        "initial_cost_units": initial_cost, "spent_units": state["spent_units"],
        "acquisition_cost_units": state["spent_units"] - initial_cost,
        "canonical_key": key, "canonical_key_json": canonical(key),
        "seed_decimal": str(seed), "seed_hex_256": format(seed, "064x"),
        "declared_planning_draws": PLANNING_DRAWS,
        "index_basis": "successful acquisition policy calls, not response cost or total requests"}
    return seed, receipt


def choose_public(envelope, *, binding, expected_binding_sha256, expected_decision_index=None):
    """Select using only the validated public state/task and declared policy key."""
    seed, receipt = decision_rng(envelope, binding=binding,
        expected_binding_sha256=expected_binding_sha256,
        expected_decision_index=expected_decision_index)
    state, task, _, _, _ = unpack_envelope(envelope)
    if receipt["arm"] == "target_risk":
        decision, planning = choose_target_risk(state, seed=seed, draws=PLANNING_DRAWS,
                                               **task.risk_arguments())
    else:
        decision, planning = choose_numerical(state, method=receipt["arm"], seed=seed,
                                              draws=PLANNING_DRAWS)
    receipt.update(status="selected", decision=decision, decision_sha256=digest(decision),
                   planning_metadata=planning, environment_responses=0,
                   scientific_provenance_qualified=False)
    return decision, receipt


def _read_json(raw):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=reject_constant)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--binding-json", required=True)
    parser.add_argument("--binding-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if len(args.binding_json.encode("utf-8")) > 4096:
            raise ValueError("binding exceeds fixed command argument bound")
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("public envelope exceeds byte bound")
        decision, receipt = choose_public(_read_json(raw), binding=_read_json(args.binding_json),
                                          expected_binding_sha256=args.binding_sha256)
        # ProcessPolicy preserves stderr together with command, input and output.
        # Exactly one JSON decision on stdout retains its existing ABI.
        print(canonical(receipt), file=sys.stderr)
        print(canonical(decision))
        return 0
    except Exception as exc:
        print(canonical({"version": VERSION, "status": "rejected", "retry_allowed": False,
                         "exception_type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

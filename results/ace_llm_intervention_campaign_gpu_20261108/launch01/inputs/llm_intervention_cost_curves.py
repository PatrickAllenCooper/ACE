"""Private, retrospective reporting from explicitly sealed paid records only.

No files, callbacks, generators, policy, model weights or training are executed.
The prospective runner must independently establish the pins, execution closure,
shared initialization, predeclared protocol and scientific split provenance.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from llm_intervention_contract import Action, canonical, digest, finite, integer, validate_design
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_task import PredictionTask
from llm_intervention_world import WorldDescriptor, private_structural_means


VERSION = "ace-offline-cost-curve-v1"
MAX_REQUESTS, MAX_MENU, MAX_BYTES = 80, 32, 8 * 1024 * 1024
BASE_FILES = frozenset(("world.json", "task.json", "learner.json", "plan.json",
                        "shared-initial.json", "history.json", "terminal.json"))
OUTCOMES = ("accepted", "returned_invalid", "returned_unaccepted", "unknown")


class CurveInputError(ValueError):
    """Rejected evidence must stay on the caller's roster, not be filtered out."""


def _require(condition, message):
    if not condition:
        raise CurveInputError(message)


def _keys(value, fields):
    _require(type(value) is dict and set(value) == set(fields), "unexpected record fields")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def _json(raw):
    def reject(value):
        raise CurveInputError("nonfinite JSON constant: " + value)
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=reject)
    canonical(value)  # also rejects overflowing float literals such as 1e999
    return value


def _action(value):
    _keys(value, ("action_id", "kind", "targets", "measure", "cost_units"))
    _require(type(value["action_id"]) is str and 0 < len(value["action_id"]) <= 256
             and type(value["kind"]) is str and type(value["targets"]) is list
             and all(type(t) is list and len(t) == 2 for t in value["targets"])
             and type(value["measure"]) is list
             and all(type(n) is str for n in value["measure"]), "invalid action fields")
    return Action(value["action_id"], value["kind"], tuple(tuple(t) for t in value["targets"]),
                  tuple(value["measure"]), value["cost_units"])


def _valid_values(value, action):
    return (type(value) is dict and set(value) == set(action.measure)
            and all(finite(x) for x in value.values())
            and all(value[n] == v for n, v in action.targets if n in value))


def _validate(inputs, pins):
    _require(type(inputs) is dict and type(pins) is dict and set(inputs) == set(pins)
             and BASE_FILES <= set(inputs) and len(inputs) <= len(BASE_FILES)+MAX_REQUESTS,
             "exact independently pinned bundle required")
    _require(all(type(k) is str and type(v) is bytes for k, v in inputs.items())
             and sum(len(v) for v in inputs.values()) <= MAX_BYTES, "bounded byte inputs required")
    for name, raw in inputs.items():
        _require(_is_sha(pins[name]) and _sha(raw) == pins[name], "byte pin mismatch: " + name)
    objects = {name: _json(inputs[name]) for name in BASE_FILES}
    plan, history, terminal = (objects[n] for n in ("plan.json", "history.json", "terminal.json"))
    _keys(plan, ("version", "scope", "world_id", "policy_id", "world_sha256", "task_sha256",
                 "learner_sha256", "shared_initial_sha256", "menu", "initial_actions",
                 "acquisition_cap_units", "threshold"))
    _require(plan["version"] == VERSION and type(plan["policy_id"]) is str
             and 0 < len(plan["policy_id"]) <= 256, "invalid plan identity")
    for field, name in (("world_sha256", "world.json"), ("task_sha256", "task.json"),
                        ("learner_sha256", "learner.json"), ("shared_initial_sha256", "shared-initial.json")):
        _require(plan[field] == pins[name], "plan input pin mismatch: " + field)
    world = WorldDescriptor.from_private_payload(objects["world.json"],
                                                 expected_sha256=digest(objects["world.json"]))
    _require(type(plan["scope"]) is str and plan["scope"] == world.scope
             and integer(plan["world_id"]) and plan["world_id"] == world.world_id, "world identity differs")
    task_data = objects["task.json"]
    _keys(task_data, ("target", "queries", "weights", "scale", "normalized_mse_tolerance"))
    _require(type(task_data["queries"]) is list and type(task_data["weights"]) is list,
             "task arrays required")
    task = PredictionTask(task_data["target"], tuple(_action(q) for q in task_data["queries"]),
                          tuple(task_data["weights"]), task_data["scale"], task_data["normalized_mse_tolerance"])
    task.validate(world.graph)
    _require(all(w == 1/len(task.queries) for w in task.weights)
             and all(task.target in q.measure for q in task.queries), "fixed equal-weight target queries required")
    _require(finite(plan["threshold"]) and plan["threshold"] == task.normalized_mse_tolerance,
             "threshold must equal the independently pinned public task tolerance")
    config = objects["learner.json"]
    _keys(config, ("graph", "noise_variances", "prior_variance"))
    _require(canonical(config["graph"]) == canonical(objects["world.json"]["graph"])
             and type(config["noise_variances"]) is dict, "learner graph/noise configuration differs")
    learner = LinearGaussianSCM(world.graph, config["noise_variances"], config["prior_variance"])
    _require(type(plan["menu"]) is list and 1 <= len(plan["menu"]) <= MAX_MENU, "bounded menu required")
    menu = tuple(_action(a) for a in plan["menu"])
    validate_design(world.graph, menu)
    actions = {a.action_id: a for a in menu}
    initial = plan["initial_actions"]
    _require(type(initial) is list and len(initial) <= MAX_REQUESTS
             and all(type(a) is str and a in actions for a in initial), "invalid fixed initial schedule")
    initial_cost = sum(actions[a].cost_units for a in initial)
    cap = plan["acquisition_cap_units"]
    _require(integer(cap, 1) and initial_cost+cap <= 10**6, "invalid common acquisition cap")
    shared = objects["shared-initial.json"]
    _keys(shared, ("receipt_sha256",))
    _require(type(shared["receipt_sha256"]) is list and len(shared["receipt_sha256"]) == len(initial)
             and all(_is_sha(p) for p in shared["receipt_sha256"]), "shared initial receipt pins required")
    _keys(history, ("version", "plan_sha256", "attempts"))
    _require(history["version"] == VERSION and history["plan_sha256"] == pins["plan.json"]
             and type(history["attempts"]) is list and len(history["attempts"]) <= MAX_REQUESTS,
             "invalid sealed history")
    _keys(terminal, ("version", "sealed", "input_sha256", "status", "reason", "attempt_count",
                     "accepted_count", "known_return_count", "initial_cost_units", "acquisition_cost_units"))
    _require(terminal["version"] == VERSION and terminal["sealed"] is True
             and terminal["input_sha256"] == {k: v for k, v in pins.items() if k != "terminal.json"},
             "terminal seal/inventory differs")
    _require(terminal["status"] in ("complete", "failed") and type(terminal["reason"]) is str
             and 0 < len(terminal["reason"]) <= 1024, "terminal status/reason required")
    counters = {outcome: {"attempts": 0, "cost_units": 0} for outcome in OUTCOMES}
    phases = {p: {"attempts": 0, "cost_units": 0} for p in ("initial", "acquisition")}
    used, accepted = set(BASE_FILES), []
    ended = False
    for request, event in enumerate(history["attempts"]):
        _keys(event, ("request_id", "phase", "action_id", "cost_units", "outcome", "evidence_file"))
        _require(not ended and integer(event["request_id"]) and event["request_id"] == request,
                 "reordered request or continuation after failed response")
        phase = "initial" if request < len(initial) else "acquisition"
        _require(event["phase"] == phase and type(event["action_id"]) is str
                 and event["action_id"] in actions, "phase/action mismatch")
        action = actions[event["action_id"]]
        _require(integer(event["cost_units"], 1) and event["cost_units"] == action.cost_units
                 and (phase != "initial" or action.action_id == initial[request]), "reservation cost/schedule differs")
        outcome = event["outcome"]
        _require(type(outcome) is str and outcome in OUTCOMES, "unknown response classification")
        filename = event["evidence_file"]
        expected_name = (f"receipt-{request:03d}.json" if outcome == "accepted" else
                         None if outcome == "unknown" else f"return-{request:03d}.bin")
        _require(filename == expected_name, "receipt/evidence order or identity differs")
        if filename is not None:
            _require(filename in inputs, "missing serialized response evidence")
            used.add(filename)
        if outcome == "accepted":
            receipt = _json(inputs[filename])
            _keys(receipt, ("request_id", "action_id", "kind", "targets", "values", "cost_units"))
            _require(integer(receipt["request_id"]) and receipt["request_id"] == request
                     and receipt["action_id"] == action.action_id and receipt["kind"] == action.kind
                     and type(receipt["targets"]) is dict and receipt["targets"] == dict(action.targets)
                     and all(finite(v) for v in receipt["targets"].values())
                     and integer(receipt["cost_units"], 1) and receipt["cost_units"] == action.cost_units
                     and _valid_values(receipt["values"], action), "accepted receipt contradicts reservation/clamps")
            if phase == "initial":
                _require(pins[filename] == shared["receipt_sha256"][request], "shared initial history differs")
            accepted.append(receipt)
        elif outcome != "unknown":
            # These are exact returned bytes, not an adapter's prepared-outcome log.
            try:
                valid = _valid_values(_json(inputs[filename]), action)
            except (ValueError, TypeError, OverflowError, RecursionError):
                valid = False
            _require(valid == (outcome == "returned_unaccepted"), "returned evidence contradicts its classification")
        ended = outcome != "accepted"
        counters[outcome]["attempts"] += 1
        counters[outcome]["cost_units"] += action.cost_units
        phases[phase]["attempts"] += 1
        phases[phase]["cost_units"] += action.cost_units
        _require(phases["acquisition"]["cost_units"] <= cap, "paid acquisition exceeds common cap")
    _require(set(inputs) == used, "unreferenced or extra evidence files")
    known_count = sum(counters[o]["attempts"] for o in OUTCOMES if o != "unknown")
    for field, value in (("attempt_count", len(history["attempts"])), ("accepted_count", len(accepted)),
                         ("known_return_count", known_count), ("initial_cost_units", phases["initial"]["cost_units"]),
                         ("acquisition_cost_units", phases["acquisition"]["cost_units"])):
        _require(integer(terminal[field]) and terminal[field] == value, "terminal counters contradict sealed history: " + field)
    initialized = len(accepted) >= len(initial)
    if terminal["status"] == "complete":
        _require(initialized and not ended and terminal["reason"] in ("policy_stop", "budget_exhausted"),
                 "incomplete initialization/response cannot be successful")
        if terminal["reason"] == "budget_exhausted":
            _require(not any(a.cost_units <= cap-phases["acquisition"]["cost_units"] for a in menu),
                     "budget-exhausted terminal still has affordable actions")
    else:
        _require(terminal["reason"] not in ("policy_stop", "budget_exhausted"), "failure needs a failure reason")
    charges = {"by_outcome": counters, "by_phase": phases,
        "actual_total_cost_units": sum(p["cost_units"] for p in phases.values()),
        "actual_initial_cost_units": phases["initial"]["cost_units"],
        "actual_acquisition_cost_units": phases["acquisition"]["cost_units"],
        "reserved_attempts": len(history["attempts"]), "known_returns": known_count,
        "accepted_responses": len(accepted), "unknown_returns": counters["unknown"]["attempts"],
        "known_return_cost_units": sum(counters[o]["cost_units"] for o in OUTCOMES if o != "unknown")}
    return world, task, learner, plan, terminal, accepted, initialized, charges


def report_cost_curve(inputs, *, expected_sha256):
    """Return a private JSON-only report; never writes or changes supplied bytes.

    See the guide for the deliberately new prospective sealing schema. This is
    not an adapter certifying legacy episode output. An authenticated operational
    failure is a reportable row. Malformed/unsealed evidence raises CurveInputError
    and must remain an unresolved capped row in the caller's frozen roster.
    """
    try:
        world, task, learner, plan, terminal, accepted, initialized, charges = _validate(inputs, expected_sha256)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise CurveInputError(str(exc)) from exc
    curve, evaluation_error = [], None
    initial_count = len(plan["initial_actions"])
    acquisition_cost = 0
    try:
        truths = [r[task.target] for r in private_structural_means(world, task.queries, expected_sha256=world.sha256)]

        def point(kind, receipt_count, receipt=None):
            predictions = [learner.predict_mean(q)[task.target] for q in task.queries]
            loss = math.fsum(((p-t)/task.scale)**2 for p, t in zip(predictions, truths))/len(truths)
            if not finite(loss):
                raise ValueError("nonfinite structural normalized MSE")
            curve.append({"kind": kind, "accepted_receipts": receipt_count,
                "request_id": receipt["request_id"] if receipt is not None else None,
                "action_id": receipt["action_id"] if receipt is not None else None,
                "acquisition_cost_units": acquisition_cost,
                "predictions": predictions, "normalized_mse": loss})

        if initialized:
            for receipt in accepted[:initial_count]:
                learner.update(receipt)
            point("shared_initial", initial_count)
            for count, receipt in enumerate(accepted[initial_count:], initial_count+1):
                learner.update(receipt)
                acquisition_cost += receipt["cost_units"]
                point("accepted_acquisition", count, receipt)
        # A failed partial initialization has no comparable shared-initial point.
    except (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
        evaluation_error = {"exception_type": type(exc).__name__, "message": str(exc)}
    crossing = next((p for p in curve if p["normalized_mse"] <= plan["threshold"]), None)
    failed = terminal["status"] == "failed" or evaluation_error is not None
    reason = ("operational_failure" if terminal["status"] == "failed" else
              "evaluation_failure" if evaluation_error else "nonattainment" if crossing is None else "first_crossing")
    final_error = curve[-1]["normalized_mse"] if curve and evaluation_error is None else None
    result = {"version": VERSION, "scope": world.scope, "world_id": world.world_id,
        "policy_id": plan["policy_id"], "input_sha256": dict(expected_sha256),
        "report_status": "evaluation_failed" if evaluation_error else "reported",
        "operational_status": terminal["status"], "termination_reason": terminal["reason"],
        "initialization_complete": initialized, "curve_complete": initialized and evaluation_error is None,
        "curve": curve, "final_error": final_error, "threshold": plan["threshold"],
        "scale": task.scale, "query_weighting": "equal_fixed_queries",
        "attained": crossing is not None, "final_attained": final_error is not None and final_error <= plan["threshold"],
        "attainment_scope": "observed_accepted_replay_prefix",
        "first_crossing_cost_units": crossing["acquisition_cost_units"] if crossing is not None else None,
        "acquisition_cap_units": plan["acquisition_cap_units"],
        "capped_cost_units": plan["acquisition_cap_units"] if failed or crossing is None else crossing["acquisition_cost_units"],
        "capped_cost_reason": reason, "charges": charges, "evaluation_error": evaluation_error,
        "private_evaluator_only": True, "private_scores_authorize_policy_stop": False,
        "new_environment_responses": 0, "policy_updates": 0,
        "source_qualification": "pending_caller", "split_qualification": "pending_caller",
        "execution_provenance_qualification": "pending_caller", "scientific_provenance_qualified": False,
        "complete_case_filtering_allowed": False}
    return json.loads(canonical(result))

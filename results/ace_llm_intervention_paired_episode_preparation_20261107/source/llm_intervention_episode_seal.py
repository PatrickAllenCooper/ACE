"""Pure journal-to-cost-bundle adapter; no replay, scoring, RNG or file I/O.

Hashes bind supplied bytes, not trusted execution. The caller retains original
evidence and qualifies its source/runtime/outer process. Rejections stay explicit
unresolved capped roster cells; this module never repairs an ambiguous history.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import re

from llm_intervention_contract import (Action, MechanismSummary, VERSION as CONTRACT_VERSION,
    canonical, digest, parse_completion, validate_design)
from llm_intervention_world import WorldDescriptor


VERSION = "ace-episode-seal-v1"
CURVE_VERSION = "ace-offline-cost-curve-v1"
BASE = frozenset(("world.json", "task.json", "learner.json", "episode-plan.json",
                  "executor.jsonl", "episode-disposition.json", "outer-outcome.json"))
MAX_BYTES, MAX_LINE_BYTES, MAX_REQUESTS = 8*1024*1024, 128*1024, 80
RECEIPT_FIELDS = ("request_id", "action_id", "kind", "targets", "values", "cost_units")


class SealError(ValueError):
    """Malformed, inconsistent or unsealed evidence; preserve caller cap row."""


def _need(value, message):
    if not value:
        raise SealError(message)


def _keys(value, fields):
    _need(type(value) is dict and set(value) == set(fields), "unexpected object fields")


def _int(value, maximum=1000000):
    return type(value) is int and 0 <= value <= maximum


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _text(value, limit=4096, empty=False):
    return type(value) is str and (empty or bool(value)) and len(value) <= limit


def _sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(obj):
    return (canonical(obj)+"\n").encode("utf-8")


def _json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    def constant(value):
        raise SealError("nonfinite JSON constant: "+value)
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=constant)
    canonical(value)  # catches overflowing float literals too
    return value


def _preflight(inputs, pins):
    _need(type(inputs) is dict and type(pins) is dict and set(inputs) == set(pins)
          and BASE <= set(inputs) and len(inputs) <= len(BASE)+MAX_REQUESTS, "exact bounded input/pin inventory required")
    _need(all(type(k) is str and type(v) is bytes for k, v in inputs.items())
          and sum(len(v) for v in inputs.values()) <= MAX_BYTES, "bounded raw bytes required")
    for name, raw in inputs.items():
        _need(name in BASE or re.fullmatch(r"return-0[0-7][0-9]\.json", name) is not None, "unexpected input member")
        _need(_sha(pins[name]) and _hash(raw) == pins[name], "independent byte pin mismatch: "+name)


def _action(obj):
    _keys(obj, ("action_id", "kind", "targets", "measure", "cost_units"))
    _need(_text(obj["action_id"], 256) and type(obj["kind"]) is str
          and type(obj["targets"]) is list and type(obj["measure"]) is list
          and len(obj["targets"]) <= 8 and 1 <= len(obj["measure"]) <= 8
          and all(type(x) is list and len(x) == 2 for x in obj["targets"])
          and all(_text(x, 256) for x in obj["measure"]) and _int(obj["cost_units"]) and obj["cost_units"] > 0,
          "invalid bounded action")
    return Action(obj["action_id"], obj["kind"], tuple(tuple(x) for x in obj["targets"]), tuple(obj["measure"]), obj["cost_units"])


def _source(objects, scope, world_id, policy_id, cap, shared):
    _need(scope in ("artificial", "development") and _int(world_id, 2**63-1)
          and _text(policy_id, 256) and _int(cap) and cap > 0, "invalid admitted identity/cap")
    world_data, task, learner, plan = (objects[n] for n in ("world.json", "task.json", "learner.json", "episode-plan.json"))
    _keys(world_data, ("version", "scope", "world_id", "graph", "mechanisms"))
    _need(type(world_data["graph"]) is list and 1 <= len(world_data["graph"]) <= 8
          and type(world_data["mechanisms"]) is list and len(world_data["mechanisms"]) == len(world_data["graph"]), "bounded descriptor required")
    world = WorldDescriptor.from_private_payload(world_data, expected_sha256=digest(world_data))
    _need(world.scope == scope and world.world_id == world_id, "world identity differs")
    _keys(task, ("target", "queries", "weights", "scale", "normalized_mse_tolerance"))
    _need(type(task["queries"]) is list and 1 <= len(task["queries"]) <= 64
          and type(task["weights"]) is list and len(task["weights"]) == len(task["queries"])
          and _text(task["target"], 256) and task["target"] in {n.name for n in world.graph}
          and all(_finite(w) and w == 1/len(task["queries"]) for w in task["weights"])
          and _finite(task["scale"]) and task["scale"] > 0
          and _finite(task["normalized_mse_tolerance"]) and task["normalized_mse_tolerance"] > 0, "invalid equal-query prediction task")
    queries = tuple(_action(q) for q in task["queries"])
    validate_design(world.graph, queries)
    _need(all(task["target"] in q.measure for q in queries), "every query must measure task target")
    _keys(learner, ("graph", "noise_variances", "prior_variance"))
    _need(canonical(learner["graph"]) == canonical(world_data["graph"])
          and type(learner["noise_variances"]) is dict and set(learner["noise_variances"]) == {n.name for n in world.graph}
          and all(_finite(x) and x > 0 for x in learner["noise_variances"].values())
          and _finite(learner["prior_variance"]) and learner["prior_variance"] > 0, "invalid learner specification")
    _keys(plan, ("graph", "menu", "budget", "initial_actions", "objective", "domain_notes", "policy_command", "policy_timeout_seconds"))
    _need(canonical(plan["graph"]) == canonical(world_data["graph"]) and type(plan["menu"]) is list
          and 1 <= len(plan["menu"]) <= 32, "plan graph/menu differs")
    menu = tuple(_action(a) for a in plan["menu"])
    validate_design(world.graph, menu)
    actions = {a.action_id: a for a in menu}
    _need(type(plan["initial_actions"]) is list and len(plan["initial_actions"]) <= MAX_REQUESTS
          and all(type(a) is str and a in actions for a in plan["initial_actions"])
          and _text(plan["objective"], 65536) and _text(plan["domain_notes"], 65536, empty=True)
          and type(plan["policy_command"]) is list and 1 <= len(plan["policy_command"]) <= 64
          and all(_text(s) for s in plan["policy_command"])
          and _finite(plan["policy_timeout_seconds"]) and plan["policy_timeout_seconds"] > 0, "invalid original episode plan")
    initial_cost = sum(actions[a].cost_units for a in plan["initial_actions"])
    _need(_int(plan["budget"]) and plan["budget"] == initial_cost+cap, "total budget must equal fixed initialization plus acquisition cap")
    _need(type(shared) is tuple and len(shared) == len(plan["initial_actions"]) and all(_sha(s) for s in shared), "full parent-supplied initial pin tuple required")
    return world.graph, menu


def _journal(raw, plan, menu):
    _need(bool(raw) and raw.endswith(b"\n"), "unsealed/empty journal")
    lines = raw.splitlines()
    _need(1 <= len(lines) <= 3*MAX_REQUESTS+3 and all(0 < len(line) <= MAX_LINE_BYTES for line in lines), "bounded complete journal lines required")
    events, previous, clock = [], None, -1
    for line in lines:
        record = _json(line)
        _need(type(record) is dict and {"sha256", "previous_sha256", "monotonic_ns", "event"} <= set(record), "invalid journal envelope")
        claimed = record.pop("sha256")
        _need(_sha(claimed) and record["previous_sha256"] == previous and digest(record) == claimed
              and _int(record["monotonic_ns"], 2**64-1) and record["monotonic_ns"] >= clock, "journal hash/order mismatch")
        clock, previous = record.pop("monotonic_ns"), claimed
        record.pop("previous_sha256")
        events.append(record)
    first = events[0]
    _keys(first, ("event", "version", "budget_units", "graph", "menu"))
    _need(first["event"] == "begin" and first["version"] == CONTRACT_VERSION
          and _int(first["budget_units"]) and first["budget_units"] == plan["budget"]
          and canonical(first["graph"]) == canonical(plan["graph"])
          and canonical(first["menu"]) == canonical(plan["menu"]), "begin does not bind original plan")
    attempts, receipts, proposals = [], [], []
    spent, decisions, i, terminal, stop_reason = 0, 0, 1, None, None
    while i < len(events):
        event = events[i]
        if event["event"] == "driver_failure":
            _keys(event, ("event", "exception_type", "message", "spent_units"))
            _need(_text(event["exception_type"]) and _text(event["message"], 65536, empty=True)
                  and _int(event["spent_units"]) and event["spent_units"] == spent, "invalid driver failure")
            terminal = event; i += 1; break
        _keys(event, ("event", "proposal_id", "completion", "provenance"))
        _need(event["event"] == "proposal" and _int(event["proposal_id"], MAX_REQUESTS)
              and event["proposal_id"] == len(proposals), "proposal order differs")
        provenance, initial = event["provenance"], len(attempts) < len(plan["initial_actions"])
        _need(type(provenance) is dict and "kind" in provenance, "proposal provenance required")
        if initial:
            _keys(provenance, ("kind",))
            _need(provenance["kind"] == "shared_initial_schedule"
                  and event["completion"] == canonical({"action_id": plan["initial_actions"][len(attempts)]}), "initial proposal differs from fixed schedule")
        elif provenance["kind"] == "budget_cap":
            _keys(provenance, ("kind",))
            _need(event["completion"] == '{"stop":true}' and not any(a.cost_units <= plan["budget"]-spent for a in menu), "premature budget stop")
        else:
            _keys(provenance, ("kind", "decision_index", "public_state_sha256"))
            _need(provenance["kind"] == "process_policy" and _int(provenance["decision_index"], MAX_REQUESTS)
                  and provenance["decision_index"] == decisions and _sha(provenance["public_state_sha256"])
                  and any(a.cost_units <= plan["budget"]-spent for a in menu), "invalid process decision binding")
            decisions += 1
        proposals.append(event)
        try:
            parsed = parse_completion(event["completion"], menu)
            parse_failed = False
        except (ValueError, TypeError, OverflowError, RecursionError):
            parsed, parse_failed = None, True
        i += 1
        _need(i < len(events), "proposal has no disposition")
        result = events[i]
        if result["event"] == "rejected":
            _keys(result, ("event", "proposal_id", "stage", "exception_type", "reason", "spent_units", "retry_allowed"))
            _need(_int(result["proposal_id"]) and result["proposal_id"] == event["proposal_id"]
                  and _int(result["spent_units"]) and result["spent_units"] == spent and result["retry_allowed"] is False
                  and _text(result["exception_type"]) and _text(result["reason"], 65536, empty=True), "invalid rejected proposal")
            too_costly = not parse_failed and not parsed.stop and next(a.cost_units for a in menu if a.action_id == parsed.action_id) > plan["budget"]-spent
            _need((result["stage"] == "completion_parse" and parse_failed)
                  or (result["stage"] == "budget" and too_costly), "rejection stage contradicts run_episode completion")
            terminal = result; i += 1; break
        _need(not parse_failed, "invalid completion cannot reserve/stop")
        if parsed.stop:
            _keys(result, ("event", "proposal_id", "completion", "spent_units"))
            _need(not initial and result["event"] == "stop" and _int(result["proposal_id"])
                  and result["proposal_id"] == event["proposal_id"] and result["completion"] == event["completion"]
                  and _int(result["spent_units"]) and result["spent_units"] == spent, "stop differs from proposal/accounting")
            stop_reason = "budget_exhausted" if provenance["kind"] == "budget_cap" else "policy_stop"
            terminal = result; i += 1; break
        _keys(result, ("event", "proposal_id", "request_id", "action", "completion", "cost_units"))
        action = next(a for a in menu if a.action_id == parsed.action_id)
        _need(result["event"] == "reserve" and _int(result["proposal_id"]) and result["proposal_id"] == event["proposal_id"]
              and _int(result["request_id"], MAX_REQUESTS-1) and result["request_id"] == len(attempts)
              and canonical(result["action"]) == canonical(asdict(action)) and result["completion"] == event["completion"]
              and _int(result["cost_units"]) and result["cost_units"] == action.cost_units
              and spent+action.cost_units <= plan["budget"], "reservation differs from proposal/action/budget")
        spent += action.cost_units
        attempts.append({"action": action, "request_id": result["request_id"], "phase": "initial" if initial else "acquisition"})
        i += 1
        _need(i < len(events), "reservation has no durable response/failure disposition")
        returned = events[i]
        if returned["event"] == "failure":
            _keys(returned, ("event", "request_id", "response_returned_to_driver", "exception_type", "message", "spent_units"))
            _need(_int(returned["request_id"]) and returned["request_id"] == result["request_id"]
                  and (returned["response_returned_to_driver"] is True or returned["response_returned_to_driver"] is None)
                  and _int(returned["spent_units"]) and returned["spent_units"] == spent
                  and _text(returned["exception_type"]) and _text(returned["message"], 65536, empty=True), "invalid response failure")
            terminal = returned; i += 1; break
        _keys(returned, ("event", *RECEIPT_FIELDS))
        _need(returned["event"] == "response" and _int(returned["request_id"])
              and returned["request_id"] == result["request_id"] and returned["action_id"] == action.action_id
              and returned["kind"] == action.kind and type(returned["targets"]) is dict
              and all(_finite(v) for v in returned["targets"].values()) and returned["targets"] == dict(action.targets)
              and _int(returned["cost_units"]) and returned["cost_units"] == action.cost_units
              and _valid_values(returned["values"], action), "accepted response violates action/clamp/charge")
        receipts.append({k: returned[k] for k in RECEIPT_FIELDS})
        i += 1
    _need(terminal is not None and i == len(events), "missing terminal or continuation after terminal/failure")
    return {"attempts": attempts, "receipts": receipts, "terminal": terminal, "spent": spent,
            "decisions": decisions, "stop_reason": stop_reason, "chain_tip": previous, "event_count": len(events)}


def _valid_values(value, action):
    return (type(value) is dict and set(value) == set(action.measure) and all(_finite(v) for v in value.values())
            and all(value[k] == v for k, v in action.targets if k in value))


def _returns(objects, chain, shared):
    history, evidence, used = [], {}, set()
    for a in chain["attempts"]:
        request, action = a["request_id"], a["action"]
        filename = f"return-{request:03d}.json"
        raw = None
        if filename in objects:
            used.add(filename)
            wrapper = objects[filename]
            _keys(wrapper, ("request_id", "action", "returned_to_wrapper", "response"))
            _need(_int(wrapper["request_id"]) and wrapper["request_id"] == request
                  and canonical(wrapper["action"]) == canonical(asdict(action)) and wrapper["returned_to_wrapper"] is True,
                  "return wrapper differs from reserved request/action")
            capture = wrapper["response"]
            _need(type(capture) is dict and "raw_json_available" in capture and type(capture["raw_json_available"]) is bool, "invalid raw return capture")
            representation = "raw_response_repr" if "raw_response_repr" in capture else "repr_error"
            _keys(capture, ("python_type", representation, "raw_json_available", *(("raw_json",) if capture["raw_json_available"] else ())))
            _need(_text(capture["python_type"], 1024) and _text(capture[representation], MAX_LINE_BYTES, empty=True), "invalid typed return representation")
            _need(capture["raw_json_available"], "non-JSON callback return cannot be sealed; retain caller unresolved cell")
            raw = capture["raw_json"]
        accepted = request < len(chain["receipts"])
        if accepted:
            _need(filename in used and _valid_values(raw, action), "accepted response lacks consistent captured return")
            receipt = chain["receipts"][request]
            _need(receipt["values"] == {k: float(raw[k]) for k in action.measure}, "captured values contradict executor acceptance")
            evidence_name, outcome = f"receipt-{request:03d}.json", "accepted"
            evidence[evidence_name] = _bytes(receipt)
            if a["phase"] == "initial":
                _need(_hash(evidence[evidence_name]) == shared[request], "accepted initial receipt differs from supplied common pin")
        elif filename in used:
            evidence_name = f"return-{request:03d}.bin"
            outcome = "returned_unaccepted" if _valid_values(raw, action) else "returned_invalid"
            evidence[evidence_name] = _bytes(raw)
        else:
            _need(chain["terminal"]["event"] == "failure" and chain["terminal"]["response_returned_to_driver"] is None,
                  "known driver return has no captured wrapper evidence")
            evidence_name, outcome = None, "unknown"
        history.append({"request_id": request, "phase": a["phase"], "action_id": action.action_id,
                        "cost_units": action.cost_units, "outcome": outcome, "evidence_file": evidence_name})
    _need(used == set(objects)-BASE, "unreserved/extra callback capture")
    return history, evidence


def _state(state, plan, graph, learner, chain, applied_options):
    if type(state) is dict and "state_unavailable" in state:
        _keys(state, ("state_unavailable", "accounting_source"))
        _need(_text(state["state_unavailable"], 65536) and state["accounting_source"] == "retained journal.jsonl", "invalid unavailable state evidence")
        return
    _keys(state, ("version", "objective", "domain_notes", "graph", "menu", "history", "scm", "spent_units", "remaining_units", "requests", "returned_responses", "terminal"))
    _need(state["version"] == CONTRACT_VERSION and state["objective"] == plan["objective"] and state["domain_notes"] == plan["domain_notes"]
          and canonical(state["graph"]) == canonical(plan["graph"]) and canonical(state["menu"]) == canonical(plan["menu"])
          and canonical(state["history"]) == canonical(chain["receipts"]) and state["terminal"] is True, "terminal public state differs from journal/plan")
    for k, v in (("spent_units", chain["spent"]), ("remaining_units", plan["budget"]-chain["spent"]),
                 ("requests", len(chain["attempts"])), ("returned_responses", len(chain["receipts"]))):
        _need(_int(state[k]) and state[k] == v, "terminal public counter differs: "+k)
    _need(type(state["scm"]) is list and len(state["scm"]) == len(graph), "one public summary per node required")
    observed = []
    for n, summary in zip(graph, state["scm"]):
        _keys(summary, ("node", "terms", "parameter_mean", "parameter_covariance", "noise_variance", "eligible_rows"))
        _need(summary["node"] == n.name and summary["terms"] == ["intercept", *n.parents]
              and type(summary["parameter_mean"]) is list and type(summary["parameter_covariance"]) is list
              and all(type(r) is list for r in summary["parameter_covariance"])
              and summary["noise_variance"] == learner["noise_variances"][n.name]
              and _int(summary["eligible_rows"], MAX_REQUESTS), "terminal summary identity/configuration differs")
        MechanismSummary(summary["node"], tuple(summary["terms"]), tuple(summary["parameter_mean"]),
            tuple(tuple(r) for r in summary["parameter_covariance"]), summary["noise_variance"], summary["eligible_rows"])
        observed.append(summary["eligible_rows"])
    expected = []
    for count in applied_options:
        expected.append([sum(n.name not in r["targets"] and n.name in r["values"] and set(n.parents) <= set(r["values"]) | set(r["targets"])
                             for r in chain["receipts"][:count]) for n in graph])
    _need(observed in expected, "terminal eligible-row counters cannot follow accepted prefix")


def _disposition(objects, pins, plan, graph, chain, scope, world_id, policy_id):
    outer, disposition = objects["outer-outcome.json"], objects["episode-disposition.json"]
    _keys(outer, ("version", "scope", "world_id", "policy_id", "status", "reason", "episode_returned_cleanly", "adapters_closed_cleanly",
                  "episode_disposition_sha256", "outer_provenance_receipt_sha256"))
    _need(outer["version"] == "ace-paired-episode-v1" and outer["scope"] == scope
          and _int(outer["world_id"], 2**63-1) and outer["world_id"] == world_id and outer["policy_id"] == policy_id
          and outer["episode_disposition_sha256"] == pins["episode-disposition.json"] and _sha(outer["outer_provenance_receipt_sha256"])
          and type(outer["episode_returned_cleanly"]) is bool and type(outer["adapters_closed_cleanly"]) is bool
          and outer["status"] in ("complete", "failed") and _text(outer["reason"], 1024), "invalid outer disposition binding")
    clean, closed = outer["episode_returned_cleanly"], outer["adapters_closed_cleanly"]
    if outer["status"] == "complete":
        _need(clean and closed and outer["reason"] == chain["stop_reason"], "outer success contradicts source terminal")
    else:
        _need(not (clean and closed) and (
            (not clean and outer["reason"].startswith("episode_failure:") and len(outer["reason"]) > len("episode_failure:")) or
            (not closed and outer["reason"].startswith("adapter_close_failure:") and len(outer["reason"]) > len("adapter_close_failure:"))), "outer failure flags/reason disagree")
    if clean:
        _keys(disposition, ("status", "initialized_responses", "policy_decisions", "final_state", "scientific_evaluation_performed"))
        _need(disposition["status"] == "complete" and disposition["scientific_evaluation_performed"] is False
              and chain["terminal"]["event"] == "stop" and len(chain["receipts"]) == len(chain["attempts"]), "clean episode lacks successful journal stop")
        count_key, state = "policy_decisions", disposition["final_state"]
        _need(type(state) is dict and "state_unavailable" not in state, "complete episode lacks final public state")
    else:
        _keys(disposition, ("status", "exception_type", "message", "initialized_responses", "completed_policy_calls", "partial_state", "retry_allowed"))
        _need(disposition["status"] == "failed" and disposition["retry_allowed"] is False
              and _text(disposition["exception_type"]) and _text(disposition["message"], 65536, empty=True), "invalid original failure")
        terminal = chain["terminal"]
        if terminal["event"] != "stop":
            _need(disposition["exception_type"] == terminal["exception_type"]
                  and disposition["message"] == terminal.get("message", terminal.get("reason")), "episode failure contradicts journal exception")
        count_key, state = "completed_policy_calls", disposition["partial_state"]
    _need(_int(disposition[count_key], MAX_REQUESTS+1) and disposition[count_key] == chain["decisions"]
          and _int(disposition["initialized_responses"], MAX_REQUESTS), "episode decision/initial counters invalid")
    initialized, accepted = disposition["initialized_responses"], len(chain["receipts"])
    initial_count = len(plan["initial_actions"])
    expected_initial = min(initial_count, accepted)
    applied = [accepted]
    if initialized != expected_initial:
        _need(not clean and chain["terminal"]["event"] == "driver_failure" and 0 < accepted <= initial_count
              and initialized == accepted-1, "initialized count contradicts accepted prefix")
        applied = [accepted-1]
    elif not clean and chain["terminal"]["event"] == "driver_failure" and accepted > initial_count:
        applied.append(accepted-1)  # last receipt accepted but atomic learner update may have failed
    if clean:
        _need(initialized == initial_count, "complete episode has partial initialization")
    _state(state, plan, graph, objects["learner.json"], chain, applied)
    return outer


def seal_episode(*, inputs: dict[str, bytes], expected_sha256: dict[str, str],
                 shared_initial_receipt_sha256: tuple[str, ...], scope: str, world_id: int,
                 policy_id: str, acquisition_cap_units: int):
    """Authenticate saved chain consistency and emit a new canonical-LF bundle.

    Every supplied byte pin is checked before any JSON decoding. Accepted receipt
    fields come only from executor response events. Known returns require wrapper
    captures; adapter outcome/prepared records are never evidence of return.
    Source/runtime/independent pin custody and outer acceptance remain caller-owned.
    """
    try:
        _preflight(inputs, expected_sha256)
        objects = {k: _json(v) for k, v in inputs.items() if k != "executor.jsonl"}
        graph, menu = _source(objects, scope, world_id, policy_id, acquisition_cap_units, shared_initial_receipt_sha256)
        plan = objects["episode-plan.json"]
        chain = _journal(inputs["executor.jsonl"], plan, menu)
        history, evidence = _returns(objects, chain, shared_initial_receipt_sha256)
        outer = _disposition(objects, expected_sha256, plan, graph, chain, scope, world_id, policy_id)
        bundle = {k: _bytes(objects[k]) for k in ("world.json", "task.json", "learner.json")}
        bundle["shared-initial.json"] = _bytes({"receipt_sha256": list(shared_initial_receipt_sha256)})
        new_plan = {"version": CURVE_VERSION, "scope": scope, "world_id": world_id, "policy_id": policy_id,
            **{field: _hash(bundle[name]) for field, name in (("world_sha256", "world.json"), ("task_sha256", "task.json"),
                ("learner_sha256", "learner.json"), ("shared_initial_sha256", "shared-initial.json"))},
            "menu": plan["menu"], "initial_actions": plan["initial_actions"], "acquisition_cap_units": acquisition_cap_units,
            "threshold": objects["task.json"]["normalized_mse_tolerance"]}
        bundle["plan.json"] = _bytes(new_plan)
        bundle["history.json"] = _bytes({"version": CURVE_VERSION, "plan_sha256": _hash(bundle["plan.json"]), "attempts": history})
        bundle.update(evidence)
        pins = {k: _hash(v) for k, v in bundle.items()}
        initial_cost = sum(e["cost_units"] for e in history if e["phase"] == "initial")
        acquisition_cost = sum(e["cost_units"] for e in history if e["phase"] == "acquisition")
        _need(acquisition_cost <= acquisition_cap_units, "acquisition charges exceed common cap")
        terminal = {"version": CURVE_VERSION, "sealed": True, "input_sha256": dict(pins), "status": outer["status"],
            "reason": outer["reason"], "attempt_count": len(history), "accepted_count": len(chain["receipts"]),
            "known_return_count": sum(e["outcome"] != "unknown" for e in history),
            "initial_cost_units": initial_cost, "acquisition_cost_units": acquisition_cost}
        bundle["terminal.json"] = _bytes(terminal)
        pins["terminal.json"] = _hash(bundle["terminal.json"])
        _need(sum(map(len, bundle.values())) <= MAX_BYTES, "sealed bundle exceeds reporter byte bound")
        provenance = {"version": VERSION, "scope": scope, "world_id": world_id, "policy_id": policy_id,
            "input_sha256": dict(expected_sha256), "input_bytes": {k: len(v) for k, v in inputs.items()},
            "bundle_sha256": dict(pins), "source_episode_status": objects["episode-disposition.json"]["status"],
            "sealed_status": outer["status"], "reason": outer["reason"], "journal_chain_tip": chain["chain_tip"],
            "journal_events": chain["event_count"], "outer_provenance_receipt_sha256": outer["outer_provenance_receipt_sha256"],
            "encoding": "UTF-8 canonical JSON plus LF for every bundle member; original input hashes retained separately",
            "return_evidence": "wrapper capture and executor response only; no adapter-outcome inference",
            "qualification": "byte/chain/schema/accounting consistency only; source/runtime/outer receipt authentication caller-owned",
            "posterior_replay_performed": False, "private_scoring_performed": False,
            "scientific_provenance_qualified": False, "scientific_acceptance": False}
        return {"bundle": bundle, "bundle_sha256": pins, "provenance": provenance}
    except SealError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, StopIteration, OverflowError, RecursionError) as exc:
        raise SealError(f"{type(exc).__name__}: {exc}") from exc

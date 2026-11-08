"""Charged one-response branches and shared scoring for replicated feedback.

Callbacks supply the responses. Their noise, independent streams, world split,
and honesty remain external obligations; this driver does not generate worlds
or qualify provenance. Failed or ambiguous attempts cannot be resumed here.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import traceback

from llm_intervention_contract import ChargedExecutor, canonical, digest, finite, integer, validate_design
from llm_intervention_replicated_feedback import aggregate_replicated_feedback
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_task import PredictionTask


VERSION = "ace-replicated-charged-branches-v1"


def _save(path, value):
    payload = canonical(value) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _snapshot(learner):
    # summaries alone omit request identity and constructor priors. Include both
    # so a callback cannot silently change which rows a subsequent fork accepts.
    return json.loads(canonical({"graph": [asdict(n) for n in learner.graph],
        "summaries": [asdict(s) for s in learner.summaries()],
        "request_ids": sorted(learner._requests),
        "initial_heads": {name: {"mean": h.mean.tolist(), "covariance": h.covariance.tolist(),
            "noise_variance": h.noise_variance, "eligible_rows": h.eligible_rows}
            for name, h in learner._initial_heads.items()}}))


def _check_clamps(receipt):
    if any(name in receipt["values"] and receipt["values"][name] != value
           for name, value in receipt["targets"].items()):
        raise ValueError("response contradicts hard-intervention clamp")


def _raw_return(response):
    # Invalid/nonfinite returns must survive rejection by ChargedExecutor.
    # JSON is exact for its valid numeric response contract. For invalid Python
    # objects, retain a typed representation, without unsafe pickle loading.
    result = {"python_type": type(response).__module__ + "." + type(response).__qualname__}
    try:
        result["raw_response_repr"] = repr(response)
    except BaseException as exc:
        result["repr_error"] = type(exc).__name__ + ": " + str(exc)
    try:
        result["raw_json"] = json.loads(canonical(response))
        result["raw_json_available"] = True
    except (TypeError, ValueError, OverflowError, RecursionError):
        result["raw_json_available"] = False
    return result


def _accounting(output, journals, returned):
    reservations, accepted, known = {}, set(), set()
    known.update((x["journal"], x["request_id"]) for x in returned)
    errors, files = [], []
    for entry in journals:
        filename = entry["journal"]
        path = output/filename
        if not path.exists():
            continue
        try:
            raw = path.read_bytes()
            files.append({"journal": filename, "sha256": hashlib.sha256(raw).hexdigest()})
            previous = None
            for line_number, line in enumerate(raw.splitlines(), 1):
                record = json.loads(line)
                claimed = record.pop("sha256")
                if record["previous_sha256"] != previous or digest(record) != claimed:
                    raise ValueError("journal hash chain mismatch at line " + str(line_number))
                previous = claimed
                if record["event"] == "reserve":
                    key = (filename, record["request_id"])
                    if key in reservations or not integer(record["cost_units"], 1):
                        raise ValueError("invalid or duplicated reservation")
                    reservations[key] = {"journal": filename, "kind": entry["kind"],
                        "replicate_id": entry["replicate_id"], "request_id": record["request_id"],
                        "action_id": record["action"]["action_id"], "cost_units": record["cost_units"]}
                elif record["event"] == "response":
                    key = (filename, record["request_id"])
                    if key in accepted:
                        raise ValueError("duplicate response")
                    accepted.add(key)
                    known.add(key)
                elif record["event"] == "failure" and record.get("response_returned_to_driver") is True:
                    known.add((filename, record["request_id"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"journal": filename, "error": type(exc).__name__ + ": " + str(exc)})
    if not known <= set(reservations):
        errors.append({"error": "known return without readable reservation"})
    totals = {}
    for kind in ("branch", "scoring"):
        keys = {k for k, v in reservations.items() if v["kind"] == kind}
        totals[kind] = {"reserved_cost_units": sum(reservations[k]["cost_units"] for k in keys),
            "reserved_response_attempts": len(keys), "known_returns": len(keys & known),
            "executor_accepted_responses": len(keys & accepted),
            "returned_without_accepted_response": len((keys & known)-accepted),
            "unknown_return_attempts": len(keys-known),
            "unknown_return_reserved_cost": sum(reservations[k]["cost_units"] for k in keys-known)}
    return {**totals, "total_reserved_cost_units": sum(v["reserved_cost_units"] for v in totals.values()),
        "reserved_response_attempts": len(reservations), "known_returns": len(set(reservations) & known),
        "executor_accepted_responses": len(accepted), "unknown_return_attempts": len(set(reservations)-known),
        "reservations": list(reservations.values()), "journal_files": files,
        "accounting_complete": not errors, "accounting_errors": errors,
        "cost_rule": "every durable reservation stays charged, including failed or unknown returns"}


def build_replicated_training_preferences(*, scope, graph, menu, candidates, learner,
        next_learner_request_id, task, replicate_ids, branch_respond, score_respond,
        tie_multiplier, minimum_gain_gap, output):
    """Execute exactly one branch response per candidate per replicate.

    branch_respond(replicate_id, candidate_id, action, journal_request_id) and
    score_respond(replicate_id, action, journal_request_id) own all external
    response generation. Each task query is scored once per replicate. Queries
    must measure task.target. Scoring receipts never update any learner.

    Every branch forks the same frozen pre-action learner; journal-local request
    zero maps to next_learner_request_id in that independent fork. The full plan
    and planned mappings precede callbacks. Return artifacts also preserve
    rejected values. Callback exceptions leave response outcome unknown unless
    a returned value was observed. No retry, fallback, or complete-case filtering
    occurs. Only policy_train and artificial declarations are admitted; neither
    authenticates world/stream separation. Output must not already exist.
    """
    if type(scope) is not str or scope not in ("policy_train", "artificial"):
        raise ValueError("replicated branches are policy_train/artificial only")
    validate_design(graph, menu)
    if not isinstance(learner, LinearGaussianSCM) or learner.graph != graph or not isinstance(task, PredictionTask):
        raise ValueError("matching LinearGaussianSCM and PredictionTask required")
    task.validate(graph)
    by_id = {a.action_id: a for a in menu}
    if (type(candidates) is not tuple or len(candidates) < 2
            or any(type(x) is not str or not x for x in candidates)
            or len(set(candidates)) != len(candidates) or not set(candidates) <= set(by_id)
            or type(replicate_ids) is not tuple or len(replicate_ids) < 2
            or any(type(x) is not str or not x for x in replicate_ids)
            or len(set(replicate_ids)) != len(replicate_ids)
            or not integer(next_learner_request_id) or next_learner_request_id in learner._requests
            or any(task.target not in q.measure for q in task.queries)
            or not callable(branch_respond) or not callable(score_respond)
            or not finite(tie_multiplier) or tie_multiplier < 0
            or not finite(minimum_gain_gap) or minimum_gain_gap < 0):
        raise ValueError("invalid fixed replicate/candidate/query-target/request design")
    base_state = _snapshot(learner)
    preaction = learner.fork()
    base_pin = digest(base_state)

    def unchanged():
        if digest(_snapshot(learner)) != base_pin or digest(_snapshot(preaction)) != base_pin:
            raise RuntimeError("base learner changed during isolated branches")

    unchanged()
    fixed_base_predictions = [preaction.predict_mean(q)[task.target] for q in task.queries]
    task_payload, costs = task.payload(), {c: by_id[c].cost_units for c in candidates}
    journals = []
    for ri, rid in enumerate(replicate_ids):
        for ci, candidate in enumerate(candidates):
            journals.append({"kind": "branch", "replicate_id": rid, "candidate_id": candidate,
                "journal": f"replicate-{ri:03d}.branch-{ci:03d}.jsonl",
                "budget_units": costs[candidate], "source_request_id": 0,
                "learner_request_id": next_learner_request_id})
        journals.append({"kind": "scoring", "replicate_id": rid,
            "journal": f"replicate-{ri:03d}.scoring.jsonl",
            "budget_units": sum(q.cost_units for q in task.queries),
            "query_action_ids": [q.action_id for q in task.queries]})
    plan = {"version": VERSION, "scope": scope, "graph": [asdict(n) for n in graph],
        "menu": [asdict(a) for a in menu], "candidates": list(candidates), "candidate_costs": costs,
        "replicate_ids": list(replicate_ids), "task": task_payload, "task_sha256": digest(task_payload),
        "target": task.target, "scale": task.scale, "query_weights": list(task.weights),
        "query_action_ids": [q.action_id for q in task.queries], "journals": journals,
        "next_learner_request_id": next_learner_request_id, "base_state": base_state,
        "base_state_sha256": base_pin, "fixed_base_predictions": fixed_base_predictions,
        "tie_multiplier": tie_multiplier, "minimum_gain_gap": minimum_gain_gap,
        "expected_branch_responses": len(candidates)*len(replicate_ids),
        "expected_scoring_responses": len(task.queries)*len(replicate_ids),
        "expected_branch_cost": sum(costs.values())*len(replicate_ids),
        "expected_scoring_cost": sum(q.cost_units for q in task.queries)*len(replicate_ids),
        "provenance_qualified": False, "retry_allowed": False,
        "external_obligations": "honest task and responses, branch noise, independent joint branch/scoring "
            "replicate blocks, training-world and stream separation; no implicit namespace qualification"}
    output = Path(output)
    output.mkdir(exist_ok=False)
    returned, completed_branches, completed_scoring = [], [], []
    branch_predictions = {c: [] for c in candidates}
    truths = []

    def responder(entry, callback):
        def respond(action, request):
            unchanged()
            value = callback(action, request)
            record = {"journal": entry["journal"], "request_id": request,
                "replicate_id": entry["replicate_id"], "action_id": action.action_id,
                "returned_to_wrapper": True, **_raw_return(value)}
            returned.append(record)
            _save(output/(Path(entry["journal"]).stem + f".return-{request:03d}.json"), record)
            unchanged()
            return value
        return respond

    try:
        _save(output/"plan.json", plan)
        for ri, rid in enumerate(replicate_ids):
            for candidate in candidates:
                unchanged()
                branch = preaction.fork()
                if digest(_snapshot(branch)) != base_pin:
                    raise RuntimeError("branch did not start from the fixed pre-action learner")
                entry = next(e for e in journals if e["kind"] == "branch"
                             and e["replicate_id"] == rid and e["candidate_id"] == candidate)
                prefix = Path(entry["journal"]).stem
                mapping = {**entry, "base_state_sha256": base_pin, "status": "planned_before_response"}
                _save(output/(prefix + ".mapping.json"), mapping)
                callback = lambda a, r, rid=rid, c=candidate: branch_respond(rid, c, a, r)
                with ChargedExecutor(graph, menu, costs[candidate], output/entry["journal"],
                                     responder(entry, callback)) as executor:
                    receipt = executor.execute_completion(canonical({"action_id": candidate}),
                        provenance={"scope": scope, "replicate_id": rid, "kind": "candidate_branch"})
                    _check_clamps(receipt)
                    if receipt["request_id"] != entry["source_request_id"]:
                        raise RuntimeError("branch receipt does not match planned request mapping")
                    branch.update(dict(receipt, request_id=next_learner_request_id))
                    predictions = [branch.predict_mean(q)[task.target] for q in task.queries]
                    outcome = {"replicate_id": rid, "candidate_id": candidate, "receipt": receipt,
                        "mapping": mapping, "post_state": _snapshot(branch), "query_predictions": predictions}
                    _save(output/(prefix + ".branch.json"), outcome)
                branch_predictions[candidate].append(predictions)
                completed_branches.append(prefix + ".branch.json")
                unchanged()
            entry = next(e for e in journals if e["kind"] == "scoring" and e["replicate_id"] == rid)
            callback = lambda a, r, rid=rid: score_respond(rid, a, r)
            receipts = []
            with ChargedExecutor(graph, task.queries, entry["budget_units"], output/entry["journal"],
                                 responder(entry, callback)) as executor:
                for query in task.queries:
                    receipt = executor.execute_completion(canonical({"action_id": query.action_id}),
                        provenance={"scope": scope, "replicate_id": rid, "kind": "shared_scoring"})
                    _check_clamps(receipt)
                    receipts.append(receipt)
            values = [r["values"][task.target] for r in receipts]
            _save(output/(Path(entry["journal"]).stem + ".scoring.json"),
                {"replicate_id": rid, "receipts": receipts, "target": task.target,
                 "truths": values, "learner_updates": 0})
            truths.append(values)
            completed_scoring.append(rid)
            unchanged()
        feedback = aggregate_replicated_feedback(scope=scope, task=task_payload,
            task_sha256=plan["task_sha256"], target=task.target, scale=task.scale,
            query_action_ids=plan["query_action_ids"], query_weights=task.weights,
            candidate_ids=candidates, candidate_costs=costs, replicate_ids=replicate_ids,
            base_predictions=[list(fixed_base_predictions) for _ in replicate_ids],
            branch_predictions=branch_predictions, scoring_truths=truths,
            tie_multiplier=tie_multiplier, minimum_gain_gap=minimum_gain_gap)
        unchanged()
        accounting = _accounting(output, journals, returned)
        for kind in ("branch", "scoring"):
            if (not accounting["accounting_complete"]
                    or accounting[kind]["reserved_cost_units"] != plan["expected_"+kind+"_cost"]
                    or accounting[kind]["reserved_response_attempts"] != plan["expected_"+kind+"_responses"]
                    or accounting[kind]["executor_accepted_responses"] != plan["expected_"+kind+"_responses"]):
                raise RuntimeError("incomplete charged response accounting")
        result = {"status": "complete", "version": VERSION, "scope": scope,
            "plan_sha256": digest(plan), "feedback": feedback, "accounting": accounting,
            "charged_branch_cost": accounting["branch"]["reserved_cost_units"],
            "charged_scoring_cost": accounting["scoring"]["reserved_cost_units"],
            "environment_responses": accounting["known_returns"],
            "base_learner_unchanged": True, "base_state_sha256": base_pin,
            "scoring_learner_updates": 0, "provenance_qualified": False,
            "stop_inference_performed": False, "retry_allowed": False}
        _save(output/"terminal.json", result)
        return result
    except BaseException as exc:
        try:
            base_unchanged = digest(_snapshot(learner)) == base_pin and digest(_snapshot(preaction)) == base_pin
        except BaseException:
            base_unchanged = None
        _save(output/"failure.json", {"status": "failed", "exception_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(), "plan_sha256": digest(plan),
            "accounting": _accounting(output, journals, returned), "callback_returns": returned,
            "completed_branches": completed_branches, "completed_scoring_replicates": completed_scoring,
            "partial_branch_predictions": branch_predictions, "completed_scoring_truths": truths,
            "base_learner_unchanged": base_unchanged, "retry_allowed": False,
            "complete_case_filtering_allowed": False, "provenance_qualified": False,
            "cost_instruction": "retain all journals; unknown returns are not zero responses or refunded costs"})
        raise

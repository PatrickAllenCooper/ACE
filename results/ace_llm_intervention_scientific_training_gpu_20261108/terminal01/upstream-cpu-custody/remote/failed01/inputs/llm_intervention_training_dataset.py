"""Bounded single-world CPU assembly, with no world generation or training.

Only the detached public envelope reaches numerical behavior/teachers. The
trusted driver owns descriptor, response streams, labels and charge journals.
Pins bind supplied inputs; they do not qualify scientific world separation.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np

from llm_intervention_contract import ChargedExecutor, canonical, digest, finite, integer, validate_design
from llm_intervention_numerical_policy import choose
from llm_intervention_replicated_branches import (
    _raw_return, _save, _snapshot, build_replicated_training_preferences,
)
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_supervision import VERSION as RECORD_VERSION, matched_views
from llm_intervention_target_risk import choose_target_risk
from llm_intervention_task import MAX_ACTIONS, MAX_HISTORY, PredictionTask, make_envelope
from llm_intervention_world import ResponseAdapter, WorldDescriptor, keyed_seed, training_callbacks


VERSION = "ace-single-world-training-dataset-v1"
METHODS = ("uniform_random", "menu_coverage", "local_information", "target_risk")
MAX_SNAPSHOTS, MAX_REPLICATES, MAX_FEEDBACK_RESPONSES = 16, 16, 4096
SOURCE_FILES = tuple("llm_intervention_" + name + ".py" for name in (
    "training_dataset", "world", "contract", "scm", "task", "episode",
    "numerical_policy", "target_risk", "replicated_branches", "replicated_feedback", "supervision"))


@dataclass(frozen=True)
class Snapshot:
    """Milestone is paid acquisition cost AFTER the fixed initial schedule.

    Empty candidates explicitly request a budget-terminal stop example. Other
    snapshots require at least two fixed, affordable candidates; none are pruned.
    """
    acquisition_cost_units: int
    teacher_method: str
    candidates: tuple[str, ...]


def _pin(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _name(value):
    return type(value) is str and 0 < len(value) <= 256


def _select(envelope, method, seed, draws):
    if method == "target_risk":
        # Reuse the declared object via the public schema, never a private law.
        from llm_intervention_task import unpack_envelope
        _, task, _, _, _ = unpack_envelope(envelope)
        return choose_target_risk(envelope["state"], seed=seed, draws=draws,
                                  **task.risk_arguments())
    return choose(envelope["state"], method=method, seed=seed, draws=draws)


def _accounting(output, journal_plan):
    """Reconcile durable executor reservations, including unfinished attempts.

    Adapter 'outcome' means prepared, not returned. Raw-return evidence and
    executor acceptance are counted separately; an unknown return stays charged.
    """
    reservations, accepted, known, errors, files = {}, set(), set(), [], []
    for entry in journal_plan:
        path = output / entry["path"]
        if not path.exists():
            continue
        try:
            raw, previous = path.read_bytes(), None
            files.append({"path": entry["path"], "sha256": hashlib.sha256(raw).hexdigest(),
                          "bytes": len(raw)})
            if not raw.endswith(b"\n"):
                raise ValueError("incomplete executor journal")
            for line in raw.splitlines():
                event = json.loads(line)
                claimed = event.pop("sha256")
                if event["previous_sha256"] != previous or digest(event) != claimed:
                    raise ValueError("executor hash chain differs")
                previous = claimed
                key = (entry["path"], event.get("request_id"))
                if event["event"] == "reserve":
                    if key in reservations or not integer(event["cost_units"], 1):
                        raise ValueError("duplicate or invalid reservation")
                    reservations[key] = {"journal": entry["path"], "kind": entry["kind"],
                        "request_id": event["request_id"], "action_id": event["action"]["action_id"],
                        "cost_units": event["cost_units"]}
                    returned = path.with_name(path.stem + f".return-{event['request_id']:03d}.json")
                    if returned.exists():
                        evidence = json.loads(returned.read_text())
                        if (evidence["request_id"] != event["request_id"]
                                or evidence["journal"] != path.name
                                or evidence["returned_to_wrapper"] is not True):
                            raise ValueError("raw return does not bind to its reservation")
                        known.add(key)
                elif event["event"] == "response":
                    if key not in reservations or key in accepted:
                        raise ValueError("response without unique reservation")
                    accepted.add(key)
                    known.add(key)
                elif event["event"] == "failure" and event.get("response_returned_to_driver") is True:
                    known.add(key)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": entry["path"], "error": type(exc).__name__ + ": " + str(exc)})
    if not known <= set(reservations):
        errors.append({"error": "known return without readable reservation"})
    totals = {}
    for kind in ("trajectory", "branch", "scoring"):
        keys = {k for k, v in reservations.items() if v["kind"] == kind}
        totals[kind] = {"reserved_cost_units": sum(reservations[k]["cost_units"] for k in keys),
            "reserved_response_attempts": len(keys), "known_returns": len(keys & known),
            "executor_accepted_responses": len(keys & accepted),
            "returned_without_acceptance": len((keys & known) - accepted),
            "unknown_return_attempts": len(keys - known),
            "unknown_return_reserved_cost": sum(reservations[k]["cost_units"] for k in keys - known)}
    return {**totals, "total_reserved_cost_units": sum(x["reserved_cost_units"] for x in totals.values()),
        "reserved_response_attempts": len(reservations), "known_returns": len(known & set(reservations)),
        "executor_accepted_responses": len(accepted), "unknown_return_attempts": len(set(reservations) - known),
        "reservations": list(reservations.values()), "journal_files": files,
        "accounting_complete": not errors, "accounting_errors": errors,
        "cost_rule": "All durable reservations remain charged, including rejected and unknown returns."}


def assemble_training_dataset(*, world, expected_world_sha256, scope, graph, menu,
        task, expected_task_sha256, learner, initial_actions, total_budget,
        behavior_method, snapshots, replicate_ids, planning_draws, tie_multiplier,
        minimum_gain_gap, structural_scores, objective, domain_notes,
        request_namespace, output):
    """Assemble one supplied world; fail closed and retain partial data on error.

    A fresh supplied learner is forked; only trajectory receipts update the
    working learner. Fixed Snapshot assignments precede ALL responses. Milestone
    zero fires after initialization; each other threshold fires at its first
    completed acquisition with cumulative acquisition cost >= threshold. Several
    thresholds crossed by one response each retain an independently keyed block.

    No arbitrary response/policy callbacks, scientific defaults, private
    evaluation, optimizer, retries, resume or replacement-world selection exist.
    total_budget includes initialization but excludes separately charged feedback.
    Runtime/namespace admission and honest public modeling assumptions remain the
    caller's responsibility. This sequential trusted CPU primitive is not a sandbox.
    """
    if (scope not in ("artificial", "policy_train") or type(world) is not WorldDescriptor
            or world.scope != scope or world.sha256 != expected_world_sha256 or graph != world.graph):
        raise ValueError("matching pinned artificial/policy_train descriptor and graph required")
    validate_design(graph, menu)
    if (type(menu) is not tuple or not 1 <= len(menu) <= MAX_ACTIONS
            or type(task) is not PredictionTask or digest(task.payload()) != expected_task_sha256
            or type(learner) is not LinearGaussianSCM or learner.graph != graph or not learner.is_fresh()):
        raise ValueError("bounded exact menu, pinned task and fresh matching learner required")
    task.validate(graph)
    by_id = {a.action_id: a for a in menu}
    if (any(task.target not in q.measure for q in task.queries)
            or type(initial_actions) is not tuple or any(x not in by_id for x in initial_actions)
            or not integer(total_budget, 1) or total_budget > 10**6
            or behavior_method not in METHODS or not integer(planning_draws, 2) or planning_draws > 256
            or type(structural_scores) is not bool or not _name(request_namespace)
            or type(objective) is not str or not 0 < len(objective) <= 16384
            or type(domain_notes) is not str or len(domain_notes) > 16384
            or not finite(tie_multiplier) or tie_multiplier < 0
            or not finite(minimum_gain_gap) or minimum_gain_gap < 0):
        raise ValueError("invalid fixed task/initial/budget/policy/response specification")
    initial_cost = sum(by_id[x].cost_units for x in initial_actions)
    maximum_responses = len(initial_actions) + (total_budget-initial_cost)//min(a.cost_units for a in menu)
    if initial_cost > total_budget or maximum_responses > MAX_HISTORY:
        raise ValueError("initial budget or bounded public history exceeded")
    if (type(snapshots) is not tuple or not 1 <= len(snapshots) <= MAX_SNAPSHOTS
            or type(replicate_ids) is not tuple or not 2 <= len(replicate_ids) <= MAX_REPLICATES
            or any(not _name(r) for r in replicate_ids) or len(set(replicate_ids)) != len(replicate_ids)):
        raise ValueError("bounded fixed snapshot and replicate assignments required")
    last = -1
    for spec in snapshots:
        if (type(spec) is not Snapshot or not integer(spec.acquisition_cost_units)
                or not last < spec.acquisition_cost_units <= total_budget-initial_cost
                or spec.teacher_method not in METHODS or type(spec.candidates) is not tuple
                or len(spec.candidates) == 1 or any(type(c) is not str or c not in by_id for c in spec.candidates)
                or len(set(spec.candidates)) != len(spec.candidates)):
            raise ValueError("strictly increasing milestones, teachers and exact candidate membership required")
        last = spec.acquisition_cost_units
    expected_branch_count = sum(len(s.candidates) for s in snapshots)*len(replicate_ids)
    expected_scoring_count = sum(bool(s.candidates) for s in snapshots)*len(task.queries)*len(replicate_ids)
    if expected_branch_count + expected_scoring_count > MAX_FEEDBACK_RESPONSES:
        raise ValueError("feedback response cap exceeded")

    working = learner.fork()
    caller_pin = digest(_snapshot(learner))
    journal_plan = [{"path": "trajectory.jsonl", "kind": "trajectory"}]
    for si, spec in enumerate(snapshots):
        if spec.candidates:
            for ri in range(len(replicate_ids)):
                for ci in range(len(spec.candidates)):
                    journal_plan.append({"path": f"snapshot-{si:03d}/feedback/replicate-{ri:03d}.branch-{ci:03d}.jsonl",
                                         "kind": "branch"})
                journal_plan.append({"path": f"snapshot-{si:03d}/feedback/replicate-{ri:03d}.scoring.jsonl",
                                     "kind": "scoring"})
    source_pins = {name: _pin(Path(__file__).resolve().parent/name) for name in SOURCE_FILES}
    plan = {"version": VERSION, "scope": scope, "world_id": world.world_id,
        "world_sha256": expected_world_sha256, "task": task.payload(), "task_sha256": expected_task_sha256,
        "graph": [asdict(n) for n in graph], "menu": [asdict(a) for a in menu],
        "initial_actions": initial_actions, "initial_cost_units": initial_cost, "total_trajectory_budget": total_budget,
        "initial_learner": _snapshot(working), "behavior_method": behavior_method,
        "snapshots": [asdict(s) for s in snapshots], "replicate_ids": replicate_ids,
        "milestone_rule": "First completed acquisition crossing each fixed post-initial cost threshold; zero after initialization.",
        "planning_draws": planning_draws, "tie_multiplier": tie_multiplier, "minimum_gain_gap": minimum_gain_gap,
        "structural_scores": structural_scores, "privileged_training_structural_scores": structural_scores,
        "objective": objective, "domain_notes": domain_notes, "request_namespace": request_namespace,
        "policy_seed_rule": "world.keyed_seed(scope, world_id, policy, namespace, role, fixed decision/snapshot index)",
        "maximum_trajectory_responses": maximum_responses, "expected_branch_responses": expected_branch_count,
        "expected_scoring_responses": expected_scoring_count,
        "expected_branch_cost": sum(sum(by_id[c].cost_units for c in s.candidates) for s in snapshots)*len(replicate_ids),
        "expected_scoring_cost": sum(bool(s.candidates) for s in snapshots)*sum(q.cost_units for q in task.queries)*len(replicate_ids),
        "journal_plan": journal_plan, "source_file_sha256": source_pins,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "source_scope": "Observed local bytes after imports; no independent runtime or scientific source authentication.",
        "scientific_provenance_qualified": False, "retry_allowed": False}
    output = Path(output)
    output.mkdir(exist_ok=False)
    began, cpu_began = time.monotonic(), time.process_time()
    started_utc = datetime.now(timezone.utc).isoformat()
    executor, completed, started, initialized, decisions = None, [], [], 0, 0

    def public():
        return make_envelope(executor.state(objective, working.summaries(), domain_notes), task)

    def snapshot(si):
        spec = snapshots[si]
        directory = output/f"snapshot-{si:03d}"
        directory.mkdir(exist_ok=False)
        started.append(si)
        envelope, base_pin = public(), digest(_snapshot(working))
        _save(directory/"envelope.json", envelope)
        state = envelope["state"]
        _save(directory/"selection.json", {"snapshot_index": si, **asdict(spec),
            "actual_acquisition_cost_units": state["spent_units"]-initial_cost,
            "trajectory_request_count": state["requests"], "base_state_sha256": base_pin,
            "selection_depends_on": "fixed cost threshold only", "plan_sha256": digest(plan)})
        decision, metadata = _select(envelope, spec.teacher_method,
            keyed_seed(scope, world.world_id, "policy", request_namespace, "teacher", si), planning_draws)
        _save(directory/"teacher.json", {"method": spec.teacher_method, "decision": decision,
            "metadata": metadata, "envelope_sha256": digest(envelope),
            "chosen_before_branch_responses": True})
        affordable = {a.action_id for a in menu if a.cost_units <= state["remaining_units"] and not state["terminal"]}
        if spec.candidates:
            if not set(spec.candidates) <= affordable:
                raise ValueError("planned candidates are not all affordable; no filtering allowed")
            private = directory/"world-journals"
            private.mkdir(exist_ok=False)
            with ExitStack() as stack:
                branch, score = training_callbacks(stack, world, expected_sha256=expected_world_sha256,
                    menu=menu, candidates=spec.candidates, task=task, replicate_ids=replicate_ids,
                    branch_output=directory/"feedback", journal_directory=private,
                    request_namespace=digest({"namespace": request_namespace, "snapshot_index": si}),
                    structural_scores=structural_scores)
                result = build_replicated_training_preferences(scope=scope, graph=graph, menu=menu,
                    candidates=spec.candidates, learner=working, next_learner_request_id=state["requests"],
                    task=task, replicate_ids=replicate_ids, branch_respond=branch, score_respond=score,
                    tie_multiplier=tie_multiplier, minimum_gain_gap=minimum_gain_gap, output=directory/"feedback")
            pairs = result["feedback"]["pairs"]
            feedback_pin = _pin(directory/"feedback/terminal.json")
        else:
            if affordable or decision != {"stop": True}:
                raise ValueError("empty candidate snapshot requires budget-terminal stop; no gain-based stopping")
            pairs, feedback_pin = [], None
        if digest(_snapshot(working)) != base_pin:
            raise RuntimeError("feedback changed the trajectory learner")
        if len(pairs) != len(spec.candidates)*(len(spec.candidates)-1)//2:
            raise RuntimeError("incomplete candidate pair set")
        receipt = {"version": VERSION, "scope": scope, "world_id": world.world_id,
            "world_sha256": expected_world_sha256, "task_sha256": expected_task_sha256,
            "plan_file_sha256": _pin(output/"plan.json"), "snapshot_index": si,
            "envelope_file_sha256": _pin(directory/"envelope.json"),
            "teacher_file_sha256": _pin(directory/"teacher.json"),
            "selection_file_sha256": _pin(directory/"selection.json"), "feedback_terminal_sha256": feedback_pin,
            "candidate_ids": spec.candidates, "pair_count": len(pairs),
            "base_learner_unchanged": True, "scoring_learner_updates": 0,
            "stop_label_basis": "no affordable trajectory action" if not spec.candidates else None,
            "structural_scores": structural_scores, "scientific_provenance_qualified": False}
        _save(directory/"receipt.json", receipt)
        receipt_pin = _pin(directory/"receipt.json")
        record = {"version": RECORD_VERSION, "scope": scope, "world_id": world.world_id,
            "envelope": envelope, "supervised_action_id": decision.get("action_id"),
            "preferences": [{k: p[k] for k in ("left", "right", "tie", "chosen", "rejected")} for p in pairs],
            "supervision_receipt_sha256": receipt_pin}
        _save(directory/"record.json", record)
        record_pin = _pin(directory/"record.json")
        views = matched_views((directory/"record.json").read_bytes(), record_sha256=record_pin,
                              supervision_receipt_sha256=receipt_pin, allowed_scope=scope)
        _save(directory/"language.json", views["language"])
        _save(directory/"numeric-labels.json", {k: views["numeric"][k] for k in ("chosen", "pairs")})
        with (directory/"numeric-features.npy").open("xb") as handle:
            np.save(handle, views["numeric"]["features"], allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        _save(directory/"alignment.json", views["alignment"])
        completed.append({"snapshot_index": si, "path": f"snapshot-{si:03d}/record.json",
            "record_sha256": record_pin, "receipt_sha256": receipt_pin,
            "pairs": len(pairs), "ties": sum(p["tie"] for p in pairs)})

    def terminal(status, exc=None):
        accounting = _accounting(output, journal_plan)
        state = executor.state(objective, working.summaries(), domain_notes) if executor is not None else None
        value = {"version": VERSION, "status": status, "dataset_complete": status == "complete",
            "scope": scope, "world_id": world.world_id, "world_sha256": expected_world_sha256,
            "task_sha256": expected_task_sha256, "source_file_sha256": source_pins, "plan_sha256": digest(plan),
            "behavior_method": behavior_method, "teacher_assignments": [s.teacher_method for s in snapshots],
            "planned_snapshots": len(snapshots), "started_snapshots": started, "completed_records": completed,
            "unfinished_snapshot_indices": [i for i in range(len(snapshots)) if i not in {r["snapshot_index"] for r in completed}],
            "accounting": accounting, "initialized_responses": initialized, "acquisition_decisions": decisions,
            "trajectory_learner_updates": len(working._requests), "scoring_learner_updates": 0,
            "caller_learner_unchanged": digest(_snapshot(learner)) == caller_pin, "final_or_partial_state": state,
            "actual_initial_cost_units": sum(x["cost_units"] for x in accounting["reservations"]
                if x["kind"] == "trajectory" and x["request_id"] < len(initial_actions)),
            "actual_acquisition_cost_units": sum(x["cost_units"] for x in accounting["reservations"]
                if x["kind"] == "trajectory" and x["request_id"] >= len(initial_actions)),
            "training_pairs": sum(r["pairs"] for r in completed), "tied_pairs": sum(r["ties"] for r in completed),
            "scientific_evaluation_performed": False, "policy_training_performed": False,
            "provenance_qualified": False, "stop_inference_performed": False,
            "retry_allowed": False, "complete_case_filtering_allowed": False,
            "partial_records_authorized_for_training": False,
            "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic()-began, "process_cpu_seconds": time.process_time()-cpu_began}
        if exc is not None:
            value.update(exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        return value

    try:
        _save(output/"plan.json", plan)
        _save(output/"private-world.json", world.private_payload())
        with ExitStack() as stack:
            initial_adapter = stack.enter_context(ResponseAdapter(world, expected_sha256=expected_world_sha256,
                menu=menu, stream="initial", replicate_id="trajectory", request_namespace=request_namespace,
                executor_journal=output/"trajectory.jsonl", journal=output/"initial.world.jsonl",
                max_requests=max(1, len(initial_actions))))
            acquisition_adapter = stack.enter_context(ResponseAdapter(world, expected_sha256=expected_world_sha256,
                menu=menu, stream="measurement", replicate_id="trajectory", request_namespace=request_namespace,
                executor_journal=output/"trajectory.jsonl", journal=output/"acquisition.world.jsonl",
                max_requests=max(1, maximum_responses-len(initial_actions))))

            def respond(action, request):
                adapter = initial_adapter if request < len(initial_actions) else acquisition_adapter
                value = adapter(action, request)
                _save(output/f"trajectory.return-{request:03d}.json", {"journal": "trajectory.jsonl",
                    "request_id": request, "returned_to_wrapper": True, **_raw_return(value)})
                return value

            executor = stack.enter_context(ChargedExecutor(graph, menu, total_budget, output/"trajectory.jsonl", respond))
            for action_id in initial_actions:
                receipt = executor.execute_completion(canonical({"action_id": action_id}),
                    provenance={"kind": "fixed_initial_schedule", "plan_sha256": digest(plan)})
                working.update(receipt)
                initialized += 1
            si = 0
            while True:
                state = executor.state(objective, working.summaries(), domain_notes)
                if not any(a.cost_units <= state["remaining_units"] for a in menu):
                    executor.execute_completion('{"stop":true}', provenance={"kind": "budget_exhausted_or_unspendable"})
                    state = executor.state(objective, working.summaries(), domain_notes)
                while si < len(snapshots) and snapshots[si].acquisition_cost_units <= state["spent_units"]-initial_cost:
                    snapshot(si)
                    si += 1
                if state["terminal"]:
                    break
                envelope = public()
                _save(output/f"decision-{decisions:03d}.envelope.json", envelope)
                decision, metadata = _select(envelope, behavior_method,
                    keyed_seed(scope, world.world_id, "policy", request_namespace, "behavior", decisions), planning_draws)
                _save(output/f"decision-{decisions:03d}.choice.json", {"method": behavior_method,
                    "decision": decision, "metadata": metadata, "envelope_sha256": digest(envelope)})
                decisions += 1
                receipt = executor.execute_completion(canonical(decision),
                    provenance={"kind": "fixed_behavior_method", "method": behavior_method, "decision_index": decisions-1})
                if receipt is None:
                    raise RuntimeError("behavior stopped with affordable actions")
                working.update(receipt)
            if si != len(snapshots):
                raise ValueError("fixed milestones remain unreached; no snapshot filtering allowed")
        result = terminal("complete")
        accounting = result["accounting"]
        if (not accounting["accounting_complete"] or accounting["unknown_return_attempts"]
                or accounting["known_returns"] != accounting["executor_accepted_responses"]
                or accounting["trajectory"]["reserved_cost_units"] != result["final_or_partial_state"]["spent_units"]
                or accounting["trajectory"]["executor_accepted_responses"] != len(working._requests)
                or not result["caller_learner_unchanged"]):
            raise RuntimeError("incomplete trajectory/accounting isolation")
        for kind in ("branch", "scoring"):
            if (accounting[kind]["reserved_cost_units"] != plan["expected_"+kind+"_cost"]
                    or accounting[kind]["executor_accepted_responses"] != plan["expected_"+kind+"_responses"]):
                raise RuntimeError("incomplete planned feedback accounting")
        _save(output/"terminal.json", result)
        return result
    except BaseException as exc:
        _save(output/"failure.json", terminal("failed", exc))
        raise

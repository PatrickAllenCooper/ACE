"""Bounded supplied-world CPU collection; no world draws, policy training or private evaluation.

Independent pins authenticate supplied bindings, not their scientific admission.
Trusted sequential caller only: runtime, scheduler and scientific freeze remain
external. An existing output, including an interrupted intent, cannot be reused.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import time

from llm_intervention_contract import canonical, digest, finite, integer
from llm_intervention_replicated_branches import _save
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_screen_protocol import REQUIRED_SOURCE_PATHS, assembly_recipe, validate_proposal
from llm_intervention_supervision import matched_views
from llm_intervention_task import MAX_ACTIONS, MAX_HISTORY, PredictionTask
from llm_intervention_training_dataset import (
    MAX_FEEDBACK_RESPONSES, MAX_REPLICATES, MAX_SNAPSHOTS, METHODS, Snapshot,
    _accounting, assemble_training_dataset,
)
from llm_intervention_world import WorldDescriptor, _menu as validate_response_menu


VERSION = "ace-supplied-training-collection-v1"
SOURCE_PATHS = tuple(sorted((*REQUIRED_SOURCE_PATHS,
    "scripts/research/llm_intervention_training_collection.py")))
MAX_WORLDS, MAX_RECORDS, MAX_ATTEMPTS, MAX_COST = 32, 256, 16384, 1000000
ASSEMBLY_KEYS = frozenset(("scope", "graph", "menu", "task", "expected_task_sha256",
    "initial_actions", "total_budget", "behavior_method", "snapshots", "replicate_ids",
    "planning_draws", "tie_multiplier", "minimum_gain_gap", "structural_scores",
    "objective", "domain_notes", "request_namespace"))
RUNTIME_BINDINGS = ("world", "expected_world_sha256", "learner", "output")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if type(value) is dict:
        return {k: _plain(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return [_plain(v) for v in value]
    return value


def configuration_payload(configuration):
    """Canonical JSON projection of the complete typed assembly_recipe result.

    Pin this independently before collection, including learner parameters and
    the pending-runtime markers. The markers themselves grant no authorization.
    """
    return json.loads(canonical(_plain(configuration)))


def _pin(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_terminal(output, value):
    """Retain a fsynced candidate; exclusive link is the last fallible operation."""
    candidate = output/"terminal-candidate.json"
    _save(candidate, value)
    os.link(candidate, output/"terminal.json")


def _failure_receipt(output, failure):
    # An interruption may arrive after link committed. Never contradict a
    # visible terminal with a second final disposition. Caller must reconcile.
    if os.path.lexists(output/"terminal.json"):
        _save(output/"publication-interruption.json",
              dict(failure, status="publication_interrupted_reconciliation_required"))
    else:
        _save(output/"failure.json", failure)


def _journal_plan(configuration):
    """Addresses from the pinned primitive's fixed recipe, before any response."""
    a = configuration["assembly_parameters"]
    plan = [{"path": "trajectory.jsonl", "kind": "trajectory"}]
    for si, spec in enumerate(a["snapshots"]):
        if spec.candidates:
            for ri in range(len(a["replicate_ids"])):
                prefix = f"snapshot-{si:03d}/feedback/replicate-{ri:03d}"
                plan.extend({"path": prefix+f".branch-{ci:03d}.jsonl", "kind": "branch"}
                            for ci in range(len(spec.candidates)))
                plan.append({"path": prefix+".scoring.jsonl", "kind": "scoring"})
    return plan


def _world_accounting(directory, journal_plan):
    # Reconcile readable journals even if the primitive plan write was truncated.
    # Optional reconstruction cannot suppress the original batch failure.
    try:
        value = _accounting(directory, journal_plan)
    except Exception as exc:
        value = {k: 0 for k in ("total_reserved_cost_units", "reserved_response_attempts",
            "known_returns", "executor_accepted_responses", "unknown_return_attempts")}
        value.update(accounting_complete=False,
            accounting_errors=[{"error": "journal reconstruction failed: " + repr(exc)}])
    try:
        if directory.exists():
            plan = json.loads((directory/"plan.json").read_text())
            _require(plan["journal_plan"] == journal_plan, "world plan differs from predeclared journal roster")
    except Exception as exc:
        value["accounting_complete"] = False
        value["accounting_errors"].append({"error": "world plan unreadable/unmatched: " + repr(exc)})
    value["counts_are_lower_bounds"] = not value["accounting_complete"]
    return value


def _adopt_recipes(scope, roster, configurations, proposal, proposal_pin, sources):
    if scope == "artificial":
        _require(proposal is None and proposal_pin is None, "artificial inputs cannot adopt a scientific proposal")
        return
    protocol_pins = {p: sources[p] for p in REQUIRED_SOURCE_PATHS}
    _require(validate_proposal(proposal, expected_source_pins=protocol_pins) == proposal_pin,
             "independent current proposal pin required")
    _require(roster == tuple(r["world_id"] for r in proposal["training_worlds"]),
             "complete ordered current policy_train proposal roster required")
    for wid in roster:
        expected = assembly_recipe(proposal, world_id=wid, expected_source_pins=protocol_pins)
        _require(canonical(configuration_payload(configurations[wid])) == canonical(configuration_payload(expected)),
                 "configuration differs from exact current assembly_recipe")


def _validate_configuration(world, configuration, scope):
    """Static whole-roster preflight; never invoke a response to validate inputs."""
    _require(type(configuration) is dict and set(configuration) == {
        "execution_authorized", "runtime_bindings_pending", "learner_parameters", "assembly_parameters"},
        "exact assembly_recipe configuration fields required")
    _require(configuration["execution_authorized"] is False
             and configuration["runtime_bindings_pending"] == RUNTIME_BINDINGS,
             "recipe runtime bindings must remain explicit")
    a, learner = configuration["assembly_parameters"], configuration["learner_parameters"]
    _require(type(a) is dict and set(a) == ASSEMBLY_KEYS and type(learner) is dict
             and set(learner) == {"graph", "noise_variances", "prior_variance"}, "exact assembly/learner fields required")
    graph, menu, task = a["graph"], a["menu"], a["task"]
    _require(a["scope"] == scope and graph == world.graph and learner["graph"] == graph,
             "scope and descriptor/assembly/learner graphs must match")
    _require(type(menu) is tuple and 1 <= len(menu) <= MAX_ACTIONS and type(task) is PredictionTask,
             "bounded typed menu and task required")
    validate_response_menu(graph, menu)
    task.validate(graph)
    validate_response_menu(graph, task.queries)
    _require(digest(task.payload()) == a["expected_task_sha256"]
             and all(task.target in q.measure for q in task.queries), "pinned measured task required")
    by_id, initial, budget = {x.action_id: x for x in menu}, a["initial_actions"], a["total_budget"]
    _require(type(initial) is tuple and all(x in by_id for x in initial)
             and integer(budget, 1) and budget <= MAX_COST, "bounded initial schedule/budget required")
    initial_cost = sum(by_id[x].cost_units for x in initial)
    maximum = len(initial) + (budget-initial_cost)//min(x.cost_units for x in menu)
    _require(initial_cost <= budget and maximum <= MAX_HISTORY, "bounded trajectory history required")
    _require(a["behavior_method"] in METHODS and integer(a["planning_draws"], 2)
             and a["planning_draws"] <= 256 and type(a["structural_scores"]) is bool
             and all(finite(a[k]) and a[k] >= 0 for k in ("tie_multiplier", "minimum_gain_gap")),
             "fixed bounded behavior/feedback settings required")
    for key, minimum, maximum_length in (("objective", 1, 16384), ("domain_notes", 0, 16384),
                                         ("request_namespace", 1, 256)):
        _require(type(a[key]) is str and minimum <= len(a[key]) <= maximum_length, "bounded text/namespace required")
    snapshots, replicates = a["snapshots"], a["replicate_ids"]
    _require(type(snapshots) is tuple and 1 <= len(snapshots) <= MAX_SNAPSHOTS
             and type(replicates) is tuple and 2 <= len(replicates) <= MAX_REPLICATES
             and all(type(r) is str and 1 <= len(r) <= 256 for r in replicates)
             and len(set(replicates)) == len(replicates), "fixed snapshots and replicates required")
    last = -1
    for s in snapshots:
        _require(type(s) is Snapshot and integer(s.acquisition_cost_units)
                 and last < s.acquisition_cost_units <= budget-initial_cost
                 and s.teacher_method in METHODS and type(s.candidates) is tuple
                 and len(s.candidates) != 1 and all(type(c) is str and c in by_id for c in s.candidates)
                 and len(set(s.candidates)) == len(s.candidates), "invalid fixed snapshot recipe")
        last = s.acquisition_cost_units
    branches = sum(len(s.candidates) for s in snapshots)*len(replicates)
    scores = sum(bool(s.candidates) for s in snapshots)*len(replicates)*len(task.queries)
    _require(branches+scores <= MAX_FEEDBACK_RESPONSES, "feedback cap exceeded")
    if scope == "policy_train":
        _require({m.node: m.noise_variance for m in world.mechanisms} == learner["noise_variances"],
                 "descriptor noise differs from proposed known noise")
    fresh = LinearGaussianSCM(**learner)  # Constructor validates all public prior/noise inputs.
    cost = budget + len(replicates)*sum(sum(by_id[c].cost_units for c in s.candidates)
        + (sum(q.cost_units for q in task.queries) if s.candidates else 0) for s in snapshots)
    return fresh, maximum+branches+scores, cost


def _inventory(output, planned, worlds):
    """Keep every planned record, including failed/unstarted and partial records."""
    records = []
    for row in planned:
        entry = dict(row, status="unstarted", record_sha256=None, supervision_receipt_sha256=None)
        directory = output/row["path"]
        record, receipt = directory/"record.json", directory/"receipt.json"
        if record.exists() and receipt.exists():
            raw, receipt_raw = record.read_bytes(), receipt.read_bytes()
            r, s = json.loads(raw), json.loads(receipt_raw)
            rp, sp = _pin(record), _pin(receipt)
            matched_views(raw, record_sha256=rp, supervision_receipt_sha256=sp, allowed_scope=row["scope"])
            _require(r["world_id"] == row["world_id"] and s["world_id"] == row["world_id"]
                     and s["world_sha256"] == worlds[row["world_id"]].sha256
                     and s["snapshot_index"] == row["snapshot_index"]
                     and s["plan_file_sha256"] == _pin(directory.parent/"plan.json")
                     and s["candidate_ids"] == row["candidate_ids"]
                     and s["pair_count"] == len(row["pairs"])
                     and [[p["left"], p["right"]] for p in r["preferences"]] == row["pairs"],
                     "record/receipt differs from full predeclared pair roster")
            entry.update(status="complete", record_sha256=rp, supervision_receipt_sha256=sp)
        elif directory.exists():
            entry["status"] = "failed"
        records.append(entry)
    return records


def collect_training_batch(*, scope, world_roster, expected_roster_sha256, worlds,
        world_pins, configurations, configuration_pins, expected_source_pins, output,
        proposal=None, expected_proposal_sha256=None):
    """Execute the entire supplied roster sequentially, stopping at first failure.

    Dicts are keyed by integer world ID; roster is an independently pinned tuple.
    Configuration pins are digest(configuration_payload(config)); descriptor pins
    are WorldDescriptor.sha256. Source paths are exactly SOURCE_PATHS. policy_train
    additionally requires the current exact proposal and its independent digest.
    No callback, retry, resume, replacement, filtering or partial training path.
    """
    _require(scope in ("artificial", "policy_train") and type(world_roster) is tuple
             and 1 <= len(world_roster) <= MAX_WORLDS and all(integer(w) for w in world_roster)
             and len(set(world_roster)) == len(world_roster), "bounded unique ordered training roster required")
    for mapping in (worlds, world_pins, configurations, configuration_pins):
        _require(type(mapping) is dict and all(type(k) is int for k in mapping)
                 and set(mapping) == set(world_roster), "exact supplied roster membership required")
    worlds, configurations, world_pins, configuration_pins, proposal, sources = deepcopy(
        (worlds, configurations, world_pins, configuration_pins, proposal, expected_source_pins))
    output = Path(output)
    output.mkdir(exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    started_utc = datetime.now(timezone.utc).isoformat()
    entries = [{"world_id": w, "path": f"world-{i:03d}", "status": "unstarted"}
               for i, w in enumerate(world_roster)]
    planned, active, stage = [], None, "preflight"

    def finish(status, exc=None):
        per_world = []
        for entry in entries:
            directory = output/entry["path"]
            accounting = _world_accounting(directory, entry.get("journal_plan", []))
            per_world.append({**entry, "accounting": accounting})
        totals = {key: sum(row["accounting"][key] for row in per_world) for key in (
            "total_reserved_cost_units", "reserved_response_attempts", "known_returns",
            "executor_accepted_responses", "unknown_return_attempts")}
        result = {"version": VERSION, "scope": scope, "status": status, "stage": stage,
            "world_roster": world_roster, "worlds": per_world, "record_roster": [r["record_id"] for r in planned],
            "planned_records": planned, "accounting": totals,
            "accounting_complete": all(r["accounting"]["accounting_complete"] for r in per_world),
            "accounting_totals_are_lower_bounds": any(not r["accounting"]["accounting_complete"] for r in per_world),
            "exposure_ready": status == "complete", "retry_allowed": False,
            "partial_records_authorized_for_training": False, "complete_case_filtering_allowed": False,
            "scientific_provenance_qualified": False, "scientific_evaluation_performed": False,
            "policy_training_performed": False, "source_runtime_scheduler_freeze_qualified": False,
            "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic()-started, "process_cpu_seconds": time.process_time()-cpu}
        if exc is not None:
            result.update(exception_type=type(exc).__name__, message=str(exc))
        return result

    try:
        _save(output/"intent.json", {"version": VERSION, "scope": scope, "world_roster": world_roster,
            "expected_roster_sha256": expected_roster_sha256, "worlds": entries,
            "world_pins": {str(k): v for k, v in world_pins.items()},
            "configuration_pins": {str(k): v for k, v in configuration_pins.items()},
            "expected_source_pins": sources, "proposal_sha256": expected_proposal_sha256,
            "failure_rule": "stop entire batch at first failure; no retry, resume, replacement or subset exposure",
            "started_utc": started_utc})
        _require(digest(world_roster) == expected_roster_sha256, "independent ordered roster pin differs")
        _require(type(sources) is dict and set(sources) == set(SOURCE_PATHS), "exact source-pin map required")
        observed = {p: _pin(Path(__file__).resolve().parent/Path(p).name) for p in SOURCE_PATHS}
        _save(output/"observed-source-pins.json", observed)
        _require(observed == sources, "supplied source pins differ from current local bytes")
        if proposal is not None:
            _save(output/"supplied-proposal.json", proposal)
        # Save and authenticate EVERY descriptor/config before constructing any response adapter.
        for i, wid in enumerate(world_roster):
            active = i
            world, config = worlds[wid], configurations[wid]
            _require(type(world) is WorldDescriptor, "supplied typed world descriptor required")
            _save(output/f"input-{i:03d}.json", {"world": world.private_payload(),
                "configuration": configuration_payload(config)})
            _require(world.scope == scope and world.world_id == wid and world.sha256 == world_pins[wid],
                     "independent descriptor identity/scope/pin differs")
            _require(digest(configuration_payload(config)) == configuration_pins[wid],
                     "independent configuration pin differs")
        active = None
        _adopt_recipes(scope, world_roster, configurations, proposal, expected_proposal_sha256, sources)
        learners, attempts, cost = {}, 0, 0
        for i, wid in enumerate(world_roster):
            active = i
            learner, count, charge = _validate_configuration(worlds[wid], configurations[wid], scope)
            learners[wid], attempts, cost = learner, attempts+count, cost+charge
            entries[i]["journal_plan"] = _journal_plan(configurations[wid])
            for si, s in enumerate(configurations[wid]["assembly_parameters"]["snapshots"]):
                planned.append({"record_id": f"{scope}/{wid}/snapshot-{si:03d}", "scope": scope,
                    "world_id": wid, "snapshot_index": si, "path": f"world-{i:03d}/snapshot-{si:03d}",
                    "candidate_ids": list(s.candidates), "pairs": [list(p) for p in combinations(s.candidates, 2)]})
        active = None
        _require(len(planned) <= MAX_RECORDS and attempts <= MAX_ATTEMPTS and cost <= MAX_COST,
                 "batch record/response/cost bound exceeded")
        _save(output/"plan.json", {"version": VERSION, "world_roster": world_roster, "worlds": entries, "records": planned,
            "maximum_response_attempts": attempts, "maximum_reserved_cost_units": cost,
            "intent_sha256": _pin(output/"intent.json"), "source_file_sha256": sources})
        stage = "collection"
        for i, wid in enumerate(world_roster):
            active = i
            entries[i]["status"] = "started"
            _save(output/f"world-{i:03d}.start.json", {**entries[i], "plan_sha256": _pin(output/"plan.json")})
            result = assemble_training_dataset(world=worlds[wid], expected_world_sha256=world_pins[wid],
                learner=learners[wid], output=output/entries[i]["path"],
                **configurations[wid]["assembly_parameters"])
            _require(result["dataset_complete"], "single-world assembly did not complete")
            entries[i]["status"] = "complete"
        active, stage = None, "inventory"
        records = _inventory(output, planned, worlds)
        _require(all(r["status"] == "complete" for r in records), "incomplete record/receipt inventory")
        inventory = {"version": VERSION, "scope": scope, "record_roster": [r["record_id"] for r in records],
            "records": records, "record_pins": {r["record_id"]: r["record_sha256"] for r in records},
            "supervision_receipt_pins": {r["record_id"]: r["supervision_receipt_sha256"] for r in records},
            "collection_plan_sha256": _pin(output/"plan.json")}
        terminal = finish("complete")
        _require(all(e["status"] == "complete" for e in entries)
                 and terminal["accounting_complete"] and not terminal["accounting"]["unknown_return_attempts"],
                 "incomplete batch accounting")
        _save(output/"inventory.json", inventory)
        terminal["inventory_sha256"] = _pin(output/"inventory.json")
        _publish_terminal(output, terminal)
        return terminal
    except BaseException as exc:
        if active is not None:
            entries[active]["status"] = "failed"
        failure = finish("failed", exc)
        # Partial records are evidence only, never an exposure input inventory.
        try:
            failure["partial_record_inventory"] = _inventory(output, planned, worlds)
        except Exception as inventory_error:
            failure["inventory_error"] = str(inventory_error)
        _failure_receipt(output, failure)
        raise

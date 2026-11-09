"""Paired supplied-world execution and private reporting, without world draws.

Run each arm separately so CPU policies need not occupy an allocated GPU. The
caller freezes source/runtime/model provenance and owns persistent workers. Only
after all arms and service cleanup can the private finalizer score the roster.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from llm_intervention_contract import canonical, digest, integer, finite, validate_design
from llm_intervention_episode import run_episode
from llm_intervention_persistent_policy import PersistentPolicy
from llm_intervention_replicated_branches import _raw_return
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_task import TaskProcessPolicy, PredictionTask, action_from_dict, MAX_HISTORY
from llm_intervention_world import WorldDescriptor, ResponseAdapter


VERSION = "ace-paired-episode-v1"
MAX_BYTES = 32*1024*1024


def _raw(value):
    return (canonical(value)+"\n").encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _name(value):
    return type(value) is str and 0 < len(value) <= 256


def _write(path, raw):
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def _save(path, value):
    _write(path, _raw(value))


def _publish(path, value):
    candidate = path.with_suffix(".candidate.json")
    _save(candidate, value)
    os.link(candidate, path)


def _task(payload, graph):
    if type(payload) is not dict or set(payload) != {"target", "queries", "weights", "scale", "normalized_mse_tolerance"}:
        raise ValueError("exact public task required")
    task = PredictionTask(payload["target"], tuple(action_from_dict(q) for q in payload["queries"]),
                          tuple(payload["weights"]), payload["scale"], payload["normalized_mse_tolerance"]).validate(graph)
    if any(w != 1/len(task.queries) for w in task.weights):
        raise ValueError("reporter requires equal fixed query weights")
    return task


def prepare_plan(*, world, world_sha256, task, menu, noise_variances, prior_variance,
                 initial_actions, acquisition_cap_units, initial_binding, measurement_bindings,
                 policy_specs, objective, domain_notes, outer_provenance_receipt_sha256):
    if (type(world) is not WorldDescriptor or world.scope not in ("artificial", "development")
            or world.sha256 != world_sha256 or type(task) is not PredictionTask
            or type(menu) is not tuple or not 1 <= len(menu) <= 32
            or type(initial_actions) is not tuple or not integer(acquisition_cap_units, 1)
            or not _pin(outer_provenance_receipt_sha256)):
        raise ValueError("pinned supplied artificial/development world and bounded design required")
    validate_design(world.graph, menu)
    _task(task.payload(), world.graph)
    learner = LinearGaussianSCM(world.graph, noise_variances, prior_variance)
    by_id = {a.action_id: a for a in menu}
    if (any(a not in by_id for a in initial_actions)
            or len(initial_actions)+acquisition_cap_units//min(a.cost_units for a in menu) > MAX_HISTORY
            or not _name(objective) or type(domain_notes) is not str or len(domain_notes) > 16384):
        raise ValueError("bounded initial history/objective required")
    initial_cost = sum(by_id[a].cost_units for a in initial_actions)
    if initial_cost+acquisition_cap_units > 10**6:
        raise ValueError("total cost bound exceeded")
    if type(policy_specs) is not tuple or not 1 <= len(policy_specs) <= 8:
        raise ValueError("complete ordered bounded policy roster required")
    ids = []
    for spec in policy_specs:
        if (type(spec) is not dict or set(spec) != {"policy_id", "kind", "command", "timeout_seconds", "identity_sha256", "policy_provenance_sha256"}
                or not _name(spec["policy_id"]) or spec["policy_id"] in ids
                or spec["kind"] not in ("task_process", "persistent")
                or type(spec["command"]) is not list or not 1 <= len(spec["command"]) <= 64
                or any(type(x) is not str or not x or len(x) > 4096 for x in spec["command"])
                or not finite(spec["timeout_seconds"]) or not 0 < spec["timeout_seconds"] <= 3600
                or not _pin(spec["policy_provenance_sha256"])
                or (not _pin(spec["identity_sha256"]) if spec["kind"] == "persistent" else spec["identity_sha256"] is not None)):
            raise ValueError("explicit policy command, timeout and provenance required")
        ids.append(spec["policy_id"])
    if type(measurement_bindings) is not dict or set(measurement_bindings) != set(ids):
        raise ValueError("exact per-arm measurement bindings required")
    for binding in (initial_binding, *measurement_bindings.values()):
        if (type(binding) is not dict or set(binding) != {"request_namespace", "replicate_id"}
                or not all(_name(v) for v in binding.values())):
            raise ValueError("explicit response stream namespace and replicate required")
    # Distinct measurement namespaces preserve arm identity. Initial keys alone
    # are intentionally shared; no uncharged cache or preliminary response draw.
    if len({canonical(v) for v in measurement_bindings.values()}) != len(ids):
        raise ValueError("measurement bindings must be distinct across arms")
    payload = {"version": VERSION, "world": world.private_payload(), "world_sha256": world_sha256,
        "task": task.payload(), "menu": [asdict(a) for a in menu],
        "learner": {"graph": [asdict(n) for n in world.graph], "noise_variances": noise_variances, "prior_variance": prior_variance},
        "initial_actions": initial_actions, "initial_cost_units": initial_cost,
        "acquisition_cap_units": acquisition_cap_units, "initial_binding": initial_binding,
        "measurement_bindings": measurement_bindings, "policy_specs": policy_specs,
        "objective": objective, "domain_notes": domain_notes,
        "outer_provenance_receipt_sha256": outer_provenance_receipt_sha256,
        "scientific_provenance_qualified": False, "retry_allowed": False}
    if len(_raw(payload)) > MAX_BYTES:
        raise ValueError("plan byte bound exceeded")
    return json.loads(canonical(payload))


def _validate(plan, expected):
    if not _pin(expected) or digest(plan) != expected or plan.get("version") != VERSION:
        raise ValueError("independent paired plan pin required")
    p = json.loads(canonical(plan))
    world = WorldDescriptor.from_private_payload(p["world"], expected_sha256=p["world_sha256"])
    task = _task(p["task"], world.graph)
    menu = tuple(action_from_dict(a) for a in p["menu"])
    rebuilt = prepare_plan(world=world, world_sha256=world.sha256, task=task, menu=menu,
        noise_variances=p["learner"]["noise_variances"], prior_variance=p["learner"]["prior_variance"],
        initial_actions=tuple(p["initial_actions"]), acquisition_cap_units=p["acquisition_cap_units"],
        initial_binding=p["initial_binding"], measurement_bindings=p["measurement_bindings"],
        policy_specs=tuple(p["policy_specs"]), objective=p["objective"], domain_notes=p["domain_notes"],
        outer_provenance_receipt_sha256=p["outer_provenance_receipt_sha256"])
    if canonical(p) != canonical(rebuilt):
        raise ValueError("paired plan contains altered or extra fields")
    return p, world, task, menu


def run_world_arm(*, plan, plan_sha256, policy_id, policy, output):
    """Execute one predeclared arm once, retaining ordinary failures as results.

    Borrowed persistent workers may span worlds. Caller closes them before final
    reporting and owns deadlines/allocations. Signals propagate after preservation.
    No policy training, private scoring, service launch or world generation here.
    """
    p, world, task, menu = _validate(plan, plan_sha256)
    spec = next((s for s in p["policy_specs"] if s["policy_id"] == policy_id), None)
    if spec is None:
        raise ValueError("policy outside complete roster")
    expected_class = PersistentPolicy if spec["kind"] == "persistent" else TaskProcessPolicy
    if (type(policy) is not expected_class or list(policy.command) != spec["command"]
            or policy.timeout != spec["timeout_seconds"]
            or spec["kind"] == "persistent" and (policy.closed or policy.identity_pin != spec["identity_sha256"])
            or spec["kind"] == "task_process" and canonical(policy.task.payload()) != canonical(task.payload())):
        raise ValueError("live public policy differs from fixed command/identity/task")
    output = Path(output)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    _save(output/"paired-plan.json", p)
    for name, value in (("world.json", p["world"]), ("task.json", p["task"]), ("learner.json", p["learner"])):
        _save(output/name, value)
    _save(output/"arm-start.json", {"utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_sha256,
        "policy_id": policy_id, "policy_spec": spec, "scientific_provenance_qualified": False})
    adapters, errors, interruption = [], [], None
    episode_clean, adapters_clean = False, True
    learner = LinearGaussianSCM(world.graph, p["learner"]["noise_variances"], p["learner"]["prior_variance"])
    initial_count = len(p["initial_actions"])
    total_budget = p["initial_cost_units"]+p["acquisition_cap_units"]
    max_requests = initial_count+p["acquisition_cap_units"]//min(a.cost_units for a in menu)
    session = {"kind": spec["kind"], "identity_sha256": spec["identity_sha256"],
        "pid": policy.process.pid if spec["kind"] == "persistent" else None,
        "request_before": policy.previous if spec["kind"] == "persistent" else None,
        "count_before": policy.count if spec["kind"] == "persistent" else None}
    try:
        if spec["kind"] == "persistent":
            policy.bind_task(task)
        initial = None
        if initial_count:
            initial = ResponseAdapter(world, expected_sha256=world.sha256, menu=menu, stream="initial",
                **p["initial_binding"], executor_journal=output/"episode/journal.jsonl",
                journal=output/"initial-world.jsonl", max_requests=initial_count)
            adapters.append(initial)
        measurement = None
        if max_requests > initial_count:
            measurement = ResponseAdapter(world, expected_sha256=world.sha256, menu=menu, stream="measurement",
                **p["measurement_bindings"][policy_id], executor_journal=output/"episode/journal.jsonl",
                journal=output/"measurement-world.jsonl", max_requests=max_requests)
            adapters.append(measurement)

        def respond(action, request_id):
            adapter = initial if request_id < initial_count else measurement
            if adapter is None:
                raise RuntimeError("response attempted outside the admitted request budget")
            response = adapter(action, request_id)
            _save(output/f"return-{request_id:03d}.json", {"request_id": request_id,
                "action": asdict(action), "returned_to_wrapper": True, "response": _raw_return(response)})
            return response

        run_episode(graph=world.graph, menu=menu, budget=total_budget, initial_actions=tuple(p["initial_actions"]),
            learner=learner, respond=respond, policy=policy, objective=p["objective"],
            domain_notes=p["domain_notes"], output=output/"episode")
        episode_clean = True
    except BaseException as exc:
        errors.append({"phase": "episode", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        if not isinstance(exc, Exception):
            interruption = exc
    finally:
        for adapter in reversed(adapters):
            try:
                adapter.close()
            except BaseException as exc:
                adapters_clean = False
                errors.append({"phase": "adapter_close", "type": type(exc).__name__, "message": str(exc)})
                if not isinstance(exc, Exception):
                    interruption = exc
    _save(output/"execution-diagnostics.json", {"errors": errors, "elapsed_seconds": time.monotonic()-started,
        "episode_returned_cleanly": episode_clean, "adapters_closed_cleanly": adapters_clean,
        "persistent_session_owned_by_caller": spec["kind"] == "persistent", "retry_allowed": False})
    terminals = [output/"episode"/n for n in ("terminal.json", "failure.json") if (output/"episode"/n).is_file()]
    inputs = {}
    unresolved = None
    try:
        if len(terminals) != 1:
            raise ValueError("one unambiguous original episode disposition required")
        inputs = {n: (output/n).read_bytes() for n in ("world.json", "task.json", "learner.json")}
        inputs.update({"episode-plan.json": (output/"episode/plan.json").read_bytes(),
            "executor.jsonl": (output/"episode/journal.jsonl").read_bytes(),
            "episode-disposition.json": terminals[0].read_bytes()})
        inputs.update({q.name: q.read_bytes() for q in output.glob("return-*.json")})
        journal = [json.loads(line) for line in inputs["executor.jsonl"].splitlines()]
        stops = [e for e in journal if e.get("event") == "stop"]
        proposals = {e["proposal_id"]: e for e in journal if e.get("event") == "proposal"}
        capped = stops and proposals[stops[-1]["proposal_id"]]["provenance"].get("kind") == "budget_cap"
        reason = ("budget_exhausted" if capped else "policy_stop") if episode_clean and adapters_clean else (
            "adapter_close_failure:see_execution_diagnostics" if episode_clean else "episode_failure:see_execution_diagnostics")
        outer = {"version": VERSION, "scope": world.scope, "world_id": world.world_id, "policy_id": policy_id,
            "status": "complete" if episode_clean and adapters_clean else "failed", "reason": reason,
            "episode_returned_cleanly": episode_clean, "adapters_closed_cleanly": adapters_clean,
            "episode_disposition_sha256": _sha(inputs["episode-disposition.json"]),
            "outer_provenance_receipt_sha256": p["outer_provenance_receipt_sha256"]}
        inputs["outer-outcome.json"] = _raw(outer)
        (output/"sealed-inputs").mkdir()
        for name, raw in inputs.items():
            _write(output/"sealed-inputs"/name, raw)
    except Exception as exc:
        unresolved = type(exc).__name__+": "+str(exc)
    session.update(request_after=policy.previous if spec["kind"] == "persistent" else None,
                   count_after=policy.count if spec["kind"] == "persistent" else None)
    receipt = {"version": VERSION, "plan_sha256": plan_sha256, "policy_id": policy_id,
        "status": "raw_ready" if unresolved is None else "unresolved", "unresolved_reason": unresolved,
        "session": session, "input_sha256": {n: _sha(b) for n, b in inputs.items()} if unresolved is None else {},
        "source_evidence_sha256": {str(q.relative_to(output)): _sha(q.read_bytes()) for q in sorted(output.rglob("*")) if q.is_file()},
        "scientific_provenance_qualified": False, "private_scores_computed": False,
        "new_training_updates": 0, "retry_allowed": False}
    _publish(output/"arm-receipt.json", receipt)
    if interruption is not None:
        raise interruption
    return receipt


def _load_arm(path, expected, plan_pin, policy_id):
    path = Path(path)
    members = [q for q in path.rglob("*") if q.is_file()]
    if len(members) > 1000 or sum(q.stat().st_size for q in members) > MAX_BYTES:
        raise ValueError("arm custody exceeds byte/file bound")
    raw = (path/"arm-receipt.json").read_bytes()
    if not _pin(expected) or _sha(raw) != expected:
        raise ValueError("independent arm receipt pin differs")
    receipt = json.loads(raw)
    if (receipt["version"] != VERSION or receipt["plan_sha256"] != plan_pin
            or receipt["policy_id"] != policy_id or receipt["status"] != "raw_ready"):
        raise ValueError("arm identity or raw disposition differs")
    files = {str(q.relative_to(path)): q for q in path.rglob("*") if q.is_file()}
    if set(files) != set(receipt["source_evidence_sha256"]) | {"arm-receipt.json", "arm-receipt.candidate.json"}:
        raise ValueError("arm evidence inventory differs")
    if files["arm-receipt.candidate.json"].read_bytes() != raw:
        raise ValueError("arm publication candidate differs")
    for name, pin in receipt["source_evidence_sha256"].items():
        if files[name].is_symlink() or _sha(files[name].read_bytes()) != pin:
            raise ValueError("arm evidence changed: "+name)
    inputs = {n: files["sealed-inputs/"+n].read_bytes() for n in receipt["input_sha256"]}
    if {n: _sha(b) for n, b in inputs.items()} != receipt["input_sha256"]:
        raise ValueError("sealed input pin differs")
    return receipt, inputs


def _initial_candidates(inputs, count):
    # Candidate only. seal_episode authenticates the complete chain and exact
    # fields before any candidate can become a common initialization witness.
    responses = [e for e in map(json.loads, inputs["executor.jsonl"].splitlines()) if e.get("event") == "response"]
    if len(responses) < count:
        return None
    fields = ("request_id", "action_id", "kind", "targets", "values", "cost_units")
    return tuple(_sha(_raw({k: e[k] for k in fields})) for e in responses[:count])


def finalize_paired_world(*, plan, plan_sha256, arm_outputs, persistent_closed, output):
    """Private all-arm sealing/reporting after execution and persistent cleanup.

    arm_outputs maps EVERY policy ID to {path,receipt_sha256,missing_reason}.
    An explicit missing cell uses path=None,pin=None and a reason. For persistent
    arms, persistent_closed supplies raw closed.json bytes plus independent pin.
    Outer actual source/model/runtime/scheduler qualification remains required.
    """
    from llm_intervention_episode_seal import seal_episode
    from llm_intervention_cost_curves import report_cost_curve
    from llm_intervention_screen_reporting import summarize_screen
    p, world, task, menu = _validate(plan, plan_sha256)
    specs = {s["policy_id"]: s for s in p["policy_specs"]}
    if type(arm_outputs) is not dict or set(arm_outputs) != set(specs) or type(persistent_closed) is not dict:
        raise ValueError("complete arm and explicit service disposition mappings required")
    if set(persistent_closed) != {k for k, s in specs.items() if s["kind"] == "persistent"}:
        raise ValueError("exact persistent-service closure roster required")
    output = Path(output)
    output.mkdir(exist_ok=False)
    _save(output/"plan.json", p)
    _save(output/"arm-inventory.json", arm_outputs)
    valid, candidates, errors, bundles = {}, {}, {}, {}
    for index, (pid, spec) in enumerate(specs.items()):
        try:
            item = arm_outputs[pid]
            if type(item) is not dict or set(item) != {"path", "receipt_sha256", "missing_reason"}:
                raise ValueError("invalid explicit arm item")
            if item["path"] is None:
                if item["receipt_sha256"] is not None or not _name(item["missing_reason"]):
                    raise ValueError("missing arm requires reason and no invented pin")
                raise ValueError("missing arm: "+item["missing_reason"])
            if type(item["path"]) is not str or item["missing_reason"] is not None:
                raise ValueError("raw arm requires path/pin and no missing reason")
            receipt, inputs = _load_arm(item["path"], item["receipt_sha256"], plan_sha256, pid)
            for name, value in (("world.json", p["world"]), ("task.json", p["task"]), ("learner.json", p["learner"])):
                if inputs[name] != _raw(value):
                    raise ValueError("arm inputs differ from common plan: "+name)
            original_plan = json.loads(inputs["episode-plan.json"])
            expected_plan = {"graph": p["world"]["graph"], "menu": p["menu"],
                "budget": p["initial_cost_units"]+p["acquisition_cap_units"], "initial_actions": p["initial_actions"],
                "objective": p["objective"], "domain_notes": p["domain_notes"],
                "policy_command": spec["command"], "policy_timeout_seconds": spec["timeout_seconds"]}
            if canonical(original_plan) != canonical(expected_plan) or receipt["session"]["kind"] != spec["kind"]:
                raise ValueError("arm execution/public policy differs from paired plan")
            if spec["kind"] == "persistent":
                close = persistent_closed[pid]
                if type(close) is not dict or set(close) != {"raw", "sha256"} or type(close["raw"]) is not bytes:
                    raise ValueError("independently pinned persistent close receipt required")
                _write(output/f"service-{index:02d}.closed.json", close["raw"])
                if not _pin(close["sha256"]) or _sha(close["raw"]) != close["sha256"]:
                    raise ValueError("persistent close pin differs")
                closed = json.loads(close["raw"])
                cleanup = closed["cleanup"]
                if (set(closed) != {"status", "cleanup", "completed_requests", "last_request_id", "captured_bytes", "provenance_qualified"}
                        or closed["status"] != "closed" or cleanup["group_still_exists"] is not False
                        or type(cleanup["leader_returncode"]) is not int or cleanup["leader_returncode"] != 0
                        or type(cleanup["errors"]) is not list or cleanup["errors"]
                        or type(cleanup["pgid"]) is not int or cleanup["pgid"] <= 0
                        or cleanup["pgid"] != receipt["session"]["pid"]
                        or not integer(closed["completed_requests"])
                        or closed["completed_requests"] < receipt["session"]["count_after"]
                        or type(closed["last_request_id"]) is not int
                        or closed["last_request_id"] < receipt["session"]["request_after"]
                        or receipt["session"]["identity_sha256"] != spec["identity_sha256"]):
                    raise ValueError("persistent worker did not close cleanly with its expected identity")
            valid[pid] = (receipt, inputs)
            candidate = _initial_candidates(inputs, len(p["initial_actions"]))
            if candidate is not None:
                result = seal_episode(inputs=inputs, expected_sha256=receipt["input_sha256"],
                    shared_initial_receipt_sha256=candidate, scope=world.scope, world_id=world.world_id,
                    policy_id=pid, acquisition_cap_units=p["acquisition_cap_units"])
                candidates[pid], bundles[pid] = candidate, result
        except Exception as exc:
            errors[pid] = type(exc).__name__+": "+str(exc)
            valid.pop(pid, None)
    witnesses = set(candidates.values())
    common = next(iter(witnesses)) if len(witnesses) == 1 else None
    if common is None:
        why = "conflicting authenticated shared initial histories" if witnesses else "no complete authenticated shared initial history"
        errors.update({pid: why for pid in specs if pid not in errors})
    cells, pins = [], []
    for index, pid in enumerate(specs):
        report = None
        if pid not in errors:
            receipt, inputs = valid[pid]
            try:
                sealed = bundles.get(pid) or seal_episode(inputs=inputs, expected_sha256=receipt["input_sha256"],
                    shared_initial_receipt_sha256=common, scope=world.scope, world_id=world.world_id,
                    policy_id=pid, acquisition_cap_units=p["acquisition_cap_units"])
                folder = output/f"arm-{index:02d}"
                folder.mkdir()
                for name, raw in sealed["bundle"].items():
                    _write(folder/name, raw)
                _save(folder/"sealing-provenance.json", sealed["provenance"])
                report = _raw(report_cost_curve(sealed["bundle"], expected_sha256=sealed["bundle_sha256"]))
                _write(folder/"cost-report.json", report)
            except Exception as exc:
                errors[pid] = type(exc).__name__+": "+str(exc)
        cells.append({"world_id": world.world_id, "policy_id": pid, "kind": "report" if report is not None else "unresolved",
            "raw_report": report, "reason": errors.get(pid)})
        pins.append({"world_id": world.world_id, "policy_id": pid, "sha256": _sha(report) if report is not None else None})
    shared = _raw({"receipt_sha256": list(common) if common is not None else []})
    roster = {"scope": world.scope, "policy_ids": list(specs), "strata_fields": [], "worlds": [{
        "world_id": world.world_id, "world_sha256": _sha(_raw(p["world"])), "task_sha256": _sha(_raw(p["task"])),
        "learner_sha256": _sha(_raw(p["learner"])), "shared_initial_sha256": _sha(shared),
        "acquisition_cap_units": p["acquisition_cap_units"], "initial_cost_units": p["initial_cost_units"],
        "scale": task.scale, "threshold": task.normalized_mse_tolerance, "strata": {}}]}
    summary = summarize_screen(roster=roster, cells=cells, report_pins=pins)
    _save(output/"report-pins.json", pins)
    _save(output/"sealing-errors.json", errors)
    _save(output/"summary.json", summary)
    _publish(output/"finalization.json", {"version": VERSION, "plan_sha256": plan_sha256,
        "status": "descriptive_roster_complete", "summary_sha256": _sha((output/"summary.json").read_bytes()),
        "report_pins": pins, "unresolved_cells": len(errors), "scientific_provenance_qualified": False,
        "scientific_acceptance": False, "new_environment_responses": 0, "new_policy_updates": 0})
    return summary

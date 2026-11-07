"""Once-only supplied-plan campaign; no generation, scheduler, training or retry.

CPU/GPU are caller-owned allocation labels. Use one run_policy call per policy,
then finalize_campaign with independently retained attempt receipt byte pins.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

from llm_intervention_contract import canonical, digest, finite, integer
import llm_intervention_paired_episode as paired
from llm_intervention_persistent_policy import PersistentPolicy, _identity
from llm_intervention_task import TaskProcessPolicy
from llm_intervention_screen_reporting import summarize_screen

VERSION = "ace-supplied-plan-campaign-v1"


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, names):
    _need(type(value) is dict and set(value) == set(names), "exact campaign fields required")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _error(exc):
    return (type(exc).__name__ + ": " + str(exc))[:240]


def _decode(raw):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            _need(k not in result, "duplicate JSON key")
            result[k] = v
        return result
    result = json.loads(raw, object_pairs_hook=unique)
    canonical(result)  # Reject nonfinite numbers, including overflowing literals.
    return result


def _source_pins(sources):
    return [{"world_id": s["plan"]["world"]["world_id"], "sha256": s["plan_sha256"]} for s in sources]


def prepare_campaign(*, sources, strata_fields, policy_phases, services):
    """Validate already pinned plans and predeclare strata/phase/service settings.

    sources: ordered [{plan, plan_sha256, strata}]. No world is generated here.
    services: exact persistent ID -> {identity,startup_seconds,cleanup_seconds,
    max_requests}; request capacity must cover the full supplied roster.
    """
    _need(type(sources) is list and 1 <= len(sources) <= 64, "bounded complete source-plan roster required")
    _need(type(strata_fields) is list and len(strata_fields) <= 4
          and all(type(f) is str and 0 < len(f) <= 64 for f in strata_fields)
          and len(set(strata_fields)) == len(strata_fields), "unique predeclared strata required")
    scope, ids, first, seen = None, None, None, set()
    for source in sources:
        _keys(source, ("plan", "plan_sha256", "strata"))
        p, world, _, _ = paired._validate(source["plan"], source["plan_sha256"])
        _need(world.world_id not in seen and (scope is None or scope == world.scope), "unique IDs and consistent scope required")
        seen.add(world.world_id)
        scope = world.scope
        _keys(source["strata"], strata_fields)
        _need(all(type(v) is str and 0 < len(v) <= 128 for v in source["strata"].values()), "predeclared string strata required")
        current = [s["policy_id"] for s in p["policy_specs"]]
        if ids is None:
            ids, first = current, p["policy_specs"]
        _need(current == ids, "consistent ordered policy roster required")
        for spec, original in zip(p["policy_specs"], first):
            # TaskProcess commands may include per-world seeds; all other
            # provenance/timeout/type fields retain the same policy meaning.
            keys = set(spec) - ({"command"} if spec["kind"] == "task_process" else set())
            _need(all(spec[k] == original[k] for k in keys), "policy reuse specification differs across worlds")
    _keys(policy_phases, ids)
    _need(all(v in ("cpu", "gpu") for v in policy_phases.values()), "explicit cpu/gpu phase required")
    persistent = [s["policy_id"] for s in first if s["kind"] == "persistent"]
    _keys(services, persistent)
    for spec in first:
        pid = spec["policy_id"]
        if spec["kind"] == "task_process":
            _need(policy_phases[pid] == "cpu", "task_process preparation/execution belongs on CPU")
            continue
        settings = services[pid]
        _keys(settings, ("identity", "startup_seconds", "cleanup_seconds", "max_requests"))
        _need(digest(_identity(settings["identity"])) == spec["identity_sha256"], "service identity differs from source plans")
        for key in ("startup_seconds", "cleanup_seconds"):
            _need(finite(settings[key]) and 0 < settings[key] <= 3600, "bounded startup/cleanup time required")
        bound = sum(s["plan"]["acquisition_cap_units"] // min(a["cost_units"] for a in s["plan"]["menu"]) for s in sources)
        _need(integer(settings["max_requests"], 1) and bound <= settings["max_requests"] <= 100000,
              "service request bound must cover complete roster")
    result = {"version": VERSION, "scope": scope, "sources": sources, "strata_fields": strata_fields,
              "policy_ids": ids, "policy_phases": policy_phases, "services": services,
              "scientific_provenance_qualified": False, "retry_allowed": False}
    _need(len(paired._raw(result)) <= 64*paired.MAX_BYTES, "campaign byte bound exceeded")
    return _decode(paired._raw(result))


def validate_campaign(*, campaign, campaign_sha256, source_plan_pins):
    _need(paired._pin(campaign_sha256) and digest(campaign) == campaign_sha256, "independent campaign pin required")
    _keys(campaign, ("version", "scope", "sources", "strata_fields", "policy_ids", "policy_phases", "services",
                     "scientific_provenance_qualified", "retry_allowed"))
    rebuilt = prepare_campaign(**{k: campaign[k] for k in ("sources", "strata_fields", "policy_phases", "services")})
    _need(canonical(rebuilt) == canonical(campaign), "campaign differs from validated source plans")
    # Exact ordered, independently supplied inventory, not inferred current pins.
    _need(canonical(source_plan_pins) == canonical(_source_pins(rebuilt["sources"])), "independent source-plan pins differ")
    return rebuilt


def initialize_campaign(*, campaign, campaign_sha256, source_plan_pins, output):
    """Claim a NEW exclusive root; output existence forbids restarting it."""
    c = validate_campaign(campaign=campaign, campaign_sha256=campaign_sha256, source_plan_pins=source_plan_pins)
    root = Path(output).absolute()
    root.mkdir(exist_ok=False)
    paired._save(root/"campaign.json", c)
    paired._save(root/"source-plan-pins.json", source_plan_pins)
    paired._save(root/"start.json", {"utc": _utc(), "campaign_sha256": campaign_sha256, "retry_allowed": False})
    return root


def _root(campaign, campaign_sha256, source_plan_pins, output):
    c = validate_campaign(campaign=campaign, campaign_sha256=campaign_sha256, source_plan_pins=source_plan_pins)
    root = Path(output).absolute()
    _need((root/"campaign.json").read_bytes() == paired._raw(c)
          and (root/"source-plan-pins.json").read_bytes() == paired._raw(source_plan_pins), "campaign root pins differ")
    return c, root


def _attempt(root, c, pid):
    _need(pid in c["policy_ids"], "policy outside campaign roster")
    return root/f"policy-{c['policy_ids'].index(pid):02d}"


def _inventory(path):
    files = sorted(q for q in path.rglob("*") if q.is_file())
    _need(not any(q.is_symlink() for q in path.rglob("*")), "symlink in campaign evidence")
    return {str(q.relative_to(path)): paired._sha(q.read_bytes()) for q in files
            if q.name not in ("attempt.json", "attempt.candidate.json") or q.parent != path}


def _service_evidence(path, attempted):
    if not attempted:
        return {"status": "not_started", "file": None, "sha256": None}
    for name in ("closed.json", "failure.json"):
        file = path/"service"/name
        if file.is_file():
            raw = file.read_bytes()
            return {"status": "recorded", "file": "service/"+name, "sha256": paired._sha(raw)}
    return {"status": "unconfirmed", "file": None, "sha256": None}


def _closure(path, record):
    """Termination proof is distinct from clean success; failure stays a failure."""
    _keys(record, ("status", "file", "sha256"))
    if record["status"] == "not_started":
        _need(record["file"] is None and record["sha256"] is None and not (path/"service").exists(),
              "unstarted service has launch evidence")
        return None
    _need(record["status"] == "recorded" and record["file"] in ("service/closed.json", "service/failure.json"),
          "persistent closure unconfirmed")
    raw = (path/record["file"]).read_bytes()
    _need(paired._pin(record["sha256"]) and paired._sha(raw) == record["sha256"], "service closure pin differs")
    closed = _decode(raw)
    cleanup = closed.get("cleanup")
    if cleanup is None:
        _need(record["file"] == "service/failure.json" and closed.get("status") == "failed"
              and closed.get("stage") == "launch" and not (path/"service/process.json").exists(),
              "missing service process cleanup")
    else:
        _need(type(cleanup) is dict and cleanup.get("group_still_exists") is False
              and type(cleanup.get("leader_returncode")) is int and cleanup.get("errors") == []
              and integer(cleanup.get("pgid"), 1), "persistent closure unconfirmed")
        process_file = path/"service/process.json"
        if process_file.exists():
            _need(_decode(process_file.read_bytes()) == {"pid": cleanup["pgid"], "pgid": cleanup["pgid"]},
                  "closure process identity differs")
    return {"raw": raw, "sha256": record["sha256"]}


def run_policy(*, campaign, campaign_sha256, source_plan_pins, output, policy_id, phase,
               unstarted_reason=None):
    """Run ONE policy over ALL worlds, or explicitly retain it wholly unstarted.

    PersistentPolicy is created here once, borrowed by each arm, and closed here
    once (its transport may self-close on failure). No external live worker is
    accepted. An interrupted call persists its inventory then propagates.
    """
    c, root = _root(campaign, campaign_sha256, source_plan_pins, output)
    path = _attempt(root, c, policy_id)
    _need(phase == c["policy_phases"][policy_id], "policy phase differs")
    _need(not (root/"finalization-intent.json").exists(), "campaign finalization already claimed")
    _need(unstarted_reason is None or paired._name(unstarted_reason), "explicit bounded unstarted reason required")
    if phase == "gpu":
        for pid in c["policy_ids"]:
            if c["policy_phases"][pid] == "cpu":
                prior = _attempt(root, c, pid)
                # This local phase barrier is not independent custody admission.
                _load_attempt(c, campaign_sha256, prior, pid, paired._sha((prior/"attempt.json").read_bytes()))
    path.mkdir(exist_ok=False)
    specs = [next(s for s in src["plan"]["policy_specs"] if s["policy_id"] == policy_id) for src in c["sources"]]
    arms = [{"world_id": src["plan"]["world"]["world_id"], "status": "unstarted", "path": None,
             "receipt_sha256": None, "reason": unstarted_reason or "campaign attempt did not reach this world"} for src in c["sources"]]
    service, attempted, interruption = None, False, None
    errors, started = [], time.monotonic()
    journal = (path/"attempt.jsonl").open("xb")

    def event(kind, **fields):
        journal.write(paired._raw({"utc": _utc(), "elapsed_seconds": time.monotonic()-started, "event": kind, **fields}))
        journal.flush()
        os.fsync(journal.fileno())

    try:
        event("attempt_start", campaign_sha256=campaign_sha256, policy_id=policy_id, phase=phase, arms=arms)
        if unstarted_reason is None:
            if specs[0]["kind"] == "persistent":
                event("service_start_intent")
                attempted = True
                settings = c["services"][policy_id]
                service = PersistentPolicy(tuple(specs[0]["command"]), specs[0]["timeout_seconds"],
                    expected_identity=settings["identity"], output=path/"service",
                    **{k: settings[k] for k in ("startup_seconds", "cleanup_seconds", "max_requests")})
                event("service_ready", pid=service.process.pid)
            for i, (source, spec) in enumerate(zip(c["sources"], specs)):
                if service is not None and service.closed:
                    for arm in arms[i:]:
                        arm["reason"] = "persistent session closed; no restart"
                    break
                arm_path = path/f"world-{i:02d}"
                arm = arms[i]
                arm.update(status="failed", reason="arm execution did not produce a receipt")
                event("arm_start_intent", world_id=arm["world_id"], path=arm_path.name)
                try:
                    p = source["plan"]
                    policy = service
                    if policy is None:
                        _, _, task, _ = paired._validate(p, source["plan_sha256"])
                        policy = TaskProcessPolicy(tuple(spec["command"]), spec["timeout_seconds"], task)
                    receipt = paired.run_world_arm(plan=p, plan_sha256=source["plan_sha256"],
                        policy_id=policy_id, policy=policy, output=arm_path)
                    diagnostics = _decode((arm_path/"execution-diagnostics.json").read_bytes())
                    clean = diagnostics["episode_returned_cleanly"] and diagnostics["adapters_closed_cleanly"]
                    arm["status"] = "completed" if clean and receipt["status"] == "raw_ready" else "failed"
                    arm["reason"] = None if arm["status"] == "completed" else "retained arm execution failure"
                except BaseException as exc:
                    arm["reason"] = _error(exc)
                    errors.append({"world_id": arm["world_id"], "error": _error(exc), "traceback": traceback.format_exc()})
                    if not isinstance(exc, Exception):
                        interruption = exc
                finally:
                    if (arm_path/"arm-receipt.json").is_file():
                        arm.update(path=arm_path.name, receipt_sha256=paired._sha((arm_path/"arm-receipt.json").read_bytes()))
                event("arm_end", arm=arm)
                if interruption is not None:
                    break
    except BaseException as exc:
        errors.append({"phase": "controller", "error": _error(exc), "traceback": traceback.format_exc()})
        if not isinstance(exc, Exception):
            interruption = exc
    finally:
        # Cleanup precedes further journal writes, including when fsync failed.
        if service is not None and not service.closed:
            try:
                service.close()
            except BaseException as exc:
                errors.append({"phase": "service_close", "error": _error(exc)})
                if not isinstance(exc, Exception):
                    interruption = exc
        try:
            closure = _service_evidence(path, attempted)
            event("attempt_end", arms=arms, service=closure, errors=errors)
        finally:
            journal.close()
    receipt = {"version": VERSION, "campaign_sha256": campaign_sha256, "source_plan_pins": source_plan_pins,
               "policy_id": policy_id, "phase": phase, "arms": arms, "service": closure, "errors": errors,
               "evidence_sha256": _inventory(path), "retry_allowed": False, "private_scores_computed": False}
    paired._publish(path/"attempt.json", receipt)
    result = {"policy_id": policy_id, "sha256": paired._sha((path/"attempt.json").read_bytes())}
    if interruption is not None:
        raise interruption
    return result


def _load_attempt(c, campaign_pin, path, pid, expected):
    raw = (path/"attempt.json").read_bytes()
    _need(paired._pin(expected) and paired._sha(raw) == expected, "independent attempt receipt pin differs")
    r = _decode(raw)
    _keys(r, ("version", "campaign_sha256", "source_plan_pins", "policy_id", "phase", "arms", "service", "errors",
              "evidence_sha256", "retry_allowed", "private_scores_computed"))
    _need(r["version"] == VERSION and r["campaign_sha256"] == campaign_pin and r["policy_id"] == pid
          and r["phase"] == c["policy_phases"][pid] and r["source_plan_pins"] == _source_pins(c["sources"])
          and r["retry_allowed"] is False and r["private_scores_computed"] is False, "attempt identity differs")
    _need((path/"attempt.candidate.json").read_bytes() == raw and _inventory(path) == r["evidence_sha256"], "attempt custody differs")
    _need(type(r["arms"]) is list and len(r["arms"]) == len(c["sources"]), "complete campaign arm inventory required")
    for i, (arm, source) in enumerate(zip(r["arms"], c["sources"])):
        _keys(arm, ("world_id", "status", "path", "receipt_sha256", "reason"))
        _need(type(arm["world_id"]) is int and arm["world_id"] == source["plan"]["world"]["world_id"]
              and arm["status"] in ("completed", "failed", "unstarted"), "campaign arm identity/status differs")
        _need(arm["reason"] is None if arm["status"] == "completed" else paired._name(arm["reason"]), "arm disposition reason required")
        if arm["path"] is None:
            _need(arm["receipt_sha256"] is None and arm["status"] != "completed", "missing arm has no success/pin")
        else:
            _need(arm["status"] != "unstarted" and arm["path"] == f"world-{i:02d}" and paired._pin(arm["receipt_sha256"]), "arm path/pin differs")
            arm_raw = (path/arm["path"]/"arm-receipt.json").read_bytes()
            _need(paired._sha(arm_raw) == arm["receipt_sha256"], "arm receipt pin differs")
            ar = _decode(arm_raw)
            _need(ar["plan_sha256"] == source["plan_sha256"] and ar["policy_id"] == pid, "arm source identity differs")
    closure = _closure(path, r["service"])
    if pid not in c["services"]:
        _need(closure is None, "task_process has unexpected service")
    elif closure is None:
        _need(all(a["status"] == "unstarted" for a in r["arms"]), "unstarted service has attempted arms")
    else:
        launch = _decode((path/"service/launch.json").read_bytes())
        spec = next(s for s in c["sources"][0]["plan"]["policy_specs"] if s["policy_id"] == pid)
        settings = c["services"][pid]
        _need(launch["command"] == spec["command"] and launch["timeout_seconds"] == spec["timeout_seconds"]
              and launch["identity_sha256"] == spec["identity_sha256"] and launch["expected_identity"] == settings["identity"]
              and all(launch[k] == settings[k] for k in ("startup_seconds", "cleanup_seconds", "max_requests")), "service launch differs from campaign")
    return r, closure


def finalize_campaign(*, campaign, campaign_sha256, source_plan_pins, output, attempt_pins):
    """Gate ALL inventories/closures before any private per-world evaluation."""
    c, root = _root(campaign, campaign_sha256, source_plan_pins, output)
    _keys(attempt_pins, c["policy_ids"])
    attempts, closed = {}, {}
    for pid in c["policy_ids"]:
        attempts[pid], close = _load_attempt(c, campaign_sha256, _attempt(root, c, pid), pid, attempt_pins[pid])
        if pid in c["services"]:
            closed[pid] = close
    paired._save(root/"finalization-intent.json", {"utc": _utc(), "campaign_sha256": campaign_sha256, "attempt_pins": attempt_pins})
    private = root/"private"
    private.mkdir(exist_ok=False)
    roster = {"scope": c["scope"], "policy_ids": c["policy_ids"], "strata_fields": c["strata_fields"], "worlds": []}
    cells, pins = [], []
    try:
        for i, source in enumerate(c["sources"]):
            arms = {}
            for pid in c["policy_ids"]:
                arm = attempts[pid]["arms"][i]
                arms[pid] = {"path": str(_attempt(root, c, pid)/arm["path"]) if arm["path"] is not None else None,
                    "receipt_sha256": arm["receipt_sha256"], "missing_reason": arm["reason"] if arm["path"] is None else None}
            folder = private/f"world-{i:02d}"
            summary = paired.finalize_paired_world(plan=source["plan"], plan_sha256=source["plan_sha256"],
                arm_outputs=arms, persistent_closed=closed, output=folder)
            world = dict(summary["roster"]["worlds"][0], strata=source["strata"])
            roster["worlds"].append(world)
            world_pins = _decode((folder/"report-pins.json").read_bytes())
            pins.extend(world_pins)
            for j, (row, pin) in enumerate(zip(summary["rows"], world_pins)):
                # Preserve per-world rejection decisions, not just file presence.
                report = row["cell_status"] == "reported"
                cells.append({"world_id": world["world_id"], "policy_id": row["policy_id"],
                    "kind": "report" if report else "unresolved",
                    "raw_report": (folder/f"arm-{j:02d}/cost-report.json").read_bytes() if report else None,
                    "reason": None if report else row["validation"]["reason"]})
        result = summarize_screen(roster=roster, cells=cells, report_pins=pins)
        paired._save(private/"report-pins.json", pins)
        paired._save(private/"summary.json", result)
        paired._publish(root/"finalization.json", {"version": VERSION, "campaign_sha256": campaign_sha256,
            "attempt_pins": attempt_pins, "summary_sha256": paired._sha((private/"summary.json").read_bytes()),
            "cell_count": result["cell_count"], "completed_utc": _utc(), "scientific_acceptance": False,
            "new_environment_responses": 0, "new_training_updates": 0, "retry_allowed": False})
        return result
    except BaseException as exc:
        paired._save(private/"failure.json", {"utc": _utc(), "error": _error(exc), "traceback": traceback.format_exc(), "retry_allowed": False})
        raise

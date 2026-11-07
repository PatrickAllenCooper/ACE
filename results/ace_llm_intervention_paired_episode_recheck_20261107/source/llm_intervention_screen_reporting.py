"""Pure roster-level descriptive summaries of independently pinned reports.

Consumes saved cost-curve JSON only. Does not import/run its reporter, replay a
learner, evaluate a world or infer scientific/receipt/source provenance. Invalid
report cells become explicit unresolved cap rows; invalid rosters reject.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import combinations
import json
import math
import re


VERSION = "ace-descriptive-screen-report-v1"
REPORT_VERSION = "ace-offline-cost-curve-v1"
MAX_WORLDS, MAX_POLICIES, MAX_REPORT_BYTES = 64, 8, 1048576
BASE_FILES = {"world.json", "task.json", "learner.json", "plan.json", "shared-initial.json", "history.json", "terminal.json"}
OUTCOMES = ("accepted", "returned_invalid", "returned_unaccepted", "unknown")
PIN_FIELDS = {"world_sha256": "world.json", "task_sha256": "task.json",
              "learner_sha256": "learner.json", "shared_initial_sha256": "shared-initial.json"}
REPORT_FIELDS = set("""version scope world_id policy_id input_sha256 report_status operational_status
termination_reason initialization_complete curve_complete curve final_error threshold scale query_weighting
attained final_attained attainment_scope first_crossing_cost_units acquisition_cap_units capped_cost_units
capped_cost_reason charges evaluation_error private_evaluator_only private_scores_authorize_policy_stop
new_environment_responses policy_updates source_qualification split_qualification execution_provenance_qualification
scientific_provenance_qualified complete_case_filtering_allowed""".split())


class RosterError(ValueError):
    """The declared cross-product or required cell/pin inventory is malformed."""


class _InvalidReport(ValueError):
    pass


def _need(condition, message, error=_InvalidReport):
    if not condition:
        raise error(message)


def _keys(value, fields, error=_InvalidReport):
    _need(type(value) is dict and set(value) == set(fields), "unexpected object fields", error)


def _int(x, minimum=0, maximum=1000000):
    return type(x) is int and minimum <= x <= maximum


def _number(x, minimum=0.):
    try:
        return type(x) in (int, float) and math.isfinite(x) and x >= minimum
    except OverflowError:
        return False


def _text(x, limit=256):
    return type(x) is str and 0 < len(x) <= limit


def _sha(x):
    return type(x) is str and re.fullmatch("[0-9a-f]{64}", x) is not None


def _identity(entry):
    return entry["world_id"], entry["policy_id"]


def _roster(roster, cells, report_pins):
    _keys(roster, ("scope", "worlds", "policy_ids", "strata_fields"), RosterError)
    _need(roster["scope"] in ("artificial", "development"), "only artificial/development rosters allowed", RosterError)
    worlds, policies, fields = roster["worlds"], roster["policy_ids"], roster["strata_fields"]
    _need(type(worlds) is list and 1 <= len(worlds) <= MAX_WORLDS
          and type(policies) is list and 1 <= len(policies) <= MAX_POLICIES
          and all(_text(p) for p in policies) and len(set(policies)) == len(policies), "bounded unique world/policy roster required", RosterError)
    _need(type(fields) is list and len(fields) <= 4 and all(_text(f, 64) for f in fields)
          and len(set(fields)) == len(fields), "bounded predeclared stratum fields required", RosterError)
    seen = set()
    for w in worlds:
        _keys(w, (*PIN_FIELDS, "world_id", "acquisition_cap_units", "scale", "threshold", "initial_cost_units", "strata"), RosterError)
        _need(_int(w["world_id"], maximum=2**63-1) and w["world_id"] not in seen
              and all(_sha(w[k]) for k in PIN_FIELDS) and _int(w["acquisition_cap_units"], 1)
              and _int(w["initial_cost_units"]) and w["initial_cost_units"]+w["acquisition_cap_units"] <= 1000000
              and _number(w["scale"]) and w["scale"] > 0 and _number(w["threshold"]) and w["threshold"] > 0,
              "invalid expected world inputs or endpoint law", RosterError)
        _keys(w["strata"], fields, RosterError)
        _need(all(_text(v, 128) for v in w["strata"].values()), "predeclared string strata required", RosterError)
        seen.add(w["world_id"])
    expected = {(w["world_id"], p) for w in worlds for p in policies}
    indices = []
    for values, keys in ((cells, ("world_id", "policy_id", "kind", "raw_report", "reason")),
                         (report_pins, ("world_id", "policy_id", "sha256"))):
        _need(type(values) is list and len(values) == len(expected), "every roster cell and pin entry must be explicit", RosterError)
        index = {}
        for entry in values:
            _keys(entry, keys, RosterError)
            _need(_int(entry["world_id"], maximum=2**63-1) and _text(entry["policy_id"]), "typed roster identity required", RosterError)
            key = _identity(entry)
            _need(key in expected and key not in index, "extra or duplicate roster identity", RosterError)
            index[key] = entry
        _need(set(index) == expected, "omitted roster identity", RosterError)
        indices.append(index)
    for cell in cells:
        _need(cell["kind"] in ("report", "unresolved"), "explicit report-or-unresolved cell required", RosterError)
        _need(cell["reason"] is None if cell["kind"] == "report" else
              cell["raw_report"] is None and _text(cell["reason"], 2048), "invalid report/unresolved declaration", RosterError)
    return indices


def _decode(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            _need(key not in obj, "duplicate JSON field")
            obj[key] = value
        return obj
    def constant(value):
        raise _InvalidReport("nonfinite JSON constant: "+value)
    result = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=constant)
    json.dumps(result, allow_nan=False)  # rejects overflowing literals, e.g. 1e999
    return result


def _charges(r, w):
    c = r["charges"]
    scalar = ("actual_total_cost_units", "actual_initial_cost_units", "actual_acquisition_cost_units",
              "reserved_attempts", "known_returns", "accepted_responses", "unknown_returns", "known_return_cost_units")
    _keys(c, ("by_outcome", "by_phase", *scalar))
    _keys(c["by_outcome"], OUTCOMES)
    _keys(c["by_phase"], ("initial", "acquisition"))
    _need(all(_int(c[k]) for k in scalar), "typed nonnegative charge counters required")
    for counts in (*c["by_outcome"].values(), *c["by_phase"].values()):
        _keys(counts, ("attempts", "cost_units"))
        _need(_int(counts["attempts"], maximum=80) and _int(counts["cost_units"])
              and counts["cost_units"] >= counts["attempts"]
              and (counts["cost_units"] == 0) == (counts["attempts"] == 0), "positive per-attempt charge accounting required")
    outcomes, phases = c["by_outcome"], c["by_phase"]
    _need(c["reserved_attempts"] <= 80
          and sum(o["attempts"] for o in outcomes.values()) == sum(p["attempts"] for p in phases.values()) == c["reserved_attempts"]
          and sum(o["cost_units"] for o in outcomes.values()) == sum(p["cost_units"] for p in phases.values()) == c["actual_total_cost_units"]
          and phases["initial"]["cost_units"] == c["actual_initial_cost_units"] <= w["initial_cost_units"]
          and phases["acquisition"]["cost_units"] == c["actual_acquisition_cost_units"] <= w["acquisition_cap_units"]
          and outcomes["accepted"]["attempts"] == c["accepted_responses"]
          and outcomes["unknown"]["attempts"] == c["unknown_returns"]
          and c["known_returns"] == c["reserved_attempts"]-c["unknown_returns"]
          and c["known_return_cost_units"] == c["actual_total_cost_units"]-outcomes["unknown"]["cost_units"],
          "charge/outcome/phase counters disagree")
    rejected = c["reserved_attempts"]-c["accepted_responses"]
    _need(rejected <= 1, "cost-curve histories stop at the first unaccepted response")
    if r["operational_status"] == "complete":
        _need(rejected == 0 and r["initialization_complete"], "complete operational status has failed/incomplete responses")
    if r["initialization_complete"]:
        _need(c["actual_initial_cost_units"] == w["initial_cost_units"]
              and phases["initial"]["attempts"] <= c["accepted_responses"], "shared initial cost/receipt count differs")
    else:
        _need(w["initial_cost_units"] > 0 and r["operational_status"] == "failed"
              and phases["acquisition"] == {"attempts": 0, "cost_units": 0}
              and (rejected > 0 or c["actual_initial_cost_units"] < w["initial_cost_units"]), "invalid partial initialization")
    expected_files = BASE_FILES | {f"receipt-{i:03d}.json" for i in range(c["accepted_responses"])}
    if outcomes["returned_invalid"]["attempts"] + outcomes["returned_unaccepted"]["attempts"]:
        expected_files.add(f"return-{c['accepted_responses']:03d}.bin")
    _need(set(r["input_sha256"]) == expected_files, "input receipt/return pin inventory disagrees with counters")


def _report(r, scope, w, policy):
    _keys(r, REPORT_FIELDS)
    _need(r["version"] == REPORT_VERSION and r["scope"] == scope
          and _int(r["world_id"], maximum=2**63-1) and r["world_id"] == w["world_id"]
          and type(r["policy_id"]) is str and r["policy_id"] == policy, "report identity/version differs")
    pins = r["input_sha256"]
    _need(type(pins) is dict and all(type(k) is str and _sha(v) for k, v in pins.items()), "typed input pins required")
    _need(all(pins.get(filename) == w[field] for field, filename in PIN_FIELDS.items()), "paired world/task/learner/shared-initial pins differ")
    _need(_int(r["acquisition_cap_units"], 1) and r["acquisition_cap_units"] == w["acquisition_cap_units"]
          and _number(r["scale"]) and r["scale"] == w["scale"]
          and _number(r["threshold"]) and r["threshold"] == w["threshold"], "paired cap/scale/threshold differs")
    for k in ("initialization_complete", "curve_complete", "attained", "final_attained",
              "private_evaluator_only", "private_scores_authorize_policy_stop", "scientific_provenance_qualified", "complete_case_filtering_allowed"):
        _need(type(r[k]) is bool, "boolean report flags required")
    _need(r["private_evaluator_only"] and not r["private_scores_authorize_policy_stop"]
          and not r["scientific_provenance_qualified"] and not r["complete_case_filtering_allowed"]
          and r["query_weighting"] == "equal_fixed_queries" and r["attainment_scope"] == "observed_accepted_replay_prefix"
          and all(r[k] == "pending_caller" for k in ("source_qualification", "split_qualification", "execution_provenance_qualification"))
          and all(_int(r[k], maximum=0) for k in ("new_environment_responses", "policy_updates")), "report qualification/scope flags differ")
    _need(r["operational_status"] in ("complete", "failed") and _text(r["termination_reason"], 1024)
          and (r["termination_reason"] in ("policy_stop", "budget_exhausted")) == (r["operational_status"] == "complete"), "inconsistent operational status/reason")
    err = r["evaluation_error"]
    if err is not None:
        _keys(err, ("exception_type", "message"))
        _need(_text(err["exception_type"]) and type(err["message"]) is str and len(err["message"]) <= 4096, "invalid evaluation failure description")
    _need(r["report_status"] == ("reported" if err is None else "evaluation_failed")
          and r["curve_complete"] == (r["initialization_complete"] and err is None), "inconsistent evaluation/curve status")
    _charges(r, w)
    c, curve = r["charges"], r["curve"]
    _need(type(curve) is list and len(curve) <= 81, "bounded curve array required")
    initial_count = c["by_phase"]["initial"]["attempts"]
    full_count = 1+c["accepted_responses"]-initial_count if r["initialization_complete"] else 0
    _need(len(curve) <= full_count and (not r["curve_complete"] or len(curve) == full_count), "curve length contradicts accepted counters")
    accepted_acq_cost = c["actual_acquisition_cost_units"]-sum(c["by_outcome"][o]["cost_units"] for o in OUTCOMES[1:]) if r["initialization_complete"] else 0
    accepted_acq_count = full_count-1 if r["initialization_complete"] else 0
    _need(accepted_acq_cost >= accepted_acq_count and (accepted_acq_cost == 0) == (accepted_acq_count == 0),
          "accepted acquisition count/cost disagrees with phase accounting")
    prior_cost, width = -1, None
    for i, p in enumerate(curve):
        _keys(p, ("kind", "accepted_receipts", "request_id", "action_id", "acquisition_cost_units", "predictions", "normalized_mse"))
        _need(p["kind"] == ("shared_initial" if i == 0 else "accepted_acquisition")
              and _int(p["accepted_receipts"], maximum=80) and p["accepted_receipts"] == initial_count+i
              and _int(p["acquisition_cost_units"]) and prior_cost < p["acquisition_cost_units"] <= accepted_acq_cost
              and _number(p["normalized_mse"]), "curve point identity/order/cost/error differs")
        if i == 0:
            _need(p["request_id"] is None and p["action_id"] is None and p["acquisition_cost_units"] == 0, "shared initial point differs")
        else:
            _need(_int(p["request_id"], maximum=79) and p["request_id"] == initial_count+i-1
                  and _text(p["action_id"]), "acquisition point receipt order differs")
        values = p["predictions"]
        _need(type(values) is list and 1 <= len(values) <= 64
              and all(_number(abs(x)) if type(x) in (int, float) else False for x in values)
              and (width is None or width == len(values)), "finite fixed-width prediction vectors required")
        prior_cost, width = p["acquisition_cost_units"], len(values)
        _need(accepted_acq_cost-prior_cost >= full_count-i-1, "curve prefix leaves insufficient cost for remaining accepted responses")
    if r["curve_complete"]:
        _need(curve[-1]["acquisition_cost_units"] == accepted_acq_cost, "complete curve omits accepted cost")
    crossing = next((p["acquisition_cost_units"] for p in curve if p["normalized_mse"] <= w["threshold"]), None)
    final = curve[-1]["normalized_mse"] if curve and err is None else None
    _need((r["final_error"] is None if final is None else _number(r["final_error"]) and r["final_error"] == final)
          and r["attained"] == (crossing is not None)
          and (r["first_crossing_cost_units"] is None if crossing is None else _int(r["first_crossing_cost_units"]) and r["first_crossing_cost_units"] == crossing)
          and r["final_attained"] == (final is not None and final <= w["threshold"]), "final/attainment endpoints contradict curve")
    failed = r["operational_status"] == "failed" or err is not None
    capped = w["acquisition_cap_units"] if failed or crossing is None else crossing
    reason = ("operational_failure" if r["operational_status"] == "failed" else "evaluation_failure" if err is not None else
              "nonattainment" if crossing is None else "first_crossing")
    _need(_int(r["capped_cost_units"]) and r["capped_cost_units"] == capped and r["capped_cost_reason"] == reason,
          "failed/nonattained row must use the common cap; attained success uses first crossing")
    return failed


def _row(scope, w, policy, cell, expected_pin):
    raw = cell["raw_report"]
    row = {"scope": scope, "world_id": w["world_id"], "policy_id": policy, "strata": deepcopy(w["strata"]),
        "expected_inputs": {k: w[k] for k in (*PIN_FIELDS, "acquisition_cap_units", "initial_cost_units", "scale", "threshold")},
        "expected_report_sha256": expected_pin if _sha(expected_pin) else None,
        "observed_report_sha256": None, "report_bytes": len(raw) if type(raw) is bytes else None,
        "cell_status": "unresolved", "failed": None, "failed_or_unresolved": True,
        "attained": None, "final_attained": None, "first_crossing_acquisition_cost_units": None,
        "first_crossing_total_cost_units": None, "final_error": None, "charges": None,
        "operational_status": None, "termination_reason": None, "report_status": None,
        "capped_acquisition_cost_units": w["acquisition_cap_units"],
        "declared_initial_cost_units": w["initial_cost_units"],
        "capped_total_cost_units": w["initial_cost_units"]+w["acquisition_cap_units"],
        "capped_cost_reason": "unresolved", "paired_input_evidence": None, "validation": None}
    if cell["kind"] == "unresolved":
        row["validation"] = {"status": "caller_declared_unresolved", "stage": None, "reason": cell["reason"]}
        return row
    stage = "byte_pin"
    try:
        _need(type(raw) is bytes and len(raw) <= MAX_REPORT_BYTES, "bounded raw report bytes required")
        row["observed_report_sha256"] = hashlib.sha256(raw).hexdigest()
        _need(_sha(expected_pin) and row["observed_report_sha256"] == expected_pin, "independent report byte pin missing/invalid/mismatched")
        stage = "json_decode"
        report = _decode(raw)
        stage = "report_consistency"
        failed = _report(report, scope, w, policy)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        row["validation"] = {"status": "rejected_to_unresolved", "stage": stage,
            "reason": str(exc)[:2048], "exception_type": type(exc).__name__}
        return row
    crossing = report["first_crossing_cost_units"]
    initial_attempts = report["charges"]["by_phase"]["initial"]["attempts"]
    initial_accepted = min(initial_attempts, report["charges"]["accepted_responses"])
    row.update(cell_status="reported", failed=failed, failed_or_unresolved=failed,
        paired_input_evidence={"initialization_complete": report["initialization_complete"],
            "initial_attempts": initial_attempts,
            "accepted_initial_pins": [report["input_sha256"][f"receipt-{i:03d}.json"] for i in range(initial_accepted)],
            "shared_initial_point": deepcopy(report["curve"][0]) if report["curve"] else None},
        attained=report["attained"], final_attained=report["final_attained"], final_error=report["final_error"],
        first_crossing_acquisition_cost_units=crossing,
        first_crossing_total_cost_units=None if crossing is None else w["initial_cost_units"]+crossing,
        charges=deepcopy(report["charges"]), operational_status=report["operational_status"],
        termination_reason=report["termination_reason"], report_status=report["report_status"],
        capped_acquisition_cost_units=report["capped_cost_units"],
        capped_total_cost_units=w["initial_cost_units"]+report["capped_cost_units"],
        capped_cost_reason=report["capped_cost_reason"], validation={"status": "consistent_report", "stage": "report_consistency", "reason": None})
    return row


def _paired(rows, roster):
    """Conflicting claims have no privileged arm: retain both as unresolved."""
    conflicts = {}
    for w in roster["worlds"]:
        members = [r for r in rows if r["world_id"] == w["world_id"] and r["cell_status"] == "reported"]
        for a, b in combinations(members, 2):
            x, y = a["paired_input_evidence"], b["paired_input_evidence"]
            n = min(len(x["accepted_initial_pins"]), len(y["accepted_initial_pins"]))
            conflict = x["accepted_initial_pins"][:n] != y["accepted_initial_pins"][:n]
            for full, other in ((x, y), (y, x)):
                if full["initialization_complete"]:
                    conflict |= (other["initial_attempts"] > full["initial_attempts"]
                        or (other["initialization_complete"] and other["initial_attempts"] != full["initial_attempts"])
                        or (not other["initialization_complete"] and len(other["accepted_initial_pins"]) >= full["initial_attempts"]))
            if x["shared_initial_point"] is not None and y["shared_initial_point"] is not None:
                conflict |= x["shared_initial_point"] != y["shared_initial_point"]
            if conflict:
                for row, other in ((a, b), (b, a)):
                    conflicts.setdefault(_identity(row), []).append(other["policy_id"])
    for i, row in enumerate(rows):
        if _identity(row) in conflicts:
            w = next(w for w in roster["worlds"] if w["world_id"] == row["world_id"])
            unresolved = _row(roster["scope"], w, row["policy_id"],
                {"kind": "unresolved", "raw_report": None, "reason": "conflicting shared-initial claims across paired reports"}, row["expected_report_sha256"])
            for key in ("observed_report_sha256", "report_bytes", "paired_input_evidence"):
                unresolved[key] = row[key]
            unresolved["validation"].update(status="rejected_to_unresolved", stage="paired_consistency",
                conflicting_policy_ids=conflicts[_identity(row)])
            rows[i] = unresolved


def _summaries(rows, worlds, policies):
    by_cell = {(r["world_id"], r["policy_id"]): r for r in rows}
    summaries = []
    for policy in policies:
        cells = [by_cell[w, policy] for w in worlds]
        acq = [r["capped_acquisition_cost_units"] for r in cells]
        total = [r["capped_total_cost_units"] for r in cells]
        errors = [r["final_error"] for r in cells]
        available = sum(v is not None for v in errors)
        logical = {}
        for label in ("initial", "acquisition", "total"):
            values = [None if r["charges"] is None else r["charges"][f"actual_{label}_cost_units"] for r in cells]
            logical[label] = {"vector": values, "available_cells": sum(x is not None for x in values),
                "all_world_sum": sum(values) if all(v is not None for v in values) else None,
                "reported_cells_subtotal": sum(v for v in values if v is not None)}
        summaries.append({"policy_id": policy, "world_ids": list(worlds), "world_count": len(worlds),
            "capped_acquisition_cost_vector": acq, "capped_acquisition_cost_sum": sum(acq), "mean_capped_acquisition_cost": sum(acq)/len(cells),
            "declared_initial_cost_vector": [r["declared_initial_cost_units"] for r in cells],
            "capped_total_cost_vector": total, "capped_total_cost_sum": sum(total), "mean_capped_total_cost": sum(total)/len(cells),
            "attained": sum(r["attained"] is True for r in cells),
            "attained_without_failure": sum(r["attained"] is True and r["failed"] is False for r in cells),
            "attained_on_failed_prefix": sum(r["attained"] is True and r["failed"] is True for r in cells),
            "not_attained": sum(r["attained"] is False for r in cells),
            "attainment_unresolved": sum(r["attained"] is None for r in cells),
            "failed": sum(r["failed"] is True for r in cells), "unresolved": sum(r["cell_status"] == "unresolved" for r in cells),
            "failed_or_unresolved": sum(r["failed_or_unresolved"] for r in cells),
            "final_error_vector": errors, "available_final_errors": available, "unavailable_final_errors": len(cells)-available,
            "all_world_mean_final_error": min(errors)+math.fsum((v-min(errors))/len(cells) for v in errors) if available == len(cells) else None,
            "logical_per_arm_actual_charges": logical})
    pairs = []
    for left, right in combinations(policies, 2):
        differences = [by_cell[w, left]["capped_acquisition_cost_units"]-by_cell[w, right]["capped_acquisition_cost_units"] for w in worlds]
        totals = [by_cell[w, left]["capped_total_cost_units"]-by_cell[w, right]["capped_total_cost_units"] for w in worlds]
        pairs.append({"left": left, "right": right, "world_ids": list(worlds), "difference_definition": "left_minus_right_capped_cost",
            "comparison_scope": "all roster worlds, including failed/unresolved cap endpoints; descriptive, not evidence of success",
            "acquisition_cost_differences": differences, "total_cost_differences": totals,
            "mean_acquisition_cost_difference": sum(differences)/len(worlds), "mean_total_cost_difference": sum(totals)/len(worlds),
            "left_wins_lower_cost": sum(d < 0 for d in differences), "ties": sum(d == 0 for d in differences),
            "left_losses_higher_cost": sum(d > 0 for d in differences),
            "worlds_with_unresolved_cell": sum(any(by_cell[w, p]["cell_status"] == "unresolved" for p in (left, right)) for w in worlds),
            "worlds_with_failed_cell": sum(any(by_cell[w, p]["failed"] is True for p in (left, right)) for w in worlds)})
    return {"policy_summaries": summaries, "policy_pairs": pairs}


def summarize_screen(*, roster, cells, report_pins):
    """All-cell descriptive report. Caller supplies BOTH roster and pin inventory.

    cells: exact dictionaries {world_id, policy_id, kind, raw_report, reason}.
    report_pins: separate exact {world_id, policy_id, sha256} entries, including
    null when unresolved. Roster-level errors raise before reporting; invalid
    report bytes/contents produce explicit common-cap unresolved rows. No files
    are read/written. Original bytes and validation provenance remain caller-owned.
    """
    cell_index, pin_index = _roster(roster, cells, report_pins)
    rows = [_row(roster["scope"], w, p, cell_index[w["world_id"], p], pin_index[w["world_id"], p]["sha256"])
            for w in roster["worlds"] for p in roster["policy_ids"]]
    _paired(rows, roster)
    ids = [w["world_id"] for w in roster["worlds"]]
    result = {"version": VERSION, "scope": roster["scope"], "roster": deepcopy(roster), "rows": rows,
        "cell_count": len(rows), **_summaries(rows, ids, roster["policy_ids"]), "strata": [],
        "analysis_scope": "descriptive equal-world summaries only; all predeclared-roster cells retained",
        "charge_scope": "logical per-arm reservations; physical shared-initial/cache charges cannot be inferred",
        "endpoint_scope": "capped acquisition endpoint plus declared full initial cost; actual partial charges remain separate",
        "validation_scope": "report bytes and internal/paired-input consistency only; original receipts, source and execution evidence caller-owned",
        "scientific_provenance_qualified": False, "scientific_acceptance": False, "scientific_protocol_frozen": False,
        "complete_case_filtering_allowed": False, "new_scores_computed": False}
    for field in roster["strata_fields"]:
        values = list(dict.fromkeys(w["strata"][field] for w in roster["worlds"]))
        for value in values:
            members = [w["world_id"] for w in roster["worlds"] if w["strata"][field] == value]
            result["strata"].append({"field": field, "value": value, "world_ids": members,
                **_summaries(rows, members, roster["policy_ids"])})
    return result

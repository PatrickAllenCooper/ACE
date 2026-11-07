"""Artificial saved-report tests only; no reporter, learner or world execution."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import llm_intervention_screen_reporting as reporting


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def world(world_id, initial, cap, scale, threshold, stratum):
    return {"world_id": world_id, "world_sha256": sha(f"world-{world_id}"),
        "task_sha256": sha(f"task-{world_id}"), "learner_sha256": sha(f"learner-{world_id}"),
        "shared_initial_sha256": sha(f"initial-{world_id}"), "initial_cost_units": initial,
        "acquisition_cap_units": cap, "scale": scale, "threshold": threshold, "strata": {"graph": stratum}}


def report(w, policy, initial_error, steps=(), *, unknown_cost=0, evaluation_failure=False):
    """Hand-authored schema fixture. Errors are stipulated, not evaluated.

    Initial responses have unit costs. steps are (cumulative cost, stipulated
    error); evaluator failure omits acquisition points while retaining charges.
    Receipt hashes are artificial labels, not claims of receipt authentication.
    """
    initial = w["initial_cost_units"]
    acq = steps[-1][0] if steps else 0
    accepted = initial+len(steps)
    outcomes = {k: {"attempts": 0, "cost_units": 0} for k in
                ("accepted", "returned_invalid", "returned_unaccepted", "unknown")}
    outcomes["accepted"] = {"attempts": accepted, "cost_units": initial+acq}
    outcomes["unknown"] = {"attempts": int(unknown_cost > 0), "cost_units": unknown_cost}
    curve = [{"kind": "shared_initial", "accepted_receipts": initial, "request_id": None,
        "action_id": None, "acquisition_cost_units": 0, "predictions": [0., 0.], "normalized_mse": initial_error}]
    if not evaluation_failure:
        for i, (cost, loss) in enumerate(steps):
            curve.append({"kind": "accepted_acquisition", "accepted_receipts": initial+i+1,
                "request_id": initial+i, "action_id": "artificial-action", "acquisition_cost_units": cost,
                "predictions": [float(i), -float(i)], "normalized_mse": loss})
    pins = {name: sha(f"{w['world_id']}/{policy}/{name}") for name in
            ("world.json", "task.json", "learner.json", "shared-initial.json", "plan.json", "history.json", "terminal.json")}
    for field, name in (("world_sha256", "world.json"), ("task_sha256", "task.json"),
                        ("learner_sha256", "learner.json"), ("shared_initial_sha256", "shared-initial.json")):
        pins[name] = w[field]
    for i in range(accepted):
        pins[f"receipt-{i:03d}.json"] = sha(f"{w['world_id']}/{'shared' if i < initial else policy}/{i}")
    crossing = next((p["acquisition_cost_units"] for p in curve if p["normalized_mse"] <= w["threshold"]), None)
    final = None if evaluation_failure else curve[-1]["normalized_mse"]
    return {"version": "ace-offline-cost-curve-v1", "scope": "artificial", "world_id": w["world_id"], "policy_id": policy,
        "input_sha256": pins, "report_status": "evaluation_failed" if evaluation_failure else "reported",
        "operational_status": "failed" if unknown_cost else "complete", "termination_reason": "callback_failure" if unknown_cost else "policy_stop",
        "initialization_complete": True, "curve_complete": not evaluation_failure, "curve": curve, "final_error": final,
        "threshold": w["threshold"], "scale": w["scale"], "query_weighting": "equal_fixed_queries",
        "attained": crossing is not None, "final_attained": final is not None and final <= w["threshold"],
        "attainment_scope": "observed_accepted_replay_prefix", "first_crossing_cost_units": crossing,
        "acquisition_cap_units": w["acquisition_cap_units"],
        "capped_cost_units": w["acquisition_cap_units"] if unknown_cost or evaluation_failure or crossing is None else crossing,
        "capped_cost_reason": "operational_failure" if unknown_cost else "evaluation_failure" if evaluation_failure else "nonattainment" if crossing is None else "first_crossing",
        "charges": {"by_outcome": outcomes, "by_phase": {"initial": {"attempts": initial, "cost_units": initial},
                "acquisition": {"attempts": len(steps)+int(unknown_cost > 0), "cost_units": acq+unknown_cost}},
            "actual_total_cost_units": initial+acq+unknown_cost, "actual_initial_cost_units": initial,
            "actual_acquisition_cost_units": acq+unknown_cost, "reserved_attempts": accepted+int(unknown_cost > 0),
            "known_returns": accepted, "accepted_responses": accepted, "unknown_returns": int(unknown_cost > 0),
            "known_return_cost_units": initial+acq},
        "evaluation_error": {"exception_type": "ArtificialError", "message": "stipulated failed evaluation"} if evaluation_failure else None,
        "private_evaluator_only": True, "private_scores_authorize_policy_stop": False,
        "new_environment_responses": 0, "policy_updates": 0, "source_qualification": "pending_caller",
        "split_qualification": "pending_caller", "execution_provenance_qualification": "pending_caller",
        "scientific_provenance_qualified": False, "complete_case_filtering_allowed": False}


def packet(roster, reports):
    cells, pins = [], []
    for w in roster["worlds"]:
        for p in roster["policy_ids"]:
            r = reports[w["world_id"], p]
            raw = None if r is None else encode(r)
            cells.append({"world_id": w["world_id"], "policy_id": p, "kind": "unresolved" if r is None else "report",
                          "raw_report": raw, "reason": "missing output declared before aggregation" if r is None else None})
            pins.append({"world_id": w["world_id"], "policy_id": p, "sha256": None if raw is None else sha(raw)})
    return {"roster": roster, "cells": cells, "report_pins": pins}


def oracle_packet():
    a, b, c = world(10, 2, 6, 2., .5, "g1"), world(20, 0, 4, 1., .5, "g1"), world(30, 1, 8, 3., .25, "g2")
    roster = {"scope": "artificial", "worlds": [a, b, c], "policy_ids": ["A", "B", "C"], "strata_fields": ["graph"]}
    reports = {(10, "A"): report(a, "A", 1., [(1, .5), (3, .8)]), (10, "B"): report(a, "B", 1., [(2, .5)]),
        (10, "C"): report(a, "C", 1., [(1, .25)], unknown_cost=2), (20, "A"): report(b, "A", 0.),
        (20, "B"): None, (20, "C"): report(b, "C", 0., [(1, 1.)]),
        (30, "A"): report(c, "A", 1., [(2, .2)], evaluation_failure=True),
        (30, "B"): report(c, "B", 1., [(8, .25)]), (30, "C"): report(c, "C", 1., [(2, .2)])}
    return packet(roster, reports)


def replace_report(data, index, value, repin=True):
    raw = value if isinstance(value, bytes) else encode(value)
    data["cells"][index]["raw_report"] = raw
    if repin:
        data["report_pins"][index]["sha256"] = sha(raw)


class ReportingTests(unittest.TestCase):
    def retain(self, label, data, result=None, rejection=None):
        root = os.environ.get("ACE_SCREEN_REPORTING_TEST_OUTPUT")
        if root is None:
            return
        path = Path(root)/self._testMethodName/label
        path.mkdir(parents=True, exist_ok=False)
        saved = deepcopy(data)
        for i, cell in enumerate(saved["cells"]):
            raw = cell["raw_report"]
            if type(raw) is bytes:
                name = f"report-{i:03d}.bin"
                (path/name).write_bytes(raw)
                cell["raw_report"] = {"file": name}
        (path/"input.json").write_bytes(encode(saved))
        if result is not None:
            (path/"output.json").write_bytes(encode(result))
        if rejection is not None:
            (path/"rejection.json").write_bytes(encode({"reason": rejection}))

    def check(self, data, label):
        try:
            result = reporting.summarize_screen(**data)
        except Exception as exc:
            self.retain(label, data, rejection=f"unexpected {type(exc).__name__}: {exc}")
            raise
        self.retain(label, data, result)
        return result

    def test_hand_calculated_all_cell_pairing_and_logical_costs(self):
        data = oracle_packet()
        before = deepcopy(data)
        result = self.check(data, "paired-oracle")
        self.assertEqual(data, before)
        self.assertEqual(result["cell_count"], 9)
        self.assertEqual([r["cell_status"] for r in result["rows"]].count("reported"), 8)
        a, b, c = result["policy_summaries"]
        self.assertEqual([s["capped_acquisition_cost_vector"] for s in (a, b, c)], [[1, 0, 8], [2, 4, 8], [6, 0, 2]])
        self.assertEqual([s["capped_total_cost_vector"] for s in (a, b, c)], [[3, 0, 9], [4, 4, 9], [8, 0, 3]])
        self.assertEqual([s["capped_acquisition_cost_sum"] for s in (a, b, c)], [9, 14, 8])
        self.assertEqual([s["capped_total_cost_sum"] for s in (a, b, c)], [12, 17, 11])
        self.assertEqual([s["attained"] for s in (a, b, c)], [2, 2, 3])
        self.assertEqual([s["failed"] for s in (a, b, c)], [1, 0, 1])
        self.assertEqual([s["unresolved"] for s in (a, b, c)], [0, 1, 0])
        self.assertEqual(c["attained_on_failed_prefix"], 1)
        self.assertEqual(c["attained_without_failure"], 2)
        self.assertEqual(a["final_error_vector"], [.8, 0., None])
        self.assertIsNone(a["all_world_mean_final_error"])
        self.assertIsNone(b["all_world_mean_final_error"])
        self.assertAlmostEqual(c["all_world_mean_final_error"], 1.45/3)
        self.assertEqual(a["logical_per_arm_actual_charges"]["total"]["all_world_sum"], 8)
        self.assertIsNone(b["logical_per_arm_actual_charges"]["total"]["all_world_sum"])
        self.assertEqual(b["logical_per_arm_actual_charges"]["total"]["reported_cells_subtotal"], 13)
        self.assertEqual(c["logical_per_arm_actual_charges"]["total"]["all_world_sum"], 9)
        for pair, expected, counts in zip(result["policy_pairs"], ([-1, -4, 0], [-5, 0, 6], [-4, 4, 6]), ((2, 1, 0), (1, 1, 1), (1, 0, 2))):
            self.assertEqual(pair["acquisition_cost_differences"], expected)
            self.assertEqual(pair["total_cost_differences"], expected)
            self.assertEqual(tuple(pair[k] for k in ("left_wins_lower_cost", "ties", "left_losses_higher_cost")), counts)
        self.assertEqual([s["world_ids"] for s in result["strata"]], [[10, 20], [30]])
        self.assertTrue(all(len(s["policy_pairs"]) == 3 for s in result["strata"]))
        self.assertIn("logical per-arm", result["charge_scope"])
        self.assertFalse(result["scientific_acceptance"])
        self.assertFalse(result["new_scores_computed"])

    def test_byte_pins_before_decode_and_reordering(self):
        base = oracle_packet()
        expected = self.check(base, "original")
        reordered = deepcopy(base)
        reordered["cells"].reverse()
        reordered["report_pins"].reverse()
        self.assertEqual(self.check(reordered, "inventory-reordered"), expected)
        for label, value in (("tampered", base["cells"][0]["raw_report"]+b" "), ("malformed", b"{invalid"),
                             ("oversize", b" "*(reporting.MAX_REPORT_BYTES+1))):
            data = deepcopy(base)
            replace_report(data, 0, value, repin=False)
            # Other rows must still decode; only the rejected first byte payload must not.
            original = reporting._decode
            def guard(raw):
                self.assertNotEqual(raw, value, "invalid bytes reached JSON decoding")
                return original(raw)
            with patch.object(reporting, "_decode", side_effect=guard):
                result = self.check(data, label)
            self.assertEqual(result["rows"][0]["validation"]["stage"], "byte_pin")
            self.assertEqual(result["rows"][0]["capped_acquisition_cost_units"], 6)
        data = deepcopy(base)
        r = json.loads(data["cells"][0]["raw_report"])
        replace_report(data, 0, json.dumps(dict(reversed(list(r.items())))).encode())
        self.assertEqual(self.check(data, "json-fields-reordered")["rows"][0]["cell_status"], "reported")
        r["curve"].reverse()
        replace_report(data, 0, r)
        self.assertEqual(self.check(data, "curve-reordered")["rows"][0]["cell_status"], "unresolved")
        data = deepcopy(base)
        replace_report(data, 0, data["cells"][1]["raw_report"])
        self.assertEqual(self.check(data, "wrong-policy")["rows"][0]["cell_status"], "unresolved")

    def test_missing_extra_duplicate_inventory_and_scope_reject(self):
        base = oracle_packet()
        cases = []
        for key in ("cells", "report_pins"):
            data = deepcopy(base); data[key].pop(); cases.append((key+"-missing", data))
            data = deepcopy(base); data[key].append(deepcopy(data[key][0])); cases.append((key+"-extra", data))
            data = deepcopy(base); data[key][1] = deepcopy(data[key][0]); cases.append((key+"-duplicate", data))
            data = deepcopy(base); data[key][0]["world_id"] = 999; cases.append((key+"-identity", data))
        for scope in ("policy_train", "confirmation", "scientific"):
            data = deepcopy(base); data["roster"]["scope"] = scope; cases.append((scope, data))
        for label, mutate in (("world-bound", lambda r: r.update(worlds=r["worlds"]*22)),
                              ("policy-bound", lambda r: r.update(policy_ids=list("ABCDEFGHI"))),
                              ("extra-key", lambda r: r.update(selection="best")),
                              ("missing-stratum", lambda r: r["worlds"][0].update(strata={})),
                              ("nonpositive-scale", lambda r: r["worlds"][0].update(scale=0.)),
                              ("bool-cap", lambda r: r["worlds"][0].update(acquisition_cap_units=True))):
            data = deepcopy(base); mutate(data["roster"]); cases.append((label, data))
        for label, data in cases:
            with self.subTest(label=label):
                with self.assertRaises(reporting.RosterError) as caught:
                    reporting.summarize_screen(**data)
                self.retain(label, data, rejection=str(caught.exception))
        development = deepcopy(base)
        development["roster"]["scope"] = "development"
        for i, cell in enumerate(development["cells"]):
            if cell["kind"] == "report":
                r = json.loads(cell["raw_report"]); r["scope"] = "development"; replace_report(development, i, r)
        self.assertEqual(self.check(development, "development-declaration")["scope"], "development")

    def test_invalid_content_becomes_retained_unresolved_cap(self):
        base = oracle_packet()
        mutations = [("paired-"+name, lambda r, name=name: r["input_sha256"].update({name: "f"*64}))
                     for name in ("world.json", "task.json", "learner.json", "shared-initial.json")]
        mutations += [("scale", lambda r: r.update(scale=3.)), ("threshold", lambda r: r.update(threshold=.2)),
            ("cap", lambda r: r.update(acquisition_cap_units=7)), ("bool-counter", lambda r: r["charges"].update(known_returns=True)),
            ("counter-sum", lambda r: r["charges"].update(actual_total_cost_units=99)),
            ("final-error", lambda r: r.update(final_error=.1)), ("attainment", lambda r: r.update(attained=False)),
            ("first-crossing", lambda r: r.update(first_crossing_cost_units=0)),
            ("complete-curve", lambda r: r["curve"].pop()), ("missing-pin", lambda r: r["input_sha256"].pop("receipt-000.json")),
            ("false-qualification", lambda r: r.update(scientific_provenance_qualified=True)),
            ("prediction-width", lambda r: r["curve"][-1].update(predictions=[0.])),
            ("initial-cost", lambda r: r["charges"]["by_phase"]["initial"].update(cost_units=1))]
        for label, mutate in mutations:
            with self.subTest(label=label):
                data = deepcopy(base); r = json.loads(data["cells"][0]["raw_report"]); mutate(r); replace_report(data, 0, r)
                result = self.check(data, label)
                row = result["rows"][0]
                self.assertEqual(row["cell_status"], "unresolved")
                self.assertEqual(row["capped_acquisition_cost_units"], 6)
                self.assertEqual(row["capped_total_cost_units"], 8)
                self.assertIsNone(row["attained"])
                self.assertIsNone(row["charges"])
                self.assertEqual(result["cell_count"], 9)
                self.assertEqual(len(result["policy_pairs"]), 3)
        for label, raw in (("duplicate-json", b'{"version": 1, "version": 2}'), ("nonfinite-json", b'{"x":NaN}'),
                           ("overflow-json", b'{"x":1e999}')):
            data = deepcopy(base); replace_report(data, 0, raw)
            self.assertEqual(self.check(data, label)["rows"][0]["validation"]["stage"], "json_decode")
        data = deepcopy(base); data["report_pins"][0]["sha256"] = None
        self.assertEqual(self.check(data, "missing-independent-pin")["rows"][0]["validation"]["stage"], "byte_pin")

    def test_shared_initial_conflicts_fail_closed_without_choosing_arm(self):
        for label, mutate in (("receipt", lambda r: r["input_sha256"].update({"receipt-000.json": "e"*64})),
                              ("prediction", lambda r: r["curve"][0].update(predictions=[.1, 0.]))):
            data = oracle_packet(); r = json.loads(data["cells"][0]["raw_report"]); mutate(r); replace_report(data, 0, r)
            result = self.check(data, label)
            self.assertEqual([r["cell_status"] for r in result["rows"][:3]], ["unresolved"]*3)
            self.assertTrue(all(r["validation"]["stage"] == "paired_consistency" for r in result["rows"][:3]))
            self.assertEqual(result["policy_pairs"][0]["worlds_with_unresolved_cell"], 2)
            self.assertEqual(result["cell_count"], 9)

    def test_failed_prefix_partial_initialization_and_zero_ties(self):
        w = world(1, 2, 5, 1., .5, "g")
        roster = {"scope": "artificial", "worlds": [w], "policy_ids": ["A", "B"], "strata_fields": ["graph"]}
        a, b = report(w, "A", 0., [(1, .2)], evaluation_failure=True), report(w, "B", 0.)
        result = self.check(packet(roster, {(1, "A"): a, (1, "B"): b}), "attained-evaluation-failure")
        self.assertEqual([r["attained"] for r in result["rows"]], [True, True])
        self.assertEqual([r["capped_acquisition_cost_units"] for r in result["rows"]], [5, 0])
        a = deepcopy(a)
        a.update(initialization_complete=False, curve_complete=False, curve=[], attained=False, final_attained=False,
                 first_crossing_cost_units=None, evaluation_error=None, report_status="reported",
                 operational_status="failed", termination_reason="initial_callback_failure", capped_cost_reason="operational_failure")
        a["input_sha256"] = {k: v for k, v in a["input_sha256"].items() if not k.startswith("receipt-")}
        a["charges"] = {"by_outcome": {k: {"attempts": int(k == "unknown"), "cost_units": int(k == "unknown")} for k in
                ("accepted", "returned_invalid", "returned_unaccepted", "unknown")},
            "by_phase": {"initial": {"attempts": 1, "cost_units": 1}, "acquisition": {"attempts": 0, "cost_units": 0}},
            "actual_total_cost_units": 1, "actual_initial_cost_units": 1, "actual_acquisition_cost_units": 0,
            "reserved_attempts": 1, "known_returns": 0, "accepted_responses": 0, "unknown_returns": 1, "known_return_cost_units": 0}
        result = self.check(packet(roster, {(1, "A"): a, (1, "B"): b}), "partial-initial")
        self.assertEqual([r["cell_status"] for r in result["rows"]], ["reported"]*2)
        self.assertEqual(result["rows"][0]["charges"]["actual_total_cost_units"], 1)
        self.assertEqual(result["rows"][0]["capped_total_cost_units"], 7)
        self.assertEqual(result["policy_pairs"][0]["total_cost_differences"], [5])
        bad = deepcopy(a); bad.update(capped_cost_units=0, capped_cost_reason="first_crossing")
        self.assertEqual(self.check(packet(roster, {(1, "A"): bad, (1, "B"): b}), "failed-row-cannot-claim-zero")["rows"][0]["cell_status"], "unresolved")
        w = world(2, 0, 5, 1., .5, "g"); roster["worlds"] = [w]
        result = self.check(packet(roster, {(2, p): report(w, p, 0.) for p in ("A", "B")}), "zero-tie")
        self.assertEqual(result["policy_pairs"][0]["ties"], 1)
        self.assertEqual([s["all_world_mean_final_error"] for s in result["policy_summaries"]], [0., 0.])
        self.assertEqual([r["capped_total_cost_units"] for r in result["rows"]], [0, 0])
        result = self.check(packet(roster, {(2, p): None for p in ("A", "B")}), "all-unresolved")
        self.assertEqual(result["policy_pairs"][0]["ties"], 1)
        for s in result["policy_summaries"]:
            self.assertEqual((s["failed"], s["unresolved"], s["attainment_unresolved"]), (0, 1, 1))
            self.assertIsNone(s["all_world_mean_final_error"])
            self.assertIsNone(s["logical_per_arm_actual_charges"]["total"]["all_world_sum"])


if __name__ == "__main__":
    unittest.main()

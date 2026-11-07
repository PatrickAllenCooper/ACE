"""New tiny artificial offline fixtures; no generation or response callbacks."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np

from llm_intervention_contract import Action, Node, canonical
from llm_intervention_cost_curves import CurveInputError, VERSION, report_cost_curve
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_world import LinearMechanism, ResponseAdapter, WorldDescriptor
import llm_intervention_world as world_module


def raw(value):
    return (canonical(value) + "\n").encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def reseal(bundle):
    """Artificial seal construction, not execution/protocol authentication."""
    plan = json.loads(bundle["plan.json"])
    for field, name in (("world_sha256", "world.json"), ("task_sha256", "task.json"),
                        ("learner_sha256", "learner.json"), ("shared_initial_sha256", "shared-initial.json")):
        plan[field] = sha(bundle[name])
    bundle["plan.json"] = raw(plan)
    history = json.loads(bundle["history.json"])
    history["plan_sha256"] = sha(bundle["plan.json"])
    bundle["history.json"] = raw(history)
    terminal = json.loads(bundle["terminal.json"])
    terminal["input_sha256"] = {k: sha(v) for k, v in bundle.items() if k != "terminal.json"}
    bundle["terminal.json"] = raw(terminal)
    return {k: sha(v) for k, v in bundle.items()}


def fixture(*, actions=("cheap", "costly", "cheap", "costly"), values=None,
            initial_count=1, constant=False, threshold=.04, cap=6,
            status="complete", reason="budget_exhausted", outcome=None, outcome_raw=None,
            initial_failure=False):
    graph = (Node("X", (), "u", (-2., 2.)), Node("Y", ("X",), "u"))
    world = WorldDescriptor("artificial", 7, graph, (
        LinearMechanism("X", 0., (), 0.), LinearMechanism("Y", 2., (("X", 0. if constant else 2.),), 0.)))
    menu = (Action("init", "intervene", (("X", 0.),), ("Y",), 1),
            Action("cheap", "intervene", (("X", 0. if constant else 1.),), ("Y",), 1),
            Action("costly", "intervene", (("X", 0. if constant else 1.),), ("Y",), 2))
    queries = (Action("q0", "intervene", (("X", 0.),), ("Y",), 1),
               Action("q1", "intervene", (("X", 1.),), ("Y",), 1))
    by_id = {a.action_id: a for a in menu}
    def receipt(request, aid, value):
        a = by_id[aid]
        return {"request_id": request, "action_id": aid, "kind": a.kind,
                "targets": dict(a.targets), "values": {"Y": value}, "cost_units": a.cost_units}
    initial_receipts = [raw(receipt(i, "init", 2.)) for i in range(initial_count)]
    bundle = {"world.json": raw(world.private_payload()),
        "task.json": raw({"target": "Y", "queries": [asdict(q) for q in queries],
            "weights": [.5, .5], "scale": 2., "normalized_mse_tolerance": threshold}),
        "learner.json": raw({"graph": [asdict(n) for n in graph],
            "noise_variances": {"X": 1., "Y": 1.}, "prior_variance": 1.}),
        "shared-initial.json": raw({"receipt_sha256": [sha(r) for r in initial_receipts]}),
        "plan.json": raw({"version": VERSION, "scope": "artificial", "world_id": 7,
            "policy_id": "analytic-fixture", "world_sha256": "", "task_sha256": "",
            "learner_sha256": "", "shared_initial_sha256": "", "menu": [asdict(a) for a in menu],
            "initial_actions": ["init"]*initial_count, "acquisition_cap_units": cap, "threshold": threshold})}
    attempts = []
    if not initial_failure:
        for i, r in enumerate(initial_receipts):
            bundle[f"receipt-{i:03d}.json"] = r
            attempts.append({"request_id": i, "action_id": "init", "phase": "initial",
                "cost_units": 1, "outcome": "accepted", "evidence_file": f"receipt-{i:03d}.json"})
        for index, aid in enumerate(actions):
            i = len(attempts)
            value = values[index] if values is not None else (2. if constant else 4.)
            bundle[f"receipt-{i:03d}.json"] = raw(receipt(i, aid, value))
            attempts.append({"request_id": i, "action_id": aid, "phase": "acquisition",
                "cost_units": by_id[aid].cost_units, "outcome": "accepted", "evidence_file": f"receipt-{i:03d}.json"})
    if outcome is not None:
        i = len(attempts)
        aid, phase = ("init", "initial") if initial_failure else ("costly", "acquisition")
        evidence = None if outcome == "unknown" else f"return-{i:03d}.bin"
        if evidence is not None:
            bundle[evidence] = outcome_raw
        attempts.append({"request_id": i, "action_id": aid, "phase": phase,
                         "cost_units": by_id[aid].cost_units, "outcome": outcome, "evidence_file": evidence})
    bundle["history.json"] = raw({"version": VERSION, "plan_sha256": "", "attempts": attempts})
    bundle["terminal.json"] = raw({"version": VERSION, "sealed": True, "input_sha256": {},
        "status": status, "reason": reason, "attempt_count": len(attempts),
        "accepted_count": sum(e["outcome"] == "accepted" for e in attempts),
        "known_return_count": sum(e["outcome"] != "unknown" for e in attempts),
        "initial_cost_units": sum(e["cost_units"] for e in attempts if e["phase"] == "initial"),
        "acquisition_cost_units": sum(e["cost_units"] for e in attempts if e["phase"] == "acquisition")})
    return bundle, reseal(bundle)


class CostCurveTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.root/self._testMethodName
        self.directory.mkdir(exist_ok=False)

    def retain(self, name, bundle, pins, *, reject=False):
        directory = self.directory/name
        directory.mkdir(exist_ok=False)
        for filename, value in bundle.items():
            with (directory/filename).open("xb") as f:
                f.write(value)
        (directory/"pins.json").write_bytes(raw(pins))
        before = deepcopy(bundle)
        try:
            result = report_cost_curve(bundle, expected_sha256=pins)
        except CurveInputError as exc:
            (directory/"rejection.json").write_bytes(raw({"exception_type": type(exc).__name__,
                "message": str(exc), "drop_from_roster_allowed": False}))
            if not reject:
                raise
            return None
        finally:
            self.assertEqual(bundle, before)
        (directory/"report.json").write_bytes(raw(result))
        self.assertFalse(reject, "contradictory bundle accepted")
        return result

    def test_analytic_equal_query_curve_and_cap_boundary_attainment(self):
        bundle, pins = fixture()
        report = self.retain("analytic", bundle, pins)
        oracle = [Fraction(5, 4), Fraction(1, 5), Fraction(5, 64), Fraction(5, 121), Fraction(5, 196)]
        for point, expected in zip(report["curve"], oracle):
            self.assertAlmostEqual(point["normalized_mse"], float(expected), places=14)
        self.assertEqual(len(report["curve"]), 5)
        self.assertEqual([p["acquisition_cost_units"] for p in report["curve"]], [0, 1, 3, 4, 6])
        self.assertEqual(report["curve"][0]["predictions"], [1., 1.])
        self.assertEqual((report["attained"], report["first_crossing_cost_units"], report["capped_cost_units"]), (True, 6, 6))
        self.assertAlmostEqual(report["final_error"], 5/196, places=14)
        self.assertEqual(report["charges"]["actual_total_cost_units"], 7)
        self.assertEqual(report["charges"]["accepted_responses"], 5)

    def test_inclusive_threshold_zero_crossing_and_early_stop_nonattainment(self):
        attained = self.retain("initial_crossing", *fixture(actions=(), threshold=1.25, reason="policy_stop"))
        self.assertEqual((attained["attained"], attained["first_crossing_cost_units"], attained["capped_cost_units"]), (True, 0, 0))
        stopped = self.retain("early_stop", *fixture(actions=(), reason="policy_stop"))
        self.assertEqual(stopped["final_error"], 1.25)
        self.assertFalse(stopped["attained"])
        self.assertIsNone(stopped["first_crossing_cost_units"])
        self.assertEqual(stopped["capped_cost_units"], 6)
        self.assertEqual(stopped["charges"]["actual_acquisition_cost_units"], 0)
        self.assertFalse(stopped["private_scores_authorize_policy_stop"])
        empty = self.retain("empty_initial", *fixture(actions=(), initial_count=0, threshold=2.5, reason="policy_stop"))
        self.assertEqual(empty["final_error"], 2.5)
        self.assertEqual(empty["first_crossing_cost_units"], 0)

    def test_first_crossing_survives_regression_but_failure_gets_common_cap(self):
        result = self.retain("regression", *fixture(actions=("cheap",)*4, values=(2., 2., 2., -10.),
            constant=True, cap=5, reason="policy_stop"))
        self.assertEqual(result["first_crossing_cost_units"], 3)
        self.assertAlmostEqual(result["final_error"], 49/36)
        self.assertTrue(result["attained"])
        self.assertFalse(result["final_attained"])
        self.assertEqual(result["capped_cost_units"], 3)
        failed = self.retain("failure_after_crossing", *fixture(actions=("cheap",)*3, constant=True,
            cap=5, status="failed", reason="callback_interrupted", outcome="unknown"))
        self.assertTrue(failed["attained"])
        self.assertEqual(failed["first_crossing_cost_units"], 3)
        self.assertEqual(failed["capped_cost_units"], 5)
        self.assertEqual(failed["charges"]["actual_total_cost_units"], 6)
        self.assertEqual(failed["charges"]["by_outcome"]["unknown"], {"attempts": 1, "cost_units": 2})

    def test_failed_known_invalid_unknown_and_incomplete_initial_costs_survive(self):
        cases = (("returned_invalid", b'{"Y":NaN}\n'), ("returned_unaccepted", b'{"Y":4.0}\n'), ("unknown", None))
        for outcome, returned in cases:
            with self.subTest(outcome=outcome):
                result = self.retain(outcome, *fixture(actions=(), status="failed", reason="response_failure",
                                                      outcome=outcome, outcome_raw=returned))
                self.assertEqual(result["charges"]["actual_total_cost_units"], 3)
                self.assertEqual(result["charges"]["by_outcome"][outcome], {"attempts": 1, "cost_units": 2})
                self.assertEqual(result["charges"]["known_returns"], 1 if outcome == "unknown" else 2)
                self.assertEqual(len(result["curve"]), 1)
                self.assertEqual(result["capped_cost_units"], 6)
        failed = self.retain("incomplete_initial", *fixture(actions=(), initial_count=2, initial_failure=True,
            status="failed", reason="initial_response_unknown", outcome="unknown"))
        self.assertEqual(failed["curve"], [])
        self.assertIsNone(failed["final_error"])
        self.assertFalse(failed["initialization_complete"])
        self.assertEqual(failed["charges"]["actual_initial_cost_units"], 1)
        self.assertEqual(failed["capped_cost_units"], 6)

    def test_partial_reordered_altered_and_contradictory_seals_rejected(self):
        cases = ("altered_bytes", "partial_json", "unsealed", "reordered", "wrong_receipt_id", "wrong_clamp",
                 "dropped_attempt", "missing_receipt", "shared_history", "extra_evidence", "boolean_cost", "duplicate_json")
        for case in cases:
            with self.subTest(case=case):
                bundle, pins = fixture()
                if case == "altered_bytes":
                    bundle["receipt-001.json"] += b" "
                elif case == "partial_json":
                    bundle["terminal.json"] = bundle["terminal.json"][:-4]
                    pins["terminal.json"] = sha(bundle["terminal.json"])
                else:
                    history, terminal = (json.loads(bundle[n]) for n in ("history.json", "terminal.json"))
                    if case == "unsealed": terminal["sealed"] = False
                    if case == "reordered": history["attempts"][1:3] = reversed(history["attempts"][1:3])
                    if case == "dropped_attempt": history["attempts"].pop()
                    if case == "missing_receipt": del bundle["receipt-001.json"]
                    if case == "extra_evidence": bundle["extra.bin"] = b"unreferenced"
                    if case == "boolean_cost": history["attempts"][1]["cost_units"] = True
                    if case in ("wrong_receipt_id", "wrong_clamp"):
                        r = json.loads(bundle["receipt-001.json"])
                        if case == "wrong_receipt_id": r["request_id"] = 7
                        else: r["targets"] = {"X": -1.}
                        bundle["receipt-001.json"] = raw(r)
                    if case == "shared_history": bundle["shared-initial.json"] = raw({"receipt_sha256": ["0"*64]})
                    bundle["history.json"], bundle["terminal.json"] = raw(history), raw(terminal)
                    if case == "duplicate_json":
                        bundle["receipt-001.json"] = bundle["receipt-001.json"].replace(b'{', b'{"request_id":1,', 1)
                    pins = reseal(bundle)
                self.retain(case, bundle, pins, reject=True)

    def test_predeclared_task_configuration_and_terminal_contradictions_rejected(self):
        for case in ("unequal_weights", "threshold_change", "invalid_prior", "scope_change", "cap_overrun",
                     "false_exhaustion", "complete_unknown", "invalid_but_valid_return", "continuation_after_failure"):
            with self.subTest(case=case):
                if case in ("complete_unknown", "invalid_but_valid_return", "continuation_after_failure"):
                    bundle, pins = fixture(actions=(), status="failed", reason="failure", outcome="unknown" if case != "invalid_but_valid_return" else "returned_invalid",
                                           outcome_raw=b'{"Y":4.0}')
                else:
                    bundle, pins = fixture(actions=() if case == "false_exhaustion" else ("costly",) if case == "cap_overrun" else ("cheap",))
                plan, task, config, terminal = (json.loads(bundle[n]) for n in ("plan.json", "task.json", "learner.json", "terminal.json"))
                if case == "unequal_weights": task["weights"] = [.25, .75]
                if case == "threshold_change": plan["threshold"] = .5
                if case == "invalid_prior": config["prior_variance"] = 0.
                if case == "scope_change": plan["scope"] = "development"
                if case == "cap_overrun": plan["acquisition_cap_units"] = 1; terminal["acquisition_cost_units"] = 2
                if case == "complete_unknown": terminal["status"], terminal["reason"] = "complete", "policy_stop"
                if case == "continuation_after_failure":
                    h = json.loads(bundle["history.json"])
                    h["attempts"].append(dict(h["attempts"][-1], request_id=2))
                    bundle["history.json"] = raw(h)
                for name, data in (("plan.json", plan), ("task.json", task), ("learner.json", config), ("terminal.json", terminal)):
                    bundle[name] = raw(data)
                self.retain(case, bundle, reseal(bundle), reject=True)

    def test_offline_purity_and_numerical_failure_retains_charges(self):
        py_state, np_state = random.getstate(), np.random.get_state()
        with patch.object(world_module, "generate_world", side_effect=AssertionError("no generator")), \
             patch.object(ResponseAdapter, "__call__", side_effect=AssertionError("no response")), \
             patch.object(LinearGaussianSCM, "simulate", side_effect=AssertionError("no predictive draw")):
            good = self.retain("pure", *fixture())
            failed = self.retain("numeric_failure", *fixture(actions=("cheap",), values=(1e308,), reason="policy_stop"))
        self.assertEqual(random.getstate(), py_state)
        after = np.random.get_state()
        self.assertEqual(np_state[0], after[0])
        np.testing.assert_array_equal(np_state[1], after[1])
        self.assertEqual(np_state[2:], after[2:])
        self.assertEqual(good["new_environment_responses"], 0)
        self.assertFalse(good["scientific_provenance_qualified"])
        self.assertEqual(failed["report_status"], "evaluation_failed")
        self.assertEqual(failed["charges"]["actual_total_cost_units"], 2)
        self.assertEqual(failed["capped_cost_units"], 6)
        self.assertIsNone(failed["final_error"])
        self.assertFalse(failed["curve_complete"])
        self.assertEqual(len(failed["curve"]), 1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_llm_intervention_cost_curves.py NEW_FIXTURE_DIRECTORY")
    CostCurveTests.root = Path(sys.argv.pop(1))
    CostCurveTests.root.mkdir(exist_ok=False)
    unittest.main(verbosity=2)

"""Artificial arithmetic oracles only: no models, worlds, actions or remote work."""
import copy
from fractions import Fraction
import hashlib
import itertools
import json
import math
import unittest

from llm_intervention_replicated_feedback import aggregate_replicated_feedback


def task_pin(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fixture():
    task = {"target": "Y", "scale": 2., "normalized_mse_tolerance": .04,
        "queries": [{"action_id": "q0", "kind": "intervene", "targets": [["X", 0.]],
                     "measure": ["Y"], "cost_units": 1},
                    {"action_id": "q1", "kind": "intervene", "targets": [["X", 1.]],
                     "measure": ["Y"], "cost_units": 1}], "weights": [.25, .75]}
    return dict(scope="artificial", task=task, task_sha256=task_pin(task), target="Y", scale=2.,
        query_action_ids=["q0", "q1"], query_weights=[.25, .75],
        candidate_ids=["a", "b", "c", "d"], candidate_costs={"a": 1, "b": 2, "c": 4, "d": 4},
        replicate_ids=["r0", "r1", "r2"], scoring_truths=[[1, 2], [3, 4], [5, 6]],
        base_predictions=[[5, 6], [7, 8], [9, 10]],
        branch_predictions={"a": [[1, 6], [7, 4], [5, 6]],
                            "b": [[5, 6], [3, 4], [9, 6]],
                            "c": [[9, 10], [11, 12], [13, 14]],
                            "d": [[9, 10], [11, 12], [13, 14]]},
        tie_multiplier=2., minimum_gain_gap=.5)


class ReplicatedFeedbackTests(unittest.TestCase):
    def test_hand_calculated_weighted_replicate_oracle(self):
        result = aggregate_replicated_feedback(**fixture())
        self.assertEqual(result["base_losses"], [[4, 4]]*3)
        self.assertEqual(result["branch_losses"]["a"], [[0, 4], [4, 0], [0, 0]])
        self.assertEqual(result["branch_losses"]["b"], [[4, 4], [0, 0], [4, 0]])
        self.assertEqual(result["branch_losses"]["c"], [[16, 16]]*3)
        self.assertEqual(result["normalized_loss_reductions"]["b"], [[0, 0], [4, 4], [0, 4]])
        self.assertEqual(result["gains_per_cost"]["a"], [[4, 0], [0, 4], [4, 4]])
        self.assertEqual(result["gains_per_cost"]["b"], [[0, 0], [2, 2], [0, 2]])
        self.assertEqual(result["gains_per_cost"]["c"], [[-3, -3]]*3)
        self.assertEqual(result["per_replicate_gains_per_cost"],
                         {"a": [1, 3, 4], "b": [0, 2, 1.5], "c": [-3]*3, "d": [-3]*3})
        self.assertAlmostEqual(result["mean_gains_per_cost"]["a"], 8/3)
        self.assertAlmostEqual(result["mean_gains_per_cost"]["b"], 7/6)
        pairs = {(p["left"], p["right"]): p for p in result["pairs"]}
        self.assertEqual(set(pairs), set(itertools.combinations(["a", "b", "c", "d"], 2)))
        ab = pairs["a", "b"]
        # Independent oracle: replicate differences 1, 1, 5/2 have mean 3/2.
        # Centered sum of squares = 3/2; sample variance = 3/4; SE = 1/2.
        self.assertEqual(ab["replicate_differences"], [1, 1, 2.5])
        self.assertEqual(ab["mean_gain_difference"], 1.5)
        self.assertAlmostEqual(ab["paired_standard_error"], .5)
        # Wrong pooled-row treatment, even preserving the same weighted mean:
        # Q*w_q*(g_a-g_b) = [2,0,-1,3,2,3]. Sum=9, sum squares=27.
        # Pooled SE^2 = (27 - 9^2/6)/(5*6) = 9/20, not 1/4.
        pooled_se = math.sqrt(float((Fraction(27)-Fraction(81, 6))/30))
        self.assertAlmostEqual(pooled_se, math.sqrt(.45))
        self.assertNotAlmostEqual(ab["paired_standard_error"], pooled_se)
        self.assertEqual(ab["tie_threshold"], 1.5)
        self.assertTrue(ab["tie"])  # Inclusive threshold, no strict preference.
        self.assertIsNone(ab["chosen"])
        self.assertIsNone(ab["rejected"])
        self.assertEqual(pairs["a", "c"]["chosen"], "a")
        self.assertEqual(pairs["c", "d"]["paired_standard_error"], 0.)
        self.assertTrue(pairs["c", "d"]["tie"])

    def test_all_negative_candidates_are_kept_without_stop_inference(self):
        args = fixture()
        args.update(tie_multiplier=0., minimum_gain_gap=0., base_predictions=copy.deepcopy(args["scoring_truths"]))
        for c, offset in {"a": 2, "b": 4, "c": 8, "d": 8}.items():
            args["branch_predictions"][c] = [[t+offset for t in row] for row in args["scoring_truths"]]
        result = aggregate_replicated_feedback(**args)
        self.assertEqual(result["mean_gains_per_cost"], {"a": -1, "b": -2, "c": -4, "d": -4})
        self.assertEqual(len(result["pairs"]), 6)
        self.assertEqual(result["pairs"][0]["chosen"], "a")
        self.assertFalse(result["stop_inference_performed"])
        self.assertNotIn("stop_label", result)

    def test_raw_custody_purity_and_declared_provenance(self):
        args = fixture()
        before = copy.deepcopy(args)
        result = aggregate_replicated_feedback(**args)
        self.assertEqual(args, before)
        for key in ("base_predictions", "branch_predictions", "scoring_truths", "task",
                    "replicate_ids", "candidate_ids", "candidate_costs", "query_weights", "query_action_ids"):
            self.assertEqual(result[key], args[key])
        json.dumps(result, allow_nan=False)
        result["branch_predictions"]["a"][0][0] = 1000
        result["task"]["weights"][0] = 0
        result["candidate_costs"]["a"] = 99
        self.assertEqual(args, before)
        self.assertFalse(result["provenance_qualified"])
        self.assertIn("not an inference certificate", result["tie_rule_scope"])
        self.assertIn("not authenticated", result["provenance_note"])
        self.assertEqual(result["standard_error_ddof"], 1)

    def test_row_permutation_and_single_query_two_replicates(self):
        args = fixture()
        original = aggregate_replicated_feedback(**args)
        order = [2, 0, 1]
        for key in ("replicate_ids", "base_predictions", "scoring_truths"):
            args[key] = [args[key][r] for r in order]
        for c in args["candidate_ids"]:
            args["branch_predictions"][c] = [args["branch_predictions"][c][r] for r in order]
        permuted = aggregate_replicated_feedback(**args)
        self.assertEqual(permuted["mean_gains_per_cost"], original["mean_gains_per_cost"])
        for a, b in zip(original["pairs"], permuted["pairs"]):
            self.assertEqual(b["replicate_differences"], [a["replicate_differences"][r] for r in order])
            self.assertEqual(a["paired_standard_error"], b["paired_standard_error"])
        args = fixture()
        args["task"]["queries"] = args["task"]["queries"][:1]
        args["task"]["weights"] = [1.]
        args.update(task_sha256=task_pin(args["task"]), query_action_ids=["q0"], query_weights=[1.],
                    replicate_ids=["r0", "r1"])
        for key in ("base_predictions", "scoring_truths"):
            args[key] = [row[:1] for row in args[key][:2]]
        for c in args["candidate_ids"]:
            args["branch_predictions"][c] = [row[:1] for row in args["branch_predictions"][c][:2]]
        ab = aggregate_replicated_feedback(**args)["pairs"][0]
        self.assertEqual(ab["replicate_differences"], [4, -2])
        self.assertAlmostEqual(ab["paired_standard_error"], 3.)

    def test_scope_rejection(self):
        for scope in ("development", "confirmation", "private_eval", "policy_train ", "", None, True):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                aggregate_replicated_feedback(**dict(fixture(), scope=scope))
        for scope in ("artificial", "policy_train"):
            result = aggregate_replicated_feedback(**dict(fixture(), scope=scope))
            self.assertEqual(result["scope"], scope)
            self.assertFalse(result["provenance_qualified"])

    def test_strict_task_binding(self):
        variants = [dict(task_sha256="0"*64), dict(target="X"), dict(scale=3.),
                    dict(query_action_ids=["q1", "q0"]), dict(query_action_ids=["q0", "q0"]),
                    dict(query_weights=[.25+1e-14, .75-1e-14]), dict(query_weights=[.5, .5])]
        for delta in variants:
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                aggregate_replicated_feedback(**dict(fixture(), **delta))
        for change in ("missing", "extra", "query_duplicate", "weight_sum", "zero_weight", "negative_scale",
                       "zero_tolerance", "pin_stale", "bad_query"):
            args = fixture()
            task = args["task"]
            if change == "missing": del task["target"]
            elif change == "extra": task["answer"] = 4.
            elif change == "query_duplicate": task["queries"][1]["action_id"] = "q0"
            elif change == "weight_sum": task["weights"] = [.25, .5]
            elif change == "zero_weight": task["weights"] = [0., 1.]
            elif change == "negative_scale": task["scale"] = -2.
            elif change == "zero_tolerance": task["normalized_mse_tolerance"] = 0.
            elif change == "pin_stale": task["queries"][0]["targets"][0][1] = 9.
            elif change == "bad_query": task["queries"][0]["cost_units"] = 0
            if change != "pin_stale": args["task_sha256"] = task_pin(task)
            with self.subTest(change=change), self.assertRaises(ValueError):
                aggregate_replicated_feedback(**args)

    def test_missing_members_and_shapes_rejected(self):
        for change in ("missing_branch", "extra_branch", "missing_cost", "extra_cost", "duplicate_candidate",
                       "duplicate_replicate", "one_replicate", "one_candidate", "base_rows", "truth_columns",
                       "branch_rows", "branch_columns", "scalar_row", "string_matrix"):
            args = fixture()
            if change == "missing_branch": del args["branch_predictions"]["b"]
            elif change == "extra_branch": args["branch_predictions"]["extra"] = args["base_predictions"]
            elif change == "missing_cost": del args["candidate_costs"]["b"]
            elif change == "extra_cost": args["candidate_costs"]["extra"] = 1
            elif change == "duplicate_candidate": args["candidate_ids"] = ["a", "a"]
            elif change == "duplicate_replicate": args["replicate_ids"] = ["r0", "r0", "r2"]
            elif change == "one_replicate": args["replicate_ids"] = ["r0"]
            elif change == "one_candidate": args["candidate_ids"] = ["a"]
            elif change == "base_rows": args["base_predictions"].pop()
            elif change == "truth_columns": args["scoring_truths"][0].pop()
            elif change == "branch_rows": args["branch_predictions"]["a"].pop()
            elif change == "branch_columns": args["branch_predictions"]["a"][0].append(0.)
            elif change == "scalar_row": args["scoring_truths"][0] = 1.
            elif change == "string_matrix": args["base_predictions"] = "not a matrix"
            with self.subTest(change=change), self.assertRaises(ValueError):
                aggregate_replicated_feedback(**args)

    def test_nonfinite_and_invalid_numbers_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf"), None, True, "1"):
            for location in ("base", "truth", "branch", "cost", "weight", "scale", "tie_multiplier", "minimum_gain_gap"):
                args = fixture()
                if location == "base": args["base_predictions"][0][0] = bad
                elif location == "truth": args["scoring_truths"][0][0] = bad
                elif location == "branch": args["branch_predictions"]["d"][2][1] = bad
                elif location == "cost": args["candidate_costs"]["a"] = bad
                elif location == "weight": args["query_weights"][0] = bad
                else: args[location] = bad
                with self.subTest(bad=bad, location=location), self.assertRaises(ValueError):
                    aggregate_replicated_feedback(**args)
        for bad in (0., -1.):
            args = fixture()
            args["candidate_costs"]["a"] = bad
            with self.subTest(cost=bad), self.assertRaises(ValueError):
                aggregate_replicated_feedback(**args)
        for key in ("tie_multiplier", "minimum_gain_gap"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                aggregate_replicated_feedback(**dict(fixture(), **{key: -1.}))

    def test_nonfinite_derived_feedback_is_rejected(self):
        for change in ("squared_loss", "gain_per_cost", "tie_threshold"):
            args = fixture()
            if change == "squared_loss": args["base_predictions"][0][0] = 1e308
            elif change == "gain_per_cost": args["candidate_costs"]["a"] = 1e-320
            else: args.update(tie_multiplier=1.7e308, minimum_gain_gap=1.7e308)
            with self.subTest(change=change), self.assertRaises(ValueError):
                aggregate_replicated_feedback(**args)


if __name__ == "__main__":
    unittest.main(verbosity=2)

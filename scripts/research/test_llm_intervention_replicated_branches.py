"""Handcrafted two-node artificial integrations, with retained test artifacts."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from llm_intervention_contract import Action, Node, canonical, digest
from llm_intervention_replicated_branches import build_replicated_training_preferences
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_task import PredictionTask


GRAPH = (Node("X", (), "unit", (-2., 2.)), Node("Y", ("X",), "unit"))
MENU = (Action("cheap", "intervene", (("X", 0.),), ("Y",), 1),
        Action("costly", "intervene", (("X", 0.),), ("Y",), 2))
QUERIES = (Action("q0", "intervene", (("X", 0.),), ("Y",), 3),
           Action("q1", "intervene", (("X", 1.),), ("Y",), 1))
TASK = PredictionTask("Y", QUERIES, (.25, .75), 2., .04)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class ReplicatedBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(cls, "root"):
            cls.root = Path(tempfile.mkdtemp(prefix="ace-replicated-branches-"))
        # Keep all artifacts, including evidence from any failed test attempt.

    def setUp(self):
        self.calls = []

    def run_case(self, name, **overrides):
        output = self.root/name
        def reserved(kind, rid, action, request):
            plan = json.loads((output/"plan.json").read_text())
            entry = next(e for e in plan["journals"] if e["kind"] == kind and e["replicate_id"] == rid
                         and (kind == "scoring" or e["candidate_id"] == action.action_id))
            last = records(output/entry["journal"])[-1]
            self.assertEqual((last["event"], last["request_id"], last["cost_units"]),
                             ("reserve", request, action.cost_units))
            self.calls.append((kind, rid, action.action_id, request))
        def branch(rid, candidate, action, request):
            reserved("branch", rid, action, request)
            return {"Y": {("r0", "cheap"): 2., ("r0", "costly"): 4.,
                          ("r1", "cheap"): 4., ("r1", "costly"): 2.}[rid, candidate]}
        def score(rid, action, request):
            reserved("scoring", rid, action, request)
            return {"Y": {("r0", "q0"): 2., ("r0", "q1"): 4.,
                          ("r1", "q0"): 4., ("r1", "q1"): 2.}[rid, action.action_id]}
        params = dict(scope="artificial", graph=GRAPH, menu=MENU, candidates=("cheap", "costly"),
            learner=LinearGaussianSCM(GRAPH, {"X": 1., "Y": 1.}), next_learner_request_id=0,
            task=TASK, replicate_ids=("r0", "r1"), branch_respond=branch, score_respond=score,
            tie_multiplier=2., minimum_gain_gap=0., output=output)
        params.update(overrides)
        return build_replicated_training_preferences(**params)

    def test_hand_computed_paired_gains_counts_and_charges(self):
        result = self.run_case("oracle")
        f = result["feedback"]
        # At X=0, prior covariance I and noise=1 give posterior [y/2,0].
        # Query predictions are therefore [y/2,y/2], independent of query X.
        self.assertEqual(f["base_predictions"], [[0., 0.]]*2)
        self.assertEqual(f["branch_predictions"], {"cheap": [[1., 1.], [2., 2.]],
                                                   "costly": [[2., 2.], [1., 1.]]})
        self.assertEqual(f["scoring_truths"], [[2., 4.], [4., 2.]])
        self.assertEqual(f["base_losses"], [[1., 4.], [4., 1.]])
        self.assertEqual(f["gains_per_cost"]["cheap"], [[.75, 1.75], [3., 1.]])
        self.assertEqual(f["gains_per_cost"]["costly"], [[.5, 1.5], [.875, .375]])
        self.assertEqual(f["per_replicate_gains_per_cost"], {"cheap": [1.5, 1.5], "costly": [1.25, .5]})
        pair = f["pairs"][0]
        self.assertEqual(pair["replicate_differences"], [.25, 1.])
        self.assertEqual(pair["mean_gain_difference"], 5/8)
        # For two replicates, SE=abs(d0-d1)/2=3/8, not a query-row SE.
        self.assertAlmostEqual(pair["paired_standard_error"], 3/8)
        self.assertTrue(pair["tie"])
        self.assertEqual((result["charged_branch_cost"], result["charged_scoring_cost"]), (6, 8))
        self.assertEqual(result["environment_responses"], 8)
        self.assertEqual(len(self.calls), 8)
        self.assertEqual(len(set(self.calls)), 8)
        accounting = result["accounting"]
        self.assertTrue(accounting["accounting_complete"])
        self.assertEqual(accounting["total_reserved_cost_units"], 14)
        self.assertEqual(accounting["unknown_return_attempts"], 0)
        self.assertEqual(accounting["branch"]["reserved_response_attempts"], 4)
        self.assertEqual(accounting["scoring"]["reserved_response_attempts"], 4)
        plan = json.loads((self.root/"oracle/plan.json").read_text())
        self.assertEqual(plan["task_sha256"], digest(TASK.payload()))
        self.assertEqual(result["plan_sha256"], digest(plan))
        self.assertFalse(result["provenance_qualified"])
        self.assertFalse(f["provenance_qualified"])

    def test_forks_preserve_existing_base_and_request_mapping(self):
        base = LinearGaussianSCM(GRAPH, {"X": 1., "Y": 1.})
        base.update({"request_id": 7, "targets": {"X": 0.}, "values": {"Y": 2.}})
        before = canonical([asdict(s) for s in base.summaries()])
        a = self.run_case("existing", learner=base, next_learner_request_id=8)
        b = self.run_case("reordered", learner=base, next_learner_request_id=8, candidates=("costly", "cheap"))
        self.assertEqual(before, canonical([asdict(s) for s in base.summaries()]))
        self.assertEqual(base._requests, {7})
        self.assertEqual(a["feedback"]["branch_predictions"], b["feedback"]["branch_predictions"])
        # Posterior intercept after old y=2 and one new y is (2+y)/3.
        for actual, expected in zip(a["feedback"]["branch_predictions"]["cheap"], (4/3, 2.)):
            for value in actual: self.assertAlmostEqual(value, expected)
        for path in (self.root/"existing").glob("*.branch.json"):
            record = json.loads(path.read_text())
            self.assertEqual(record["mapping"]["source_request_id"], 0)
            self.assertEqual(record["mapping"]["learner_request_id"], 8)
            self.assertEqual(record["post_state"]["request_ids"], [7, 8])
            self.assertEqual(record["post_state"]["summaries"][1]["eligible_rows"], 2)

    def test_scoring_is_shared_and_never_updates_learners(self):
        a = self.run_case("normal-scoring")
        calls = []
        def zero(rid, action, request):
            calls.append((rid, action.action_id, request))
            return {"Y": 0.}
        b = self.run_case("zero-scoring", score_respond=zero, tie_multiplier=0.)
        self.assertEqual(len(calls), 4)
        self.assertEqual(a["feedback"]["branch_predictions"], b["feedback"]["branch_predictions"])
        self.assertTrue(all(g < 0 for g in b["feedback"]["mean_gains_per_cost"].values()))
        self.assertEqual(b["feedback"]["pairs"][0]["chosen"], "costly")
        self.assertEqual(b["scoring_learner_updates"], 0)
        self.assertFalse(b["stop_inference_performed"])

    def test_target_scope_and_request_guards_precede_callbacks(self):
        no_target = PredictionTask("Y", (Action("q", "intervene", (("X", 0.),), ("X",), 1),), (1.,), 2., .04)
        base = LinearGaussianSCM(GRAPH, {"X": 1., "Y": 1.})
        base.update({"request_id": 7, "targets": {"X": 0.}, "values": {"Y": 2.}})
        variants = [{"scope": "development"}, {"scope": "confirmation"}, {"task": no_target},
                    {"learner": base, "next_learner_request_id": 7}, {"replicate_ids": ("r0",)},
                    {"replicate_ids": ("r0", "r0")}, {"candidates": ("cheap", "missing")}]
        for index, overrides in enumerate(variants):
            name = "invalid-"+str(index)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.run_case(name, **overrides)
            self.assertFalse((self.root/name).exists())
        self.assertEqual(self.calls, [])

    def test_callback_failure_preserves_reserved_unknown_attempt_without_retry(self):
        calls = []
        def fail(rid, candidate, action, request):
            calls.append((rid, candidate))
            if candidate == "costly": raise RuntimeError("artificial response outcome unknown")
            return {"Y": 2.}
        with self.assertRaisesRegex(RuntimeError, "outcome unknown"):
            self.run_case("callback-failed", branch_respond=fail)
        output = self.root/"callback-failed"
        failure = json.loads((output/"failure.json").read_text())
        self.assertEqual(calls, [("r0", "cheap"), ("r0", "costly")])
        self.assertFalse((output/"terminal.json").exists())
        self.assertEqual(failure["accounting"]["total_reserved_cost_units"], 3)
        self.assertEqual(failure["accounting"]["reserved_response_attempts"], 2)
        self.assertEqual(failure["accounting"]["known_returns"], 1)
        self.assertEqual(failure["accounting"]["branch"]["unknown_return_attempts"], 1)
        self.assertEqual(failure["accounting"]["branch"]["unknown_return_reserved_cost"], 2)
        events = records(output/"replicate-000.branch-001.jsonl")
        self.assertEqual([e["event"] for e in events], ["begin", "proposal", "reserve", "failure"])
        self.assertIsNone(events[-1]["response_returned_to_driver"])
        self.assertTrue((output/"replicate-000.branch-001.mapping.json").exists())
        self.assertFalse(failure["retry_allowed"])
        self.assertFalse(failure["complete_case_filtering_allowed"])
        self.assertTrue(failure["base_learner_unchanged"])
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError): self.run_case("callback-failed", branch_respond=fail)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
        self.assertEqual(len(calls), 2)

    def test_returned_invalid_scoring_value_is_retained_and_charged(self):
        calls = []
        def invalid(rid, action, request):
            calls.append((rid, request))
            return {"Y": float("nan") if request == 1 else 2.}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.run_case("invalid-return", score_respond=invalid)
        output = self.root/"invalid-return"
        failure = json.loads((output/"failure.json").read_text())
        raw = json.loads((output/"replicate-000.scoring.return-001.json").read_text())
        self.assertFalse(raw["raw_json_available"])
        self.assertIn("nan", raw["raw_response_repr"])
        self.assertEqual(calls, [("r0", 0), ("r0", 1)])
        self.assertEqual(failure["accounting"]["total_reserved_cost_units"], 7)
        self.assertEqual(failure["accounting"]["scoring"]["known_returns"], 2)
        self.assertEqual(failure["accounting"]["scoring"]["returned_without_accepted_response"], 1)
        self.assertEqual(failure["accounting"]["unknown_return_attempts"], 0)
        self.assertFalse((output/"terminal.json").exists())

    def test_scoring_clamp_contradiction_is_preserved(self):
        queries = (Action("clamped", "intervene", (("X", 0.),), ("X", "Y"), 3),)
        task = PredictionTask("Y", queries, (1.,), 2., .04)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.run_case("clamp", task=task, score_respond=lambda rid, a, r: {"X": 1., "Y": 2.})
        output = self.root/"clamp"
        events = records(output/"replicate-000.scoring.jsonl")
        self.assertEqual(events[-2]["event"], "response")
        self.assertEqual(events[-2]["values"]["X"], 1.)
        self.assertEqual(events[-1]["event"], "driver_failure")
        self.assertFalse((output/"terminal.json").exists())

    def test_callback_mutation_of_base_fails_without_later_branches(self):
        base = LinearGaussianSCM(GRAPH, {"X": 1., "Y": 1.})
        calls = []
        def mutate(rid, candidate, action, request):
            calls.append(candidate)
            base.heads["Y"].mean[0] = 99.
            return {"Y": 2.}
        with self.assertRaisesRegex(RuntimeError, "base learner changed"):
            self.run_case("mutation", learner=base, branch_respond=mutate)
        failure = json.loads((self.root/"mutation/failure.json").read_text())
        self.assertEqual(calls, ["cheap"])
        self.assertFalse(failure["base_learner_unchanged"])
        self.assertEqual(failure["accounting"]["known_returns"], 1)
        self.assertEqual(failure["accounting"]["unknown_return_attempts"], 0)
        self.assertEqual(failure["accounting"]["total_reserved_cost_units"], 1)

    def test_existing_output_is_untouched_and_calls_nobody(self):
        output = self.root/"exists"
        output.mkdir()
        (output/"sentinel").write_bytes(b"preserve")
        with self.assertRaises(FileExistsError): self.run_case("exists")
        self.assertEqual(self.calls, [])
        self.assertEqual([p.name for p in output.iterdir()], ["sentinel"])
        self.assertEqual((output/"sentinel").read_bytes(), b"preserve")


class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams: stream.write(text)
    def flush(self):
        for stream in self.streams: stream.flush()


if __name__ == "__main__":
    ReplicatedBranchTests.root = Path(tempfile.mkdtemp(prefix="ace-replicated-branches-", dir="/tmp"))
    with (ReplicatedBranchTests.root/"test-report.txt").open("x") as report:
        stream = _Tee(sys.stderr, report)
        stream.write("Interpreter: " + sys.executable + "\nVersion: " + sys.version.replace("\n", " ") + "\n")
        stream.write("Retained artifacts/report: " + str(ReplicatedBranchTests.root) + "\n")
        stream.flush()
        unittest.main(testRunner=unittest.TextTestRunner(stream=stream, verbosity=2))

"""New artificial-only assembly regressions; every fixture stays in custody."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from llm_intervention_contract import Action, Node, canonical, digest
from llm_intervention_replicated_branches import _snapshot
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_supervision import matched_views
from llm_intervention_task import PredictionTask
from llm_intervention_training_dataset import Snapshot, assemble_training_dataset
from llm_intervention_world import LinearMechanism, ResponseAdapter, WorldDescriptor


GRAPH = (Node("X", (), "unit", (-2., 2.)), Node("Y", ("X",), "unit"))
MENU = (Action("cheap", "intervene", (("X", 0.),), ("Y",), 1),
        Action("twin", "intervene", (("X", 0.),), ("Y",), 1),
        Action("costly", "intervene", (("X", 0.),), ("Y",), 2))
QUERIES = (Action("q0", "intervene", (("X", 0.),), ("Y",), 2),
           Action("q1", "intervene", (("X", 1.),), ("Y",), 3))
TASK = PredictionTask("Y", QUERIES, (.25, .75), 2., .04)
CANDIDATES = tuple(a.action_id for a in MENU)


def read(path):
    return json.loads(path.read_text())


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    with path.open("x") as handle:
        handle.write(canonical(value) + "\n")


def world(intercept=2., coefficient=0.):
    # Direct analytic descriptor, never generate_world or a scientific namespace.
    return WorldDescriptor("artificial", 0, GRAPH,
        (LinearMechanism("X", 0., (), 0.), LinearMechanism("Y", intercept, (("X", coefficient),), 0.)))


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["ACE_DATASET_TEST_OUTPUT"])
        cls.root.mkdir(exist_ok=False)

    def setUp(self):
        self.directory = self.root/self._testMethodName
        self.directory.mkdir(exist_ok=False)

    def parameters(self, name="assembly", **overrides):
        descriptor = overrides.get("world", world())
        task = overrides.get("task", TASK)
        params = dict(world=descriptor, expected_world_sha256=descriptor.sha256,
            scope="artificial", graph=GRAPH, menu=MENU, task=task, expected_task_sha256=digest(task.payload()),
            learner=LinearGaussianSCM(GRAPH, {"X": 1., "Y": 1.}), initial_actions=("cheap",),
            total_budget=7, behavior_method="menu_coverage",
            snapshots=(Snapshot(0, "menu_coverage", CANDIDATES), Snapshot(1, "local_information", CANDIDATES),
                       Snapshot(2, "target_risk", CANDIDATES), Snapshot(6, "uniform_random", ())),
            replicate_ids=("r0", "r1"), planning_draws=2, tie_multiplier=2., minimum_gain_gap=0.,
            structural_scores=True, objective="Predict the supplied artificial target law.", domain_notes="",
            request_namespace="artificial-assembly", output=self.directory/name)
        params.update(overrides)
        return params

    def test_analytic_counts_cost_crossings_precommitted_teachers_and_matched_views(self):
        params, observed = self.parameters(), []
        output, original = params["output"], ResponseAdapter.__call__
        before = digest(_snapshot(params["learner"]))

        def audited(adapter, action, request):
            plan = read(output/"plan.json")
            self.assertEqual(plan["behavior_method"], "menu_coverage")
            self.assertEqual([s["teacher_method"] for s in plan["snapshots"]],
                             ["menu_coverage", "local_information", "target_risk", "uniform_random"])
            reservation = events(adapter._executor)[-1]
            self.assertEqual((reservation["event"], reservation["request_id"], reservation["cost_units"]),
                             ("reserve", request, action.cost_units))
            if adapter._executor.parent.name == "feedback":
                snapshot = adapter._executor.parent.parent
                self.assertTrue((snapshot/"teacher.json").exists())
                self.assertTrue((snapshot/"envelope.json").exists())
                self.assertTrue((snapshot/"selection.json").exists())
                self.assertTrue(read(snapshot/"teacher.json")["chosen_before_branch_responses"])
            observed.append({"stream": adapter._stream, "journal": str(adapter._executor),
                             "request_id": request, "action": action.action_id})
            return original(adapter, action, request)

        with patch.object(ResponseAdapter, "__call__", audited):
            result = assemble_training_dataset(**params)
        write(self.directory/"reservation-observations.json", observed)
        self.assertTrue(result["dataset_complete"])
        self.assertEqual(len(observed), 35)
        self.assertEqual(result["accounting"]["total_reserved_cost_units"], 61)
        for kind, responses, cost in (("trajectory", 5, 7), ("branch", 18, 24), ("scoring", 12, 30)):
            self.assertEqual(result["accounting"][kind]["reserved_response_attempts"], responses)
            self.assertEqual(result["accounting"][kind]["reserved_cost_units"], cost)
            self.assertEqual(result["accounting"][kind]["executor_accepted_responses"], responses)
        self.assertEqual((result["actual_initial_cost_units"], result["actual_acquisition_cost_units"]), (1, 6))
        self.assertEqual((result["trajectory_learner_updates"], result["scoring_learner_updates"]), (5, 0))
        self.assertEqual(digest(_snapshot(params["learner"])), before)
        final = result["final_or_partial_state"]
        y = next(h for h in final["scm"] if h["node"] == "Y")
        self.assertEqual(y["eligible_rows"], 5)
        self.assertAlmostEqual(y["parameter_mean"][0], 5/3)
        self.assertEqual(len(final["history"]), 5)
        selections = [read(output/f"snapshot-{i:03d}/selection.json") for i in range(4)]
        self.assertEqual([s["actual_acquisition_cost_units"] for s in selections], [0, 2, 2, 6])
        self.assertEqual([s["trajectory_request_count"] for s in selections], [1, 2, 2, 5])
        self.assertEqual((output/"snapshot-001/envelope.json").read_bytes(),
                         (output/"snapshot-002/envelope.json").read_bytes())
        # Independent conjugate oracle: one y=2 observation => mean 1;
        # one further y=2 => mean 4/3. Scale=2: gain=1/4 - 1/9=5/36.
        feedback = read(output/"snapshot-000/feedback/terminal.json")["feedback"]
        for candidate, gain in (("cheap", 5/36), ("twin", 5/36), ("costly", 5/72)):
            self.assertAlmostEqual(feedback["mean_gains_per_cost"][candidate], gain)
        self.assertEqual(len(feedback["pairs"]), 3)
        self.assertEqual(feedback["query_weights"], [.25, .75])
        self.assertTrue(feedback["pairs"][0]["tie"])
        self.assertEqual(feedback["pairs"][0]["replicate_differences"], [0., 0.])
        self.assertEqual(feedback["pairs"][0]["paired_standard_error"], 0.)
        self.assertEqual((result["training_pairs"], result["tied_pairs"]), (9, 3))
        for entry in result["completed_records"]:
            directory = (output/entry["path"]).parent
            raw = (directory/"record.json").read_bytes()
            self.assertEqual(hashlib.sha256((directory/"receipt.json").read_bytes()).hexdigest(), entry["receipt_sha256"])
            views = matched_views(raw, record_sha256=entry["record_sha256"],
                supervision_receipt_sha256=entry["receipt_sha256"], allowed_scope="artificial")
            self.assertEqual(read(directory/"language.json"), json.loads(canonical(views["language"])))
            np.testing.assert_array_equal(np.load(directory/"numeric-features.npy", allow_pickle=False),
                                          views["numeric"]["features"])
            public = read(directory/"envelope.json")
            self.assertEqual(set(public), {"version", "state", "task"})
            self.assertNotIn("world_sha256", canonical(public))
            self.assertNotIn("scoring_truths", canonical(public))
        terminal_record = read(output/"snapshot-003/record.json")
        self.assertIsNone(terminal_record["supervised_action_id"])
        self.assertEqual(terminal_record["preferences"], [])
        self.assertTrue(terminal_record["envelope"]["state"]["terminal"])
        self.assertFalse(result["provenance_qualified"])
        # Identical costs/assignments with changed values keep exactly the same
        # crossing states. Teachers may change, but cannot select milestones.
        other = self.parameters("different-outcomes", world=world(intercept=-3.))
        assemble_training_dataset(**other)
        self.assertEqual([read(other["output"]/f"snapshot-{i:03d}/selection.json")["actual_acquisition_cost_units"]
                          for i in range(4)], [0, 2, 2, 6])
        before_files = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in output.rglob("*") if p.is_file()}
        with patch.object(ResponseAdapter, "__call__", side_effect=AssertionError("must not respond")):
            with self.assertRaises(FileExistsError):
                assemble_training_dataset(**params)
        self.assertEqual(before_files, {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in output.rglob("*") if p.is_file()})

    def test_negative_gains_and_ties_survive_complete_shared_pair_set(self):
        menu = MENU[:2] + (replace(MENU[2], targets=(("X", 1.),)),)
        task = replace(TASK, queries=(QUERIES[0], replace(QUERIES[1], targets=(("X", 0.),))))
        params = self.parameters(world=world(0., 2.), menu=menu, task=task,
            snapshots=(Snapshot(0, "menu_coverage", CANDIDATES),))
        result = assemble_training_dataset(**params)
        feedback = read(params["output"]/"snapshot-000/feedback/terminal.json")["feedback"]
        # After initial y(0)=0, covariance diag(1/2,1); y(1)=2 moves
        # intercept to .4. Both scoring queries have truth 0: -.4²/(2²*2).
        self.assertAlmostEqual(feedback["mean_gains_per_cost"]["costly"], -.02)
        self.assertEqual(feedback["mean_gains_per_cost"]["cheap"], 0.)
        record = read(params["output"]/"snapshot-000/record.json")
        self.assertEqual(len(record["preferences"]), 3)
        self.assertEqual(sum(p["tie"] for p in record["preferences"]), 1)
        self.assertEqual({p["rejected"] for p in record["preferences"] if not p["tie"]}, {"costly"})
        self.assertEqual(result["accounting"]["branch"]["reserved_response_attempts"], 6)

    def test_trajectory_failure_keeps_prior_record_and_reserved_unknown_return(self):
        params = self.parameters(snapshots=(Snapshot(0, "menu_coverage", CANDIDATES), Snapshot(6, "menu_coverage", ())))
        original = ResponseAdapter.__call__
        def fail(adapter, action, request):
            value = original(adapter, action, request)
            if adapter._executor.name == "trajectory.jsonl" and adapter._stream == "measurement":
                raise RuntimeError("artificial interruption after world outcome prepared")
            return value
        with patch.object(ResponseAdapter, "__call__", fail), self.assertRaisesRegex(RuntimeError, "outcome prepared"):
            assemble_training_dataset(**params)
        output = params["output"]
        failure = read(output/"failure.json")
        self.assertEqual(len(failure["completed_records"]), 1)
        self.assertEqual(failure["unfinished_snapshot_indices"], [1])
        self.assertFalse(failure["dataset_complete"])
        self.assertFalse(failure["partial_records_authorized_for_training"])
        self.assertEqual(failure["accounting"]["total_reserved_cost_units"], 21)
        self.assertEqual(failure["accounting"]["trajectory"]["reserved_cost_units"], 3)
        self.assertEqual(failure["accounting"]["trajectory"]["unknown_return_attempts"], 1)
        self.assertEqual(failure["accounting"]["trajectory"]["unknown_return_reserved_cost"], 2)
        self.assertEqual(failure["trajectory_learner_updates"], 1)
        self.assertTrue(any(e["event"] == "outcome" for e in events(output/"acquisition.world.jsonl")))
        self.assertIsNone(events(output/"trajectory.jsonl")[-1]["response_returned_to_driver"])
        self.assertFalse((output/"terminal.json").exists())

    def test_returned_invalid_scoring_label_retained_and_charged_without_completion(self):
        params, original = self.parameters(), ResponseAdapter.__call__
        def invalid(adapter, action, request):
            value = original(adapter, action, request)
            if adapter._stream == "train_score" and request == 1:
                return {"Y": float("nan")}
            return value
        with patch.object(ResponseAdapter, "__call__", invalid), self.assertRaisesRegex(ValueError, "non-finite"):
            assemble_training_dataset(**params)
        output = params["output"]
        failure = read(output/"failure.json")
        self.assertEqual(failure["completed_records"], [])
        self.assertEqual(failure["accounting"]["total_reserved_cost_units"], 10)
        self.assertEqual((failure["accounting"]["known_returns"], failure["accounting"]["executor_accepted_responses"]), (6, 5))
        self.assertEqual(failure["accounting"]["unknown_return_attempts"], 0)
        self.assertEqual(failure["accounting"]["scoring"]["returned_without_acceptance"], 1)
        raw = read(output/"snapshot-000/feedback/replicate-000.scoring.return-001.json")
        self.assertFalse(raw["raw_json_available"])
        self.assertIn("nan", raw["raw_response_repr"])
        self.assertTrue((output/"snapshot-000/teacher.json").exists())
        self.assertFalse((output/"snapshot-000/record.json").exists())
        self.assertEqual(failure["trajectory_learner_updates"], 1)
        self.assertFalse(failure["retry_allowed"])
        self.assertFalse(failure["complete_case_filtering_allowed"])

    def test_invalid_scope_pins_task_bounds_and_plan_rejected_before_response(self):
        missing_target = replace(TASK, queries=(replace(QUERIES[0], measure=("X",)), QUERIES[1]))
        variants = [dict(scope="development"), dict(scope="confirmation"),
            dict(expected_world_sha256="0"*64), dict(expected_task_sha256="0"*64),
            dict(task=missing_target), dict(initial_actions=("missing",)), dict(total_budget=100),
            dict(behavior_method="outcome_selected"), dict(replicate_ids=("r0",)),
            dict(snapshots=(Snapshot(0, "menu_coverage", ("cheap",)),)),
            dict(snapshots=(Snapshot(2, "menu_coverage", CANDIDATES), Snapshot(1, "menu_coverage", CANDIDATES))),
            dict(snapshots=(Snapshot(0, "not_a_teacher", CANDIDATES),)),
            dict(snapshots=(Snapshot(7, "menu_coverage", CANDIDATES),))]
        rejections = []
        with patch.object(ResponseAdapter, "__call__", side_effect=AssertionError("no response authorized")):
            for i, override in enumerate(variants):
                params = self.parameters(str(i), **override)
                with self.subTest(index=i):
                    try:
                        assemble_training_dataset(**params)
                    except ValueError as exc:
                        rejections.append({"case": i, "overrides_repr": repr(override), "exception": str(exc)})
                    else:
                        self.fail("invalid plan accepted")
                    self.assertFalse(params["output"].exists())
        write(self.directory/"rejections.json", rejections)
        self.assertEqual(len(rejections), len(variants))

    def test_unaffordable_candidates_and_unreached_milestones_are_visible_failures(self):
        params = self.parameters("unaffordable", total_budget=3,
                                 snapshots=(Snapshot(1, "menu_coverage", CANDIDATES),))
        with self.assertRaisesRegex(ValueError, "not all affordable"):
            assemble_training_dataset(**params)
        failure = read(params["output"]/"failure.json")
        self.assertEqual(failure["accounting"]["total_reserved_cost_units"], 3)
        self.assertEqual(failure["accounting"]["branch"]["reserved_response_attempts"], 0)
        self.assertTrue((params["output"]/"snapshot-000/teacher.json").exists())
        expensive = tuple(replace(a, cost_units=3) for a in MENU)
        params = self.parameters("unreached", menu=expensive, initial_actions=(), total_budget=2,
                                 snapshots=(Snapshot(1, "menu_coverage", ()),))
        with self.assertRaisesRegex(ValueError, "milestones remain unreached"):
            assemble_training_dataset(**params)
        failure = read(params["output"]/"failure.json")
        self.assertEqual(failure["accounting"]["total_reserved_cost_units"], 0)
        self.assertEqual(failure["unfinished_snapshot_indices"], [0])
        self.assertFalse(failure["dataset_complete"])
        self.assertTrue(failure["accounting"]["accounting_complete"])


if __name__ == "__main__":
    print("Interpreter:", sys.executable, flush=True)
    print("Version:", sys.version.replace("\n", " "), flush=True)
    print("Retained fixtures:", os.environ["ACE_DATASET_TEST_OUTPUT"], flush=True)
    unittest.main(verbosity=2)

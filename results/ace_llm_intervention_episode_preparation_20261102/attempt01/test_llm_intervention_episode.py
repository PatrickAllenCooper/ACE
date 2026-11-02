"""Artificial, analytic fixtures. No scientific generator or pretrained model."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from llm_intervention_contract import Action, ChargedExecutor, Node, canonical, digest
from llm_intervention_episode import ProcessPolicy, build_training_preferences, run_episode
from llm_intervention_numerical_policy import choose
from llm_intervention_scm import LinearGaussianSCM


GRAPH = (Node("X", (), "unit", (-2., 2.)), Node("Y", ("X",), "unit"))
MENU = (Action("low", "intervene", (("X", 0.),), ("Y",), 1),
        Action("high", "intervene", (("X", 2.),), ("Y",), 2))


def learner():
    return LinearGaussianSCM(GRAPH, {"X": 1., "Y": 1.})


def environment(action, request):
    return {"Y": 2. * dict(action.targets)["X"]}


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def preference(self, directory, **overrides):
        params = dict(scope="policy_train", graph=GRAPH, menu=MENU,
            candidates=("low", "high"), learner=learner(), next_learner_request_id=0,
            branch_respond=lambda key, a, r: environment(a, r), scoring_menu=MENU,
            scoring_actions=("high", "high"), score_respond=environment,
            target="Y", scale=1., tie_multiplier=2., minimum_gain_gap=0., output=self.root/directory)
        params.update(overrides)
        return build_training_preferences(**params)

    def test_branch_posterior_preference_and_full_cost_oracle(self):
        result = self.preference("feedback")
        # phi=[1,2], prior I, y=4, noise=1 -> posterior mean=[2/3,4/3].
        # prediction at X=2 is 10/3, squared loss4/9, gain/cost70/9.
        np.testing.assert_allclose(result["predictions"]["high"], [10/3, 10/3])
        np.testing.assert_allclose(result["gains_per_cost"]["high"], [70/9, 70/9])
        self.assertEqual(result["pairs"][0]["chosen"], "high")
        self.assertEqual(result["environment_responses"], 4)
        self.assertEqual((result["charged_branch_cost"], result["charged_scoring_cost"]), (3, 4))
        events = [e for p in (self.root/"feedback").glob("*.jsonl") for e in records(p)]
        self.assertEqual(sum(e["cost_units"] for e in events if e["event"] == "reserve"), 7)

    def test_candidate_order_independence_and_existing_rows(self):
        base = learner()
        base.update({"request_id": 5, "targets": {"X": 0.}, "values": {"Y": 1.}})
        before = canonical([asdict(x) for x in base.summaries()])
        a = self.preference("a", learner=base, next_learner_request_id=6)
        b = self.preference("b", learner=base, next_learner_request_id=6, candidates=("high", "low"))
        self.assertEqual(a["predictions"], b["predictions"])
        self.assertEqual(before, canonical([asdict(x) for x in base.summaries()]))
        self.assertEqual(base.heads["Y"].eligible_rows, 1)

    def test_scoring_outputs_never_train_branch_and_uncertain_pairs_tie(self):
        a = self.preference("a")
        b = self.preference("b", score_respond=lambda action, request: {"Y": 1000. if request else -1000.})
        self.assertEqual(a["predictions"], b["predictions"])
        self.assertTrue(b["pairs"][0]["tie"])
        self.assertIsNone(b["pairs"][0]["chosen"])

    def test_scope_and_candidate_failures_no_fallback(self):
        for scope in ("development", "confirmation", "artificial"):
            with self.assertRaises(ValueError):
                self.preference(scope, scope=scope)
            self.assertFalse((self.root/scope).exists())
        calls = []
        def fail(key, action, request):
            calls.append(key)
            raise RuntimeError("artificial interrupted response")
        with self.assertRaises(RuntimeError):
            self.preference("failed", branch_respond=fail)
        self.assertEqual(calls, ["low"])
        self.assertFalse((self.root/"failed"/"terminal.json").exists())
        events = records(self.root/"failed"/"branch-000.jsonl")
        self.assertEqual([e["event"] for e in events], ["begin", "proposal", "reserve", "failure"])
        self.assertEqual(events[-1]["spent_units"], 1)

    def test_public_process_episode_charging_and_budget_stop(self):
        worker = str(Path(__file__).with_name("llm_intervention_numerical_policy.py"))
        policy = ProcessPolicy((sys.executable, "-B", worker, "--method", "menu_coverage", "--seed", "0"), 10.)
        result = run_episode(graph=GRAPH, menu=MENU, budget=4, initial_actions=("low",),
            learner=learner(), respond=environment, policy=policy, objective="predict Y", output=self.root/"run")
        state = result["final_state"]
        self.assertEqual(state["spent_units"], 4)
        self.assertEqual([r["action_id"] for r in state["history"]], ["low", "high", "low"])
        self.assertEqual(result["policy_decisions"], 2)
        payload = json.loads((self.root/"run"/"decision-0000.input.json").read_text())
        self.assertNotIn("true_parameters", payload)
        self.assertNotIn("scoring_values", payload)
        self.assertEqual(payload["returned_responses"], 1)
        events = records(self.root/"run"/"journal.jsonl")
        previous = None
        for event in events:
            claimed = event.pop("sha256")
            self.assertEqual(event["previous_sha256"], previous)
            self.assertEqual(digest(event), claimed)
            previous = claimed
        self.assertEqual(events[-1]["provenance"] if "provenance" in events[-1] else events[-2]["provenance"],
                         {"kind": "budget_cap"})

    def test_invalid_worker_completion_is_preserved_no_response(self):
        policy = ProcessPolicy((sys.executable, "-c", "print('not an intervention')"), 5.)
        calls = []
        with self.assertRaises(ValueError):
            run_episode(graph=GRAPH, menu=MENU, budget=2, initial_actions=(), learner=learner(),
                respond=lambda a, r: calls.append(a), policy=policy, objective="predict Y", output=self.root/"bad")
        self.assertEqual(calls, [])
        events = records(self.root/"bad"/"journal.jsonl")
        self.assertEqual(events[-1]["event"], "rejected")
        self.assertEqual(events[-2]["completion"], "not an intervention\n")
        self.assertTrue((self.root/"bad"/"failure.json").exists())

    def test_policy_timeout_stops_episode_before_environment(self):
        policy = ProcessPolicy((sys.executable, "-c", "import time; time.sleep(10)"), .05)
        calls = []
        with self.assertRaises(Exception):
            run_episode(graph=GRAPH, menu=MENU, budget=2, initial_actions=(), learner=learner(),
                respond=lambda a, r: calls.append(a), policy=policy, objective="predict Y", output=self.root/"timeout")
        self.assertFalse(calls)
        process = json.loads((self.root/"timeout"/"decision-0000.process.json").read_text())
        self.assertEqual(process["status"], "timeout")
        self.assertEqual(records(self.root/"timeout"/"journal.jsonl")[-1]["event"], "driver_failure")

    def test_numerical_information_oracle_and_distinct_controls(self):
        with ChargedExecutor(GRAPH, MENU, 4, self.root/"state.jsonl", environment) as exe:
            state = exe.state("predict Y", learner().summaries())
        decision, meta = choose(state, method="local_information", seed=0, draws=4)
        self.assertAlmostEqual(meta["scores"]["low"], .5*np.log(2))
        self.assertAlmostEqual(meta["scores"]["high"], .25*np.log(6))
        self.assertEqual(decision, {"action_id": "high"})
        self.assertEqual(meta["planning_draws"], 8)
        choices = {choose(state, method="uniform_random", seed=i)[0]["action_id"] for i in range(8)}
        self.assertEqual(choices, {"low", "high"})
        self.assertEqual(choose(state, method="menu_coverage", seed=0)[1]["planning_draws"], 0)
        state["remaining_units"] = 1
        self.assertEqual(choose(state, method="local_information", seed=0)[0], {"action_id": "low"})


if __name__ == "__main__":
    unittest.main()

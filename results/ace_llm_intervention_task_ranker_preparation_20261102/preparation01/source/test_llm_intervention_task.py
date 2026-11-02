"""Artificial shared-task and process checks, no scientific worlds."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from llm_intervention_contract import Action, ChargedExecutor, Node, canonical, digest
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_task import (PredictionTask, make_envelope, unpack_envelope,
                                   task_prompt, numerical_features, run_task_episode)


GRAPH = (Node("X", (), "u", (-2., 2.)), Node("Y", ("X",), "u"))
MENU = (Action("low", "intervene", (("X", 0.),), ("Y",), 1),
        Action("high", "intervene", (("X", 2.),), ("Y",), 2))
TASK = PredictionTask("Y", (MENU[1],), (1.,), 1., .1)


def new_learner():
    return LinearGaussianSCM(GRAPH, {"X": 1., "Y": 1.})


def example():
    with tempfile.TemporaryDirectory() as d:
        with ChargedExecutor(GRAPH, MENU, 4, Path(d)/"j.jsonl", lambda a, r: {"Y": 1.}) as ex:
            model = new_learner()
            receipt = ex.execute_completion('{"action_id":"low"}')
            model.update(receipt)
            return make_envelope(ex.state("predict Y", model.summaries()), TASK)


class TaskTests(unittest.TestCase):
    def test_exact_task_shared_with_prompt_and_numeric_policy(self):
        e = example()
        state, task, graph, menu, model = unpack_envelope(e)
        prompt = task_prompt(e)
        self.assertIn(canonical(e), prompt)
        self.assertEqual(task, TASK)
        self.assertEqual(model.heads["Y"].eligible_rows, 1)
        ids, features, meta = numerical_features(e)
        self.assertEqual(ids, ("high", "low", None))
        self.assertEqual(features.shape, (3, 8891))
        self.assertEqual(meta["task_sha256"], digest(e["task"]))
        self.assertEqual(meta["envelope_sha256"], digest(e))
        self.assertTrue(np.isfinite(features).all())

    def test_unknown_nested_private_fields_rejected(self):
        paths = [(), ("state",), ("task",), ("state", "graph", 0), ("state", "menu", 0),
                 ("state", "scm", 0), ("state", "history", 0), ("task", "queries", 0)]
        for path in paths:
            e = example()
            node = e
            for part in path:
                node = node[part]
            node["private_evaluation"] = 7.
            with self.assertRaises(ValueError):
                task_prompt(e)
            with self.assertRaises(ValueError):
                numerical_features(e)

    def test_counter_receipt_and_eligibility_mismatches_rejected(self):
        mutations = [lambda s: s.update(spent_units=2),
                     lambda s: s.update(requests=2),
                     lambda s: s["history"][0].update(request_id=True),
                     lambda s: s["history"][0].update(cost_units=True),
                     lambda s: s["history"][0]["targets"].update(X=1.),
                     lambda s: s["scm"][1].update(eligible_rows=0),
                     lambda s: s["scm"][1]["parameter_mean"].__setitem__(0, "1")]
        for mutate in mutations:
            e = example()
            mutate(e["state"])
            with self.assertRaises(ValueError):
                unpack_envelope(e)

    def test_complete_numeric_features_sensitive_to_covariance_and_history(self):
        e = example()
        before = canonical(e)
        a = numerical_features(e)[1]
        changed = copy.deepcopy(e)
        changed["state"]["scm"][1]["parameter_covariance"][0][1] = .1
        changed["state"]["scm"][1]["parameter_covariance"][1][0] = .1
        self.assertFalse(np.array_equal(a, numerical_features(changed)[1]))
        changed = copy.deepcopy(e)
        changed["state"]["history"][0]["values"]["Y"] = 2.
        self.assertFalse(np.array_equal(a, numerical_features(changed)[1]))
        changed = copy.deepcopy(e)
        changed["task"]["normalized_mse_tolerance"] = .2
        self.assertFalse(np.array_equal(a, numerical_features(changed)[1]))
        changed = copy.deepcopy(e)
        changed["state"]["domain_notes"] = "Different linguistic description."
        np.testing.assert_array_equal(a, numerical_features(changed)[1])
        self.assertNotEqual(task_prompt(e), task_prompt(changed))
        self.assertEqual(canonical(e), before)

    def test_numeric_renaming_equivariance_and_affordability(self):
        e = example()
        a = numerical_features(e)[1]
        renamed = json.loads(canonical(e).replace('"X"', '"speed"').replace('"Y"', '"pressure"'))
        np.testing.assert_array_equal(a, numerical_features(renamed)[1])
        e["state"]["remaining_units"] = 1
        self.assertEqual(numerical_features(e)[0], ("low", None))
        e["state"]["terminal"] = True
        self.assertEqual(numerical_features(e)[0], (None,))

    def test_process_target_risk_episode_and_zero_decision_task_binding(self):
        cli = str(Path(__file__).with_name("llm_intervention_policy_cli.py"))
        command = (sys.executable, "-B", cli, "--method", "target_risk", "--seed", "0", "--draws", "2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def respond(action, request):
                calls.append((action.action_id, request))
                return {"Y": 2.*dict(action.targets)["X"]}
            result = run_task_episode(task=TASK, command=command, timeout_seconds=10., output=root/"run",
                graph=GRAPH, menu=MENU, budget=3, initial_actions=("low",), learner=new_learner(),
                respond=respond, objective="predict Y")
            self.assertEqual(calls, [("low", 0), ("high", 1)])
            self.assertEqual(result["final_state"]["spent_units"], 3)
            payload = json.loads((root/"run/episode/decision-0000.input.json").read_text())
            self.assertEqual(payload["task"], TASK.payload())
            process = json.loads((root/"run/episode/decision-0000.process.json").read_text())
            metadata = json.loads(process["stderr"])
            self.assertEqual(metadata["task_sha256"], digest(TASK.payload()))
            self.assertEqual(metadata["environment_responses"], 0)
            calls.clear()
            result = run_task_episode(task=TASK, command=command, timeout_seconds=10., output=root/"zero",
                graph=GRAPH, menu=MENU, budget=1, initial_actions=("low",), learner=new_learner(),
                respond=respond, objective="predict Y")
            self.assertEqual(result["policy_decisions"], 0)
            self.assertEqual(json.loads((root/"zero/public_task.json").read_text())["task"], TASK.payload())


if __name__ == "__main__":
    unittest.main()

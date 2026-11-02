"""New analytic target-risk fixtures, no scientific worlds or pretrained model."""
from dataclasses import asdict
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np

from llm_intervention_contract import Action, ChargedExecutor, Node, canonical
from llm_intervention_numerical_policy import choose
from llm_intervention_scm import Head, LinearGaussianSCM
from llm_intervention_target_risk import choose_target_risk, integrated_risk, structural_moments


GRAPH = (Node("X", (), "u", (-2., 2.)), Node("Y", ("X",), "u", (-2., 2.)),
         Node("Z", (), "u"))
MENU = (Action("low", "intervene", (("X", 0.),), ("Y",), 1),
        Action("high", "intervene", (("X", 2.),), ("Y",), 2),
        Action("nuisance", "observe", (), ("Z",), 1),
        Action("missing_parent", "observe", (), ("Y",), 1),
        Action("clamped_target", "intervene", (("Y", 1.),), ("Y",), 1))
LAW = dict(target="Y", queries=(MENU[1],), weights=(1.,), scale=1.)


def state():
    learner = LinearGaussianSCM(GRAPH, {"X": 1., "Y": 1., "Z": 1.})
    learner.heads["Z"].covariance[0, 0] = 100.
    with tempfile.TemporaryDirectory() as d:
        with ChargedExecutor(GRAPH, MENU, 4, Path(d)/"j.jsonl", lambda a, r: {}) as ex:
            return ex.state("predict Y at X=2", learner.summaries())


class TargetRiskTests(unittest.TestCase):
    def test_large_offset_retains_small_uncertainty(self):
        model = LinearGaussianSCM((Node("Y", (), "u"),), {"Y": 1.})
        model.heads["Y"].mean[0] = 1e12
        mean, cov = structural_moments(model, Action("q", "observe", (), ("Y",), 1))
        self.assertEqual(mean[0], 1e12)
        self.assertEqual(cov[0, 0], 1.)

    def test_full_shared_ancestor_second_moment_enumeration(self):
        graph = (Node("A", (), "u"), Node("B", ("A",), "u"),
                 Node("C", ("A", "B"), "u"))
        model = LinearGaussianSCM(graph, {n.name: 1. for n in graph})
        means = [np.array([1.]), np.array([.5, 2.]), np.array([1., -1., .5])]
        vectors = [np.array([.5]), np.array([.2, .3]), np.array([.1, -.4, .2])]
        for node, mean, vector in zip(graph, means, vectors):
            model.heads[node.name] = Head(mean, np.outer(vector, vector), 1.)
        query = Action("query", "observe", (), ("C",), 1)
        values = []
        # Independent two-point coefficient distributions share the specified
        # first two moments. Exhaust all combinations as a separate oracle.
        for signs in itertools.product((-1., 1.), repeat=3):
            coefs = [m+s*v for m, s, v in zip(means, signs, vectors)]
            a = coefs[0][0]
            b = coefs[1][0] + coefs[1][1]*a
            c = coefs[2][0] + coefs[2][1]*a + coefs[2][2]*b
            values.append([a, b, c])
        expected = np.array(values)
        mean, cov = structural_moments(model, query)
        np.testing.assert_allclose(mean, expected.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(cov, np.cov(expected, rowvar=False, bias=True), atol=1e-12)
        self.assertGreater(abs(cov[0, 1]), 0)

    def test_regression_gain_per_cost_and_goal_relevance_oracle(self):
        s = state()
        before = canonical(s)
        decision, meta = choose_target_risk(s, **LAW, seed=0, draws=4)
        # Query phi=[1,2], observation phi=[1,2], noise1, priorI:
        # risk5 ->5/6, gain/cost=(25/6)/2=25/12.
        self.assertEqual(decision, {"action_id": "high"})
        self.assertAlmostEqual(meta["risk_before"], 5.)
        self.assertAlmostEqual(meta["risk_after"]["high"], 5/6)
        self.assertAlmostEqual(meta["risk_reduction_per_cost"]["high"], 25/12)
        self.assertAlmostEqual(meta["risk_reduction_per_cost"]["low"], .5)
        for name in ("nuisance", "missing_parent", "clamped_target"):
            self.assertAlmostEqual(meta["risk_reduction_per_cost"][name], 0.)
        self.assertEqual(meta["environment_responses"], 0)
        self.assertEqual(meta["fantasy_responses"], 20)
        self.assertEqual(canonical(s), before)
        # A distinct objective: high nuisance uncertainty attracts local MI.
        self.assertEqual(choose(s, method="local_information", seed=0, draws=4)[0],
                         {"action_id": "nuisance"})

    def test_clamped_target_and_scale_law(self):
        model = LinearGaussianSCM(GRAPH, {"X": 1., "Y": 1., "Z": 1.})
        mean, cov = structural_moments(model, MENU[4])
        self.assertEqual(mean[1], 1.)
        np.testing.assert_array_equal(cov[1], [0., 0., 0.])
        self.assertAlmostEqual(integrated_risk(model, **dict(LAW, scale=2.)), 5/4)
        mix = dict(LAW, queries=(MENU[0], MENU[1]), weights=(.25, .75))
        self.assertAlmostEqual(integrated_risk(model, **mix), 4.)

    def test_order_independence_affordability_and_terminal(self):
        s = state()
        one = choose_target_risk(s, **LAW, seed=17, draws=3)
        s["menu"].reverse()
        two = choose_target_risk(s, **LAW, seed=17, draws=3)
        self.assertEqual(one, two)
        s["remaining_units"] = 1
        self.assertEqual(choose_target_risk(s, **LAW, seed=17, draws=3)[0], {"action_id": "low"})
        s["remaining_units"] = 0
        self.assertEqual(choose_target_risk(s, **LAW, seed=17, draws=3)[0], {"stop": True})

    def test_reject_invalid_public_law_and_planning_limits(self):
        for changes in ({"scale": 0.}, {"weights": (.9,)}, {"target": "unknown"},
                        {"queries": ()}, {"weights": (float("nan"),)}):
            with self.assertRaises(ValueError):
                choose_target_risk(state(), **dict(LAW, **changes), seed=0, draws=2)
        for draws in (1, 257, True):
            with self.assertRaises(ValueError):
                choose_target_risk(state(), **LAW, seed=0, draws=draws)

    def test_negative_fantasy_gains_not_clipped(self):
        # In a chain, realized parameter updates can increase posterior
        # uncertainty downstream. This is a valid noisy planning estimate.
        from unittest.mock import patch
        s = state()
        with patch("llm_intervention_target_risk.integrated_risk",
                   side_effect=[1.] + [2.]*(len(MENU)*2)):
            decision, meta = choose_target_risk(s, **LAW, seed=0, draws=2)
        self.assertEqual(decision, {"action_id": "high"})
        self.assertEqual(meta["risk_reduction_per_cost"]["high"], -.5)
        self.assertEqual(meta["risk_reduction_per_cost"]["low"], -1.)

    def test_downstream_fantasy_variation_and_standard_error_oracle(self):
        from unittest.mock import patch
        graph = (Node("X", (), "u", (-1., 1.)), Node("M", ("X",), "u"),
                 Node("Y", ("M",), "u"))
        menu = tuple(Action(name, "intervene", (("X", 0.),), ("M",), 1) for name in ("a", "b"))
        model = LinearGaussianSCM(graph, {n.name: 1. for n in graph})
        model.heads["Y"].mean[1] = 1.
        with tempfile.TemporaryDirectory() as d:
            with ChargedExecutor(graph, menu, 3, Path(d)/"j.jsonl", lambda a, r: {}) as ex:
                s = ex.state("predict Y", model.summaries())
        query = Action("q", "intervene", (("X", 0.),), ("Y",), 1)
        before = canonical(s)
        def run(payload):
            counts = {"a": 0, "b": 0}
            def fantasy(self, action, rng):
                i = counts[action.action_id]
                counts[action.action_id] += 1
                return {"X": 0., "M": (0., 2., 4.)[i] if action.action_id == "a" else 0., "Y": 0.}
            with patch.object(LinearGaussianSCM, "simulate", fantasy):
                return choose_target_risk(payload, target="Y", queries=(query,), weights=(1.,),
                                          scale=1., seed=0, draws=3)
        decision, meta = run(s)
        # At M=t, posterior M mean=t/2, variance=1/2. Unchanged Y has
        # intercept variance1, slope mean1/variance1. Risk=2+t^2/4.
        # a risks=[2,3,6], mean11/3, sample-mean SE=sqrt(13)/3.
        self.assertEqual(decision, {"action_id": "b"})
        self.assertAlmostEqual(meta["risk_before"], 3.)
        self.assertAlmostEqual(meta["risk_after"]["a"], 11/3)
        self.assertAlmostEqual(meta["risk_reduction_per_cost"]["a"], -2/3)
        self.assertAlmostEqual(meta["monte_carlo_standard_errors"]["a"], np.sqrt(13)/3)
        self.assertAlmostEqual(meta["risk_after"]["b"], 2.)
        self.assertEqual(canonical(s), before)
        s["menu"].reverse()
        self.assertEqual(run(s), (decision, meta))


if __name__ == "__main__":
    unittest.main()

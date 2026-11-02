"""Analytic paid-row, intervention-mask and branch-isolation fixtures only."""
import unittest
import numpy as np

from llm_intervention_contract import Action, Node
from llm_intervention_scm import LinearGaussianSCM


GRAPH = (Node("X", (), "u", (-2., 2.)), Node("M", ("X",), "u", (-3., 3.)),
         Node("Y", ("M",), "u"))


def row(i, x, m, y, targets):
    return {"request_id": i, "values": {"X": x, "M": m, "Y": y}, "targets": targets}


class SCMTests(unittest.TestCase):
    def make(self):
        return LinearGaussianSCM(GRAPH, {n.name: .25 for n in GRAPH})

    def test_posterior_matches_independent_batch_inverse(self):
        scm = self.make()
        data = [row(0, 0., 0., 0., {"X": 0.}), row(1, 1., 1., 2., {"X": 1.}),
                row(2, 2., 4., 8., {"X": 2.})]
        for receipt in data:
            self.assertEqual(scm.update(receipt), ("M", "Y"))
        for node, parent in (("M", "X"), ("Y", "M")):
            design = np.array([[1., d["values"][parent]] for d in data])
            target = np.array([d["values"][node] for d in data])
            expected_cov = np.linalg.inv(np.eye(2) + design.T @ design/.25)
            expected_mean = expected_cov @ design.T @ target/.25
            self.assertGreater(abs(expected_cov[0, 1]), .01)
            np.testing.assert_allclose(scm.heads[node].covariance, expected_cov)
            np.testing.assert_allclose(scm.heads[node].mean, expected_mean)
        self.assertEqual(scm.heads["X"].eligible_rows, 0)

    def test_internal_intervention_excludes_only_its_mechanism(self):
        scm = self.make()
        before = scm.heads["M"].covariance.copy()
        changed = scm.update(row(0, .2, 2., 1., {"M": 2.}))
        self.assertEqual(changed, ("X", "Y"))
        np.testing.assert_array_equal(before, scm.heads["M"].covariance)
        self.assertEqual(scm.heads["Y"].eligible_rows, 1)

    def test_missing_parent_does_not_create_a_training_row(self):
        scm = self.make()
        changed = scm.update({"request_id": 0, "values": {"Y": 3.}, "targets": {}})
        self.assertEqual(changed, ())
        self.assertEqual(scm.heads["Y"].eligible_rows, 0)
        with self.assertRaises(ValueError):
            scm.update({"request_id": 0, "values": {"Y": 3.}, "targets": {}})

    def test_unmeasured_hard_parent_still_supplies_a_regressor(self):
        scm = self.make()
        changed = scm.update({"request_id": 0, "values": {"M": 2.}, "targets": {"X": 1.}})
        self.assertEqual(changed, ("M",))
        np.testing.assert_allclose(scm.heads["M"].mean, np.array([2., 2.])/2.25)
        self.assertEqual(scm.heads["X"].eligible_rows, 0)
        self.assertEqual(scm.heads["Y"].eligible_rows, 0)

    def test_candidate_forks_do_not_contaminate_each_other(self):
        base = self.make()
        base.update(row(0, 0., .5, 1., {"X": 0.}))
        left, right = base.fork(), base.fork()
        original = base.summaries()
        left.update(row(1, 1., 2., 3., {"X": 1.}))
        self.assertEqual(base.summaries(), original)
        self.assertEqual(right.summaries(), original)
        right.update(row(1, -1., -2., -3., {"X": -1.}))
        self.assertNotEqual(left.summaries(), right.summaries())

    def test_prediction_composes_and_simulation_never_updates(self):
        scm = self.make()
        scm.heads["M"].mean[:] = [1., 2.]
        scm.heads["Y"].mean[:] = [2., 3.]
        action = Action("root", "intervene", (("X", 1.),), ("Y",), 1)
        self.assertEqual(scm.predict_mean(action), {"X": 1., "M": 3., "Y": 11.})
        before = scm.summaries()
        a = scm.simulate(action, np.random.default_rng(17))
        b = scm.simulate(action, np.random.default_rng(17))
        self.assertEqual(a, b)
        self.assertEqual(scm.summaries(), before)

    def test_invalid_clamp_rejected_before_any_update(self):
        scm = self.make()
        before = scm.summaries()
        for receipt in (row(0, 1., 2., 3., {"X": 0.}), row(0, 1., 2., float("nan"), {}),
                        row(0, 1., 2., 3., {"Y": 3.})):
            with self.assertRaises(ValueError):
                scm.update(receipt)
            self.assertEqual(scm.summaries(), before)

    def test_controlled_rng_oracle_preserves_covariance_and_parent_flow(self):
        scm = self.make()
        scm.update(row(0, 1., 2., 3., {"X": 1.}))
        class ControlledRng:
            def __init__(self):
                self.parameters = [np.array([2., 3.]), np.array([-1., 2.])]
                self.noises = [.25, -.5]
                self.covariances = []
                self.standard_deviations = []
            def multivariate_normal(self, mean, covariance, check_valid):
                self.covariances.append(covariance.copy())
                return self.parameters.pop(0)
            def normal(self, mean, sd):
                self.standard_deviations.append(sd)
                return self.noises.pop(0)
        rng = ControlledRng()
        action = Action("root", "intervene", (("X", 1.),), ("Y",), 1)
        result = scm.simulate(action, rng)
        self.assertEqual(result, {"X": 1., "M": 5.25, "Y": 9.})
        np.testing.assert_array_equal(rng.covariances[0], scm.heads["M"].covariance)
        np.testing.assert_array_equal(rng.covariances[1], scm.heads["Y"].covariance)
        self.assertEqual(rng.standard_deviations, [.5, .5])


if __name__ == "__main__":
    unittest.main()

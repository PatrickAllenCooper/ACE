"""Finite analytically specified measures for the manuscript risk proposition.

No models, scored study artifacts, responses, or statistical tests are used.
These fixtures verify the stated event inequalities, not deployment assumptions.
"""
import math
import unittest


def finite_quantizer(x):
    # Fixed ties go to the lower level. Endpoint cells clip.
    return min((0.0, 1.0, 2.0), key=lambda level: abs(x-level))


def lattice_quantizer(x):
    # Unit lattice with fixed half-integer ties going upward.
    return math.floor(x+0.5)


class QuantizationRiskTests(unittest.TestCase):
    def test_nonvacuous_general_bound_needs_both_terms(self):
        atoms = [(0.49, 0.51, 0.2), (1.0, 1.6, 0.05), (0.0, 0.05, 0.75)]
        tau = 0.4
        mse = sum(p*(prediction-y)**2 for y, prediction, p in atoms)
        errors = sum(p for y, prediction, p in atoms
                     if finite_quantizer(prediction) != finite_quantizer(y))
        small_margin = sum(p for y, _, p in atoms
                           if min(abs(y-b) for b in (0.5, 1.5)) <= tau)
        self.assertAlmostEqual(errors, 0.25)
        self.assertLess(small_margin, errors)
        self.assertLess(mse/tau**2, errors)
        self.assertLessEqual(errors, small_margin+mse/tau**2)
        self.assertLess(small_margin+mse/tau**2, 1.0)

    def test_margin_decomposition_on_nonuniform_discrete_measure(self):
        atoms = [(0.0, 0.5, 0.1), (0.5, 0.6, 0.2),
                 (1.0, 1.6, 0.3), (2.1, 1.9, 0.4)]
        risk = sum(p*(prediction-y)**2 for y, prediction, p in atoms)
        snapping = sum(p for y, prediction, p in atoms
                       if finite_quantizer(prediction) != finite_quantizer(y))
        for tau in (0.01, 0.1, 0.5, 0.6, 1.0, 2.0):
            small_margin = 0.0
            for y, prediction, p in atoms:
                margin = min(abs(y-b) for b in (0.5, 1.5))
                error_event = finite_quantizer(prediction) != finite_quantizer(y)
                self.assertFalse(error_event and abs(prediction-y) < margin)
                self.assertFalse(error_event and margin > tau and abs(prediction-y) <= tau)
                small_margin += p*(margin <= tau)
            self.assertLessEqual(snapping, min(1.0, small_margin+risk/tau**2)+1e-14)

    def test_positive_margin_bound_is_sharp_including_boundary_ties(self):
        # At y=1, the prediction .5 ties downward and misclassifies; at
        # y=0, prediction -.5 stays in the clipped endpoint cell.
        atoms = [(1.0, 0.5, 0.25), (0.0, 0.0, 0.75)]
        mse = sum(p*(prediction-y)**2 for y, prediction, p in atoms)
        errors = sum(p for y, prediction, p in atoms
                     if finite_quantizer(prediction) != finite_quantizer(y))
        self.assertEqual(errors, 0.25)
        self.assertEqual(mse/(0.5**2), errors)
        self.assertEqual(finite_quantizer(-0.5), 0.0)

    def test_extended_lattice_retains_endpoint_crossings(self):
        atoms = [(0.0, -0.6, 0.2), (1.0, 1.5, 0.3), (2.0, 2.0, 0.5)]
        mse = sum(p*(prediction-y)**2 for y, prediction, p in atoms)
        errors = sum(p for y, prediction, p in atoms
                     if lattice_quantizer(prediction) != lattice_quantizer(y))
        self.assertEqual(errors, 0.5)
        self.assertLessEqual(errors, mse/(0.5**2))
        self.assertNotEqual(lattice_quantizer(-0.6), finite_quantizer(-0.6))

    def test_small_mse_with_high_risk_and_reversed_model_ordering(self):
        # Continuously valued target near a boundary: every prediction crosses
        # it while MSE tends to zero. Fixed-margin assumptions are essential.
        for epsilon in (1e-2, 1e-4, 1e-6):
            y = 0.5-epsilon
            crossing, correct = 0.5+epsilon, 0.5-4*epsilon
            self.assertNotEqual(finite_quantizer(crossing), finite_quantizer(y))
            self.assertEqual(finite_quantizer(correct), finite_quantizer(y))
            self.assertLess((crossing-y)**2, (correct-y)**2)
            self.assertAlmostEqual((crossing-y)**2, 4*epsilon**2)


if __name__ == '__main__':
    unittest.main()

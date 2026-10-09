"""New artificial contract examples only; no scientific worlds or PFN weights."""
import math
import unittest
import foundation_retention_selection as s
from foundation_flexible_control import RBFControl


def linear(a):
    return lambda x: tuple(a*v for v in x)


def histories():
    fit = tuple(s.Row((-1., 0., 1.)[i % 3], (-1., 0., 1.)[i % 3], 0., c)
                for i, c in enumerate(('none',)*6 + ('X',)*9 + ('M',)*9))
    calibration = tuple(s.Row(.25, .5, 1., c)
                        for c in ('none',)*2 + ('X',)*3 + ('M',)*3)
    return fit, calibration


def heads():
    return tuple(dict(zip(s.ORDER, (linear(1), linear(2), linear(3), linear(4))))
                 for _ in range(2))


class Retention(unittest.TestCase):
    def test_gate_partition_boundaries_and_exact_outside_identity(self):
        calls = []
        def old(x): calls.append(('old', x)); return tuple(10+v for v in x)
        def new(x): calls.append(('new', x)); return tuple(20+v for v in x)
        gate = s.GatedHead(old, new, s.Interval(-1., 1.))
        self.assertEqual(gate((2., -1., 0., -2., 1.)), (12., 19., 20., 8., 21.))
        self.assertEqual(calls, [('new', (-1., 0., 1.)), ('old', (2., -2.))])
        def forbidden(x): raise AssertionError('unused updated branch evaluated')
        self.assertEqual(s.GatedHead(old, forbidden, s.Interval(0., 0.))((1.,)), (11.,))

    def test_local_constraints_predicted_parent_and_retained_feasibility(self):
        fit, cal = histories(); candidates = heads()
        chosen = s.select(fit, cal, candidates)
        self.assertEqual(chosen.selected, ('grammar', 'grammar'))
        self.assertEqual(chosen.admitted, (('retained', 'grammar'),)*2)
        self.assertEqual(chosen.composed_mse, 0.)
        self.assertEqual(len(chosen.composed_errors), 4)
        self.assertEqual(chosen.calibration_inside, (5, 8))
        self.assertEqual(s.forecast(chosen, candidates, (.25,)), (1.,))
        # x=.75 gives updated M=1.5, so the Y gate retains identity at1.5.
        self.assertEqual(s.forecast(chosen, candidates, (.75,)), (1.5,))
        scores = {(node, name): error for node, name, error in chosen.local_errors}
        for node, name in zip(('M', 'Y'), chosen.selected):
            self.assertLessEqual(scores[node, name], scores[node, 'retained'])
        self.assertLessEqual(chosen.composed_mse, dict(chosen.composed_errors)[('retained', 'retained')])

    def test_clamped_m_excluded_locally_and_in_composition_but_y_uses_all_rows(self):
        fit, cal = histories()
        changed = cal[:5] + tuple(s.Row(999., .5, 100., 'M') for _ in range(3))
        a = s.select(fit, cal, heads(), mode='raw')
        b = s.select(fit, changed, heads(), mode='raw')
        self.assertEqual(a.composed_errors, b.composed_errors)
        self.assertEqual(a.local_errors[:4], b.local_errors[:4])
        self.assertAlmostEqual(dict(((n, k), v) for n, k, v in b.local_errors)['Y', 'grammar'],
                               3*99**2/8)

    def test_ties_and_declared_ablation_sets(self):
        fit, cal = histories()
        equal = tuple({n: linear(1) for n in s.ORDER} for _ in range(2))
        for mode in s.MODES:
            chosen = s.select(fit, cal, equal, mode=mode)
            self.assertEqual(chosen.selected, ('retained', 'retained'))
            self.assertEqual(len(chosen.composed_errors), 1 if mode in ('local', 'combined') else 16)
        raw = s.select(fit, cal, heads(), mode='raw')
        # Two zero-loss one-replacement pairs tie: retained/pfn precedes pfn/retained.
        self.assertEqual(raw.selected, ('retained', 'pfn'))
        restricted = tuple({k: v for k, v in node.items() if k != 'pfn'} for node in heads())
        self.assertEqual(len(s.select(fit, cal, restricted, mode='raw', include_pfn=False).composed_errors), 9)
        with self.assertRaises(ValueError): s.select(fit, cal, heads(), include_pfn=False)

    def test_intervals_use_eligible_fit_parents_only(self):
        fit, cal = histories()
        modified = fit[:15] + tuple(s.Row(999., 2., 0., 'M') for _ in range(9))
        cal = tuple(s.Row(5., 10., 20., r.clamp) for r in cal)
        intervals, local, root = s.training_layout(modified, cal)
        self.assertEqual(intervals, (s.Interval(-1., 1.), s.Interval(-1., 2.)))
        chosen = s.select(modified, cal, heads())
        self.assertEqual(chosen.calibration_inside, (0, 0))
        self.assertEqual(chosen.selected, ('retained', 'retained'))

    def test_malformed_and_failed_candidates_are_not_dropped(self):
        fit, cal = histories()
        for bad in (lambda x: (0.,), lambda x: tuple(float('nan') for _ in x),
                    lambda x: tuple(1e308 for _ in x)):
            candidates = heads(); candidates[0]['pfn'] = bad
            with self.assertRaises(ValueError): s.select(fit, cal, candidates)
        def broken(x): raise RuntimeError('candidate failed')
        candidates = heads(); candidates[1]['pfn'] = broken
        with self.assertRaisesRegex(RuntimeError, 'candidate failed'): s.select(fit, cal, candidates)
        with self.assertRaises(ValueError): s.select(fit[:-1], cal, heads())
        with self.assertRaises(ValueError): s.select(fit, cal[::-1], heads())
        with self.assertRaises(ValueError): s.select(fit, cal, heads(), mode='best')
        with self.assertRaises(ValueError): s.Interval(1., -1.)
        with self.assertRaises(ValueError): s.Interval(0., float('inf'))
        with self.assertRaises(ValueError): s.Interval.from_parents(())


class FlexibleControl(unittest.TestCase):
    def test_two_point_closed_form_and_fit_only_normalization(self):
        control = RBFControl().fit((-1., 1.), (-1., 1.))
        a = (1-math.exp(-4))/(1.01-math.exp(-4))
        actual = control((-1., 1.))
        self.assertAlmostEqual(actual[0], -a, places=12)
        self.assertAlmostEqual(actual[1], a, places=12)
        before = (control.center_, control.scale_, control.target_center_)
        control((1e6,))
        self.assertEqual((control.center_, control.scale_, control.target_center_), before)
        # Affine parent transforms and target translation preserve this rule.
        shifted = RBFControl().fit((8., 12.), (4., 6.))
        for expected, observed in zip(actual, shifted((8., 12.))):
            self.assertAlmostEqual(expected+5, observed, places=12)

    def test_constant_parent_and_invalid_fit_paths(self):
        control = RBFControl().fit((3., 3.), (7., 7.))
        self.assertEqual(control.scale_, 1.)
        self.assertEqual(control((3., 9.)), (7., 7.))
        with self.assertRaises(ValueError): control.fit((1., 2.), (1., 2.))
        with self.assertRaises(ValueError): RBFControl()((1.,))
        for x, y in (((1.,), (1.,)), ((1., 2.), (1.,)),
                     ((1., float('nan')), (1., 2.))):
            with self.assertRaises(ValueError): RBFControl().fit(x, y)


if __name__ == '__main__':
    unittest.main(verbosity=2)

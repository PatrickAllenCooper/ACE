import math
import unittest
from delivery_prospective_analysis import analyze


class ProspectiveAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {size: [str(i) for i in range(20)] for size in (5, 30)}
        # Unequal control scales make averaging errors disagree with averaging
        # within-system log ratios: sqrt(.1*.9)=.3 versus 90.1/101=.89208.
        self.rows = [{'graph_size': size, 'system_id': str(i), 'history': history,
                      'arm': arm, 'init': 0, 'status': 'complete', 'nmse': error}
                     for size in (5, 30) for i in range(20)
                     for history, d, c in [('balanced_varied_value', .1, 1.), ('matched_random', 90., 100.)]
                     for arm, error in [('delivery', d), ('online', c), ('simpler', c)]]

    def test_history_log_aggregation_and_system_unit(self):
        result = analyze(self.rows, self.manifest)
        self.assertEqual(result['n_primary_cells'], 240)
        self.assertEqual(len(result['contrasts']), 4)
        for contrast in result['contrasts']:
            self.assertEqual(contrast['n_systems'], 20)
            self.assertAlmostEqual(contrast['ratio'], .3)
            self.assertTrue(contrast['superiority'])

    def test_incomplete_duplicate_and_failed_are_rejected(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[0]],
                     [{**self.rows[0], 'status': 'failed'}, *self.rows[1:]]]:
            with self.assertRaises(ValueError):
                analyze(rows, self.manifest)

    def test_sensitivity_init_cannot_select_primary(self):
        with self.assertRaises(ValueError):
            analyze([{**self.rows[0], 'init': 1}, *self.rows[1:]], self.manifest)

    def test_floor_activations_and_no_zero_error_superiority(self):
        result = analyze([{**r, 'nmse': 0.} for r in self.rows], self.manifest)
        for contrast in result['contrasts']:
            self.assertEqual(contrast['floor_activations']['delivery'], 40)
            self.assertEqual(contrast['floor_activations'][contrast['control']], 40)
            self.assertEqual(contrast['ratio'], 1.)
            self.assertFalse(contrast['superiority'])

    def test_four_test_holm_and_ci_against_manual_system_contrast(self):
        from delivery_theory import paired_log_ratio, holm
        rows = [{**r, 'nmse': r['nmse'] * math.exp(int(r['system_id'])/20)
                 if r['arm'] == 'delivery' else r['nmse']} for r in self.rows]
        result = analyze(rows, self.manifest)
        manual = paired_log_ratio([.3 * math.exp(i/20) for i in range(20)], [1.]*20)
        for contrast in result['contrasts']:
            self.assertAlmostEqual(contrast['ratio'], manual['ratio'])
            self.assertAlmostEqual(contrast['p_raw'], manual['p_raw'])
            self.assertAlmostEqual(contrast['ci95_marginal'][1], manual['ci95'][1])
            self.assertAlmostEqual(contrast['p_holm_four_tests'], holm([manual['p_raw']]*4)[0])


if __name__ == '__main__':
    unittest.main()

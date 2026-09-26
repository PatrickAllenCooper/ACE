"""Scientific invariants and artifact integrity for the staged research pilot."""
import json
from pathlib import Path
import tempfile
import unittest

from baselines import GroundTruthSCM, NonLeafCoveragePolicy, NonLeafRandomPolicy
from scripts.research.agenda_runner import design_experiment, prior_experiment, write_results
from scripts.research.validate_cell import valid


class ResearchAgendaTests(unittest.TestCase):
    def test_joint_actions_identify_interaction_when_single_actions_cannot(self):
        rows = design_experiment(seed=42, budget=400, background_sd=0, actuator_penalty=0)
        final = {row['method']: row for row in rows if row['budget'] == 400}
        self.assertAlmostEqual(final['design_single']['interaction_error'], 0.9**2)
        self.assertLess(final['design_joint']['interaction_error'], 0.05)
        self.assertLess(final['random_joint']['interaction_error'], 0.05)

    def test_action_cost_never_exceeds_requested_budget(self):
        rows = design_experiment(seed=123, budget=400, background_sd=0.15, actuator_penalty=4)
        self.assertTrue(all(row['budget'] <= 400 for row in rows))
        self.assertTrue(any(row['method'] == 'design_joint' for row in rows))

    def test_graph_matched_controls_exclude_leaves(self):
        scm = GroundTruthSCM()
        eligible = {'X1', 'X2', 'X4'}
        coverage = NonLeafCoveragePolicy(scm.nodes, scm.graph)
        random = NonLeafRandomPolicy(scm.nodes, scm.graph)
        self.assertEqual(set(coverage.nodes), eligible)
        self.assertEqual(set(random.nodes), eligible)
        self.assertEqual({coverage.select_intervention(None)[0] for _ in range(33)}, eligible)

    def test_receipt_detects_truncated_or_changed_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rows = prior_experiment(42)
            write_results(rows, d, {'track': 'prior', 'seed': 42})
            self.assertTrue(valid(d, 'agenda')[0])
            (d / 'metrics.csv').write_text('tampered\n')
            self.assertFalse(valid(d, 'agenda')[0])


if __name__ == '__main__':
    unittest.main()

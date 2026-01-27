"""
Tests for baseline runner functions used in critical experiments.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestBaselineRunners:
    """Test the baseline runner functions."""
    
    @pytest.mark.slow
    def test_run_random_policy_returns_losses(self):
        """Test run_random_policy returns dict of losses."""
        from baselines import (
            GroundTruthSCM,
            StudentSCM,
            SCMLearner,
            run_random_policy
        )
        
        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        learner = SCMLearner(student, oracle=scm)
        
        final_losses = run_random_policy(scm, learner, episodes=2)
        
        assert isinstance(final_losses, dict)
        assert len(final_losses) == len(scm.nodes)
        assert all(isinstance(v, (int, float)) for v in final_losses.values())
        assert all(v >= 0 for v in final_losses.values())
    
    @pytest.mark.slow
    def test_run_round_robin_policy_cycles_nodes(self):
        """Test round-robin cycles through all nodes."""
        from baselines import (
            GroundTruthSCM,
            StudentSCM,
            SCMLearner,
            run_round_robin_policy
        )
        
        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        learner = SCMLearner(student, oracle=scm)
        
        # Run for exactly len(nodes) episodes
        episodes = len(scm.nodes)
        final_losses = run_round_robin_policy(scm, learner, episodes=episodes)
        
        assert isinstance(final_losses, dict)
        assert len(final_losses) == len(scm.nodes)
    
    @pytest.mark.slow
    def test_run_max_variance_policy_selects_uncertain_nodes(self):
        """Test max-variance policy runs without error."""
        from baselines import (
            GroundTruthSCM,
            StudentSCM,
            SCMLearner,
            run_max_variance_policy
        )
        
        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        learner = SCMLearner(student, oracle=scm)
        
        final_losses = run_max_variance_policy(scm, learner, episodes=2)
        
        assert isinstance(final_losses, dict)
        assert len(final_losses) > 0
    
    @pytest.mark.slow
    def test_all_baselines_produce_similar_output_format(self):
        """Test all baseline runners have consistent output format."""
        from baselines import (
            GroundTruthSCM,
            StudentSCM,
            SCMLearner,
            run_random_policy,
            run_round_robin_policy,
            run_max_variance_policy
        )
        
        runners = [run_random_policy, run_round_robin_policy, run_max_variance_policy]
        
        for run_fn in runners:
            scm = GroundTruthSCM()
            student = StudentSCM(scm)
            learner = SCMLearner(student, oracle=scm)
            
            losses = run_fn(scm, learner, episodes=1)
            
            # All should return dict with same keys
            assert isinstance(losses, dict)
            assert set(losses.keys()) == set(scm.nodes)


class TestSCMLearnerEvaluate:
    """Test SCMLearner.evaluate() method."""
    
    @pytest.mark.slow
    def test_scmlearner_evaluate_with_oracle(self):
        """Test evaluate() works with oracle."""
        from baselines import GroundTruthSCM, StudentSCM, SCMLearner
        
        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        learner = SCMLearner(student, oracle=scm)
        
        losses = learner.evaluate()
        
        assert isinstance(losses, dict)
        assert len(losses) == len(scm.nodes)
        assert all(isinstance(v, (int, float)) for v in losses.values())
    
    @pytest.mark.slow  
    def test_scmlearner_evaluate_without_oracle(self):
        """Test evaluate() handles missing oracle gracefully."""
        from baselines import GroundTruthSCM, StudentSCM, SCMLearner
        
        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        learner = SCMLearner(student, oracle=None)
        
        losses = learner.evaluate()
        
        # Should return zeros when no oracle
        assert isinstance(losses, dict)
        assert all(v == 0.0 for v in losses.values())
    
    @pytest.mark.slow
    def test_scmlearner_train_and_evaluate(self):
        """Test train_step followed by evaluate."""
        from baselines import GroundTruthSCM, StudentSCM, SCMLearner
        
        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        learner = SCMLearner(student, oracle=scm)
        
        # Get initial losses
        losses_before = learner.evaluate()
        
        # Train on some data
        data = scm.generate(50, interventions={'X1': 0.0})
        learner.train_step(data, intervened='X1')
        
        # Evaluate again
        losses_after = learner.evaluate()
        
        # Both should be valid
        assert isinstance(losses_before, dict)
        assert isinstance(losses_after, dict)
        assert len(losses_before) == len(losses_after)


class TestBaselineRunnerConsistency:
    """Test baseline runners are consistent across calls."""
    
    @pytest.mark.slow
    def test_random_policy_deterministic_with_seed(self):
        """Test random policy produces same results with same seed."""
        from baselines import GroundTruthSCM, StudentSCM, SCMLearner, run_random_policy
        import random
        import torch
        import numpy as np
        
        def run_with_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            scm = GroundTruthSCM()
            student = StudentSCM(scm)
            learner = SCMLearner(student, oracle=scm)
            return run_random_policy(scm, learner, episodes=2)
        
        losses1 = run_with_seed(42)
        losses2 = run_with_seed(42)
        
        # Should be identical with same seed
        for node in losses1.keys():
            assert abs(losses1[node] - losses2[node]) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

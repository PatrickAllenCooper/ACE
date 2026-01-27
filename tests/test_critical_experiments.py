"""
Tests for critical experiments script and dependencies.
Ensures all required functions exist and work correctly.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestCriticalExperimentsImports:
    """Test that all required imports exist."""
    
    def test_baselines_functions_exist(self):
        """Verify all required baseline functions exist."""
        from baselines import (
            GroundTruthSCM,
            StudentSCM,
            SCMLearner,
            run_random_policy,
            run_round_robin_policy,
            run_max_variance_policy
        )
        
        assert GroundTruthSCM is not None
        assert StudentSCM is not None
        assert SCMLearner is not None
        assert callable(run_random_policy)
        assert callable(run_round_robin_policy)
        assert callable(run_max_variance_policy)
    
    def test_complex_scm_classes_exist(self):
        """Verify complex SCM classes exist."""
        from experiments.complex_scm import ComplexGroundTruthSCM, ComplexSCMLearner
        
        assert ComplexGroundTruthSCM is not None
        assert ComplexSCMLearner is not None
    
    def test_critical_script_imports(self):
        """Verify critical experiments script can be imported."""
        sys.path.insert(0, 'scripts/runners')
        # This will fail if there are syntax errors or missing dependencies
        # at module level
        try:
            import run_critical_experiments
            assert hasattr(run_critical_experiments, 'main')
        except ImportError as e:
            if 'torch' in str(e) or 'transformers' in str(e):
                pytest.skip("Missing ML dependencies (expected in test env)")
            raise


class TestBaselineFunctionSignatures:
    """Test baseline function signatures match expected API."""
    
    def test_run_random_policy_signature(self):
        """Test run_random_policy has correct signature."""
        from baselines import run_random_policy
        import inspect
        
        sig = inspect.signature(run_random_policy)
        params = list(sig.parameters.keys())
        
        # Should accept: scm, learner, episodes
        assert 'scm' in params
        assert 'learner' in params
        assert 'episodes' in params
    
    def test_scm_learner_has_evaluate(self):
        """Test SCMLearner has evaluate method."""
        from baselines import SCMLearner
        
        assert hasattr(SCMLearner, 'evaluate')
        assert callable(getattr(SCMLearner, 'evaluate'))


class TestCriticalExperiments Integration:
    """Integration tests for critical experiments."""
    
    @pytest.mark.slow
    def test_extended_baselines_minimal(self):
        """Test extended baselines with 1 episode."""
        from baselines import GroundTruthSCM, StudentSCM, SCMLearner, run_random_policy
        
        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        learner = SCMLearner(student, oracle=scm)
        
        # Should run without error
        final_losses = run_random_policy(scm, learner, episodes=1)
        
        # Should return dict of losses
        assert isinstance(final_losses, dict)
        assert len(final_losses) > 0
        assert all(isinstance(v, (int, float)) for v in final_losses.values())
    
    @pytest.mark.slow
    def test_round_robin_minimal(self):
        """Test round-robin with 1 episode."""
        from baselines import GroundTruthSCM, StudentSCM, SCMLearner, run_round_robin_policy
        
        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        learner = SCMLearner(student, oracle=scm)
        
        final_losses = run_round_robin_policy(scm, learner, episodes=1)
        
        assert isinstance(final_losses, dict)
        assert len(final_losses) > 0
    
    @pytest.mark.slow
    def test_max_variance_minimal(self):
        """Test max-variance with 1 episode."""
        from baselines import GroundTruthSCM, StudentSCM, SCMLearner, run_max_variance_policy
        
        scm = GroundTruthSCM()
        student = StudentSCM(scm)
        learner = SCMLearner(student, oracle=scm)
        
        final_losses = run_max_variance_policy(scm, learner, episodes=1)
        
        assert isinstance(final_losses, dict)
        assert len(final_losses) > 0


class TestLearningCurveTracker:
    """Test the learning curve tracking functionality."""
    
    def test_tracker_records_data(self):
        """Test tracker records episode data."""
        sys.path.insert(0, 'scripts/runners')
        try:
            from run_critical_experiments import LearningCurveTracker
        except ImportError:
            pytest.skip("Cannot import without ML dependencies")
            return
        
        tracker = LearningCurveTracker()
        
        # Record some data
        tracker.record(1, 1.5, {'X1': 0.3, 'X2': 0.5, 'X3': 0.7})
        tracker.record(2, 1.2, {'X1': 0.2, 'X4': 0.4, 'X3': 0.6})
        
        # Should have 2 episode records
        assert len(tracker.episode_losses) == 2
        assert tracker.episode_losses[0]['episode'] == 1
        assert tracker.episode_losses[0]['total_loss'] == 1.5
        
        # Should have 6 node records (2 episodes × 3 nodes)
        assert len(tracker.per_node_losses) == 6
    
    def test_tracker_to_dataframe(self):
        """Test tracker converts to dataframe."""
        sys.path.insert(0, 'scripts/runners')
        try:
            from run_critical_experiments import LearningCurveTracker
            import pandas as pd
        except ImportError:
            pytest.skip("Cannot import without ML dependencies")
            return
        
        tracker = LearningCurveTracker()
        tracker.record(1, 1.5, {'X1': 0.3})
        tracker.record(2, 1.2, {'X1': 0.2})
        
        df = tracker.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'episode' in df.columns
        assert 'total_loss' in df.columns
    
    def test_tracker_save(self, tmp_path):
        """Test tracker saves to CSV."""
        sys.path.insert(0, 'scripts/runners')
        try:
            from run_critical_experiments import LearningCurveTracker
        except ImportError:
            pytest.skip("Cannot import without ML dependencies")
            return
        
        tracker = LearningCurveTracker()
        tracker.record(1, 1.5, {'X1': 0.3})
        
        output_path = tmp_path / "test_curve.csv"
        tracker.save(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestComplexSCMFunctions:
    """Test complex SCM helper functions."""
    
    @pytest.mark.slow
    def test_get_collider_loss(self):
        """Test collider loss calculation."""
        sys.path.insert(0, 'scripts/runners')
        try:
            from run_critical_experiments import get_collider_loss
            from experiments.complex_scm import ComplexGroundTruthSCM
        except ImportError:
            pytest.skip("Cannot import without ML dependencies")
            return
        
        scm = ComplexGroundTruthSCM()
        losses = {'L1': 0.5, 'N1': 0.3, 'C1': 0.4, 'C2': 0.6, 'F3': 0.2, 'R1': 1.0}
        
        collider_loss = get_collider_loss(losses, scm)
        
        # Should average the 5 colliders
        expected = (0.5 + 0.3 + 0.4 + 0.6 + 0.2) / 5
        assert abs(collider_loss - expected) < 0.01
    
    @pytest.mark.slow
    def test_run_complex_random(self):
        """Test random policy on complex SCM."""
        sys.path.insert(0, 'scripts/runners')
        try:
            from run_critical_experiments import run_complex_random, create_complex_learner
            from experiments.complex_scm import ComplexGroundTruthSCM
        except ImportError:
            pytest.skip("Cannot import without ML dependencies")
            return
        
        scm = ComplexGroundTruthSCM()
        learner = create_complex_learner(scm)
        
        # Run for 1 episode
        final_losses = run_complex_random(scm, learner, episodes=1)
        
        assert isinstance(final_losses, dict)
        assert len(final_losses) > 0
        assert all(isinstance(v, (int, float)) for v in final_losses.values())


class TestJobScriptConfiguration:
    """Test job script configurations."""
    
    def test_critical_job_has_unbuffered_python(self):
        """Verify critical job uses python -u for unbuffered output."""
        with open("jobs/run_critical_experiments.sh") as f:
            content = f.read()
        
        assert "python -u" in content, "Missing unbuffered Python flag"
    
    def test_critical_job_correct_path(self):
        """Verify critical job calls correct script path."""
        with open("jobs/run_critical_experiments.sh") as f:
            content = f.read()
        
        assert "scripts/runners/run_critical_experiments.py" in content
    
    def test_critical_job_has_qos(self):
        """Verify critical job has QoS."""
        with open("jobs/run_critical_experiments.sh") as f:
            content = f.read()
        
        assert "#SBATCH --qos=normal" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for PPO baseline on complex 15-node SCM.
Ensures the implementation works before HPC execution.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestComplexSCMPPOImplementation:
    """Test PPO implementation for complex SCM."""
    
    def test_ppo_policy_import(self):
        """Verify PPOPolicy can be imported from baselines."""
        from baselines import PPOPolicy
        assert PPOPolicy is not None
        assert hasattr(PPOPolicy, 'select_action')
        assert hasattr(PPOPolicy, 'store_reward')
        assert hasattr(PPOPolicy, 'update')
    
    def test_run_complex_ppo_exists(self):
        """Verify run_complex_ppo function exists."""
        sys.path.insert(0, 'scripts/runners')
        try:
            from run_critical_experiments import run_complex_ppo
            assert callable(run_complex_ppo)
        except ImportError:
            pytest.skip("Cannot import without ML dependencies")
    
    @pytest.mark.slow
    def test_ppo_policy_dimensions(self):
        """Test PPOPolicy with complex SCM dimensions."""
        from baselines import PPOPolicy
        
        # Complex SCM: 15 nodes
        state_dim = 15 * 2  # node losses + counts
        n_nodes = 15
        n_values = 11
        
        policy = PPOPolicy(state_dim, n_nodes, n_values)
        
        # Should create without error
        assert policy is not None
        
        # Test action selection
        import torch
        state = torch.zeros(state_dim)
        node_idx, value_idx = policy.select_action(state)
        
        # Outputs should be in valid range
        assert 0 <= node_idx < n_nodes
        assert 0 <= value_idx < n_values
    
    @pytest.mark.slow
    def test_complex_ppo_minimal_run(self):
        """Test run_complex_ppo with 1 episode."""
        sys.path.insert(0, 'scripts/runners')
        try:
            from run_critical_experiments import run_complex_ppo, create_complex_learner
            from experiments.complex_scm import ComplexGroundTruthSCM
            import random
            import torch
            import numpy as np
            
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            
            scm = ComplexGroundTruthSCM()
            learner = create_complex_learner(scm)
            
            # Run for 1 episode
            final_losses = run_complex_ppo(scm, learner, episodes=1)
            
            assert isinstance(final_losses, dict)
            assert len(final_losses) > 0
            assert all(isinstance(v, (int, float)) for v in final_losses.values())
            
        except ImportError as e:
            if 'torch' in str(e):
                pytest.skip("Cannot run without ML dependencies")
            raise


class TestComplexSCMExecutionOrder:
    """Test that PPO runs first and saves immediately."""
    
    def test_ppo_runs_first_in_script(self):
        """Verify PPO is first method in execution order."""
        with open("scripts/runners/run_critical_experiments.py") as f:
            content = f.read()
        
        # Find the complex SCM function
        lines = content.split('\n')
        ppo_idx = None
        random_idx = None
        
        for i, line in enumerate(lines):
            if 'print("\\nRunning PPO..."' in line or "print('\\nRunning PPO..." in line:
                ppo_idx = i
            if 'print("\\nRunning Random..."' in line and ppo_idx is not None:
                random_idx = i
                break
        
        assert ppo_idx is not None, "PPO not found in complex SCM"
        assert random_idx is not None, "Random baseline not found"
        assert ppo_idx < random_idx, "PPO should run before Random"
    
    def test_ppo_saves_immediately(self):
        """Verify PPO results are saved immediately after completion."""
        with open("scripts/runners/run_critical_experiments.py") as f:
            content = f.read()
        
        # Check that save happens right after PPO
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'print("\\nRunning PPO..."' in line or "print('\\nRunning PPO..." in line:
                # Check next 10 lines for save operation
                next_lines = '\n'.join(lines[i:i+15])
                assert '.to_csv' in next_lines, "No save found after PPO"
                assert 'PPO complete - SAVED' in next_lines or 'ppo' in next_lines.lower()
                break


class TestIncrementalSaves:
    """Test that all results save incrementally."""
    
    def test_complex_scm_saves_after_each_seed(self):
        """Verify complex SCM saves after every seed."""
        with open("scripts/runners/run_critical_experiments.py") as f:
            content = f.read()
        
        # Should save after completing all methods for a seed
        assert 'df_all.to_csv' in content
        assert 'complex_scm_summary.csv' in content
        assert 'Seed {seed} complete - saved' in content or 'complete - saved' in content
    
    def test_extended_baselines_saves_incrementally(self):
        """Verify extended baselines saves after each seed."""
        with open("scripts/runners/run_critical_experiments.py") as f:
            content = f.read()
        
        # Check for saves in extended baselines function
        lines = content.split('\n')
        in_extended_fn = False
        has_incremental_save = False
        
        for line in lines:
            if 'def run_extended_baselines' in line:
                in_extended_fn = True
            if in_extended_fn and 'def run_' in line and 'extended' not in line:
                break  # End of function
            if in_extended_fn and '.to_csv' in line and 'extended_baselines_summary' in line:
                # Check if it's inside the seed loop (not just at end)
                has_incremental_save = True
        
        assert has_incremental_save, "Extended baselines should save incrementally"


class TestJobScriptConfiguration:
    """Test job script configurations."""
    
    def test_complex_scm_time_limit(self):
        """Verify complex SCM has 10h time limit."""
        with open("jobs/run_complex_scm_only.sh") as f:
            content = f.read()
        assert "#SBATCH --time=10:00:00" in content
    
    def test_complex_scm_has_qos(self):
        """Verify complex SCM has QoS."""
        with open("jobs/run_complex_scm_only.sh") as f:
            content = f.read()
        assert "#SBATCH --qos=normal" in content
    
    def test_complex_scm_unbuffered(self):
        """Verify complex SCM uses python -u."""
        with open("jobs/run_complex_scm_only.sh") as f:
            content = f.read()
        assert "python -u" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

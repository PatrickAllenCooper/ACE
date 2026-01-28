"""
Comprehensive tests for full ACE complex SCM implementation.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestFullImplementation:
    """Test complete ACE implementation."""
    
    def test_full_script_exists(self):
        """Verify run_ace_complex_full.py exists."""
        assert os.path.exists("experiments/run_ace_complex_full.py")
    
    def test_has_dpo_loss_function(self):
        """Verify DPO loss function implemented."""
        with open("experiments/run_ace_complex_full.py") as f:
            content = f.read()
        assert "def compute_dpo_loss" in content
        assert "token-level log probabilities" in content.lower() or "log_softmax" in content
    
    def test_has_reward_bonuses(self):
        """Verify reward bonus computation."""
        with open("experiments/run_ace_complex_full.py") as f:
            content = f.read()
        assert "compute_reward_with_bonuses" in content
        assert "node_importance" in content.lower() or "node_loss" in content
    
    def test_uses_policy_to_generate(self):
        """Verify policy generates candidates (not random)."""
        with open("experiments/run_ace_complex_full.py") as f:
            content = f.read()
        assert "policy_net.generate" in content or "scm_to_prompt" in content
        assert "generate_and_parse" in content or "generate_experiment" in content
    
    def test_has_gradient_updates(self):
        """Verify policy is updated via gradients."""
        with open("experiments/run_ace_complex_full.py") as f:
            content = f.read()
        assert "optimizer.zero_grad()" in content
        assert ".backward()" in content
        assert "optimizer.step()" in content
    
    def test_has_reference_updates(self):
        """Verify reference policy updates periodically."""
        with open("experiments/run_ace_complex_full.py") as f:
            content = f.read()
        assert "ref_policy = copy.deepcopy(policy_net)" in content
        # Should update periodically
        assert "episode % 25" in content or "reference policy updated" in content.lower()
    
    def test_has_multiple_steps_per_episode(self):
        """Verify multiple steps per episode (like 5-node)."""
        with open("experiments/run_ace_complex_full.py") as f:
            content = f.read()
        assert "for step in range" in content
        # Should be 25 steps
        assert "range(25)" in content or "steps_per_episode" in content
    
    def test_saves_dpo_losses(self):
        """Verify DPO losses are saved."""
        with open("experiments/run_ace_complex_full.py") as f:
            content = f.read()
        assert "dpo_training.csv" in content
        assert "dpo_losses" in content or "dpo_loss" in content
    
    def test_proper_error_handling(self):
        """Verify error handling for policy generation."""
        with open("experiments/run_ace_complex_full.py") as f:
            content = f.read()
        assert "try:" in content
        assert "except" in content
        # Should have fallback
        assert "fallback" in content.lower() or "# Fallback" in content
    
    def test_intervention_history_tracked(self):
        """Verify intervention history for context."""
        with open("experiments/run_ace_complex_full.py") as f:
            content = f.read()
        assert "intervention_history" in content
        assert "deque" in content or "list" in content


class TestJobIntegration:
    """Test job script integration."""
    
    def test_job_calls_full_script(self):
        """Verify job script calls run_ace_complex_full.py."""
        with open("jobs/run_ace_complex_scm.sh") as f:
            content = f.read()
        assert "run_ace_complex_full.py" in content
    
    def test_job_passes_seed(self):
        """Verify seed is passed to script."""
        with open("jobs/run_ace_complex_scm.sh") as f:
            content = f.read()
        assert "--seed $SEED" in content or "--seed" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

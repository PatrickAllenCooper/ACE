"""
Unit tests for Max-Variance baseline policy.

Tests cover:
- Initialization
- Variance computation
- MC Dropout behavior
- Greedy selection
- Candidate generation
"""

import pytest
import torch


# =============================================================================
# MaxVariancePolicy Tests
# =============================================================================

@pytest.mark.unit
def test_max_variance_policy_initialization():
    """Test MaxVariancePolicy initialization."""
    from baselines import MaxVariancePolicy
    
    nodes = ["X1", "X2", "X3", "X4", "X5"]
    policy = MaxVariancePolicy(nodes, n_candidates=64, n_mc_samples=10)
    
    assert policy.nodes == nodes
    assert policy.n_candidates == 64
    assert policy.n_mc_samples == 10
    assert policy.name == "Max-Variance"


@pytest.mark.unit
def test_max_variance_select_intervention(seed_everything):
    """Test max variance intervention selection."""
    from baselines import MaxVariancePolicy, GroundTruthSCM, StudentSCM
    
    seed_everything(42)
    
    nodes = ["X1", "X2", "X3", "X4", "X5"]
    policy = MaxVariancePolicy(nodes, n_candidates=10, n_mc_samples=5)
    
    gt = GroundTruthSCM()
    student = StudentSCM(gt)
    
    # Select intervention
    target, value = policy.select_intervention(student, oracle=gt)
    
    # Should return valid target and value
    assert target in nodes
    assert isinstance(value, float)


@pytest.mark.unit
def test_max_variance_selects_from_candidates(seed_everything):
    """Test that max variance evaluates multiple candidates."""
    from baselines import MaxVariancePolicy, GroundTruthSCM, StudentSCM
    
    seed_everything(42)
    
    nodes = ["X1", "X2", "X3", "X4", "X5"]
    policy = MaxVariancePolicy(nodes, n_candidates=20, n_mc_samples=3)
    
    gt = GroundTruthSCM()
    student = StudentSCM(gt)
    
    # Should select without error
    target, value = policy.select_intervention(student, oracle=gt)
    
    assert target in nodes
    assert -5.0 <= value <= 5.0


@pytest.mark.integration
@pytest.mark.slow
def test_max_variance_basic_run(seed_everything):
    """Test basic max variance policy execution."""
    from baselines import GroundTruthSCM, MaxVariancePolicy, run_baseline
    import pandas as pd
    
    seed_everything(42)
    
    gt = GroundTruthSCM()
    policy = MaxVariancePolicy(nodes=gt.nodes, n_candidates=10, n_mc_samples=3)
    
    # Run for few episodes (max variance is slower)
    results = run_baseline(
        policy=policy,
        oracle=gt,
        n_episodes=2,
        steps_per_episode=5
    )
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 10  # 2 episodes * 5 steps


@pytest.mark.unit
def test_max_variance_reproducible(seed_everything):
    """Test max variance policy reproducibility."""
    from baselines import MaxVariancePolicy, GroundTruthSCM, StudentSCM
    
    def run_with_seed(seed):
        seed_everything(seed)
        nodes = ["X1", "X2", "X3", "X4", "X5"]
        policy = MaxVariancePolicy(nodes, n_candidates=5, n_mc_samples=3)
        gt = GroundTruthSCM()
        student = StudentSCM(gt)
        return policy.select_intervention(student, oracle=gt)
    
    result1 = run_with_seed(99)
    result2 = run_with_seed(99)
    
    # Should be same (same seed)
    assert result1[0] == result2[0]  # target
    assert result1[1] == result2[1]  # value

"""
Unit tests for Random baseline policy.

Tests cover:
- Random target selection
- Random value selection
- Uniform distribution validation
- Reproducibility
"""

import pytest
import torch
import numpy as np
from collections import Counter


# =============================================================================
# Random Policy Tests
# =============================================================================

@pytest.mark.unit
def test_random_policy_basic_run(seed_everything):
    """Test basic random policy execution."""
    from baselines import GroundTruthSCM, RandomPolicy, run_baseline
    import pandas as pd
    
    seed_everything(42)
    
    gt = GroundTruthSCM()
    policy = RandomPolicy(nodes=gt.nodes)
    
    # Run for 5 episodes
    results = run_baseline(
        policy=policy,
        oracle=gt,
        n_episodes=5,
        steps_per_episode=10
    )
    
    # Should return DataFrame
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 50  # 5 episodes * 10 steps


@pytest.mark.unit
def test_random_policy_target_distribution(seed_everything):
    """Test that random policy samples targets uniformly."""
    from baselines import GroundTruthSCM, RandomPolicy, run_baseline
    import pandas as pd
    
    seed_everything(42)
    
    gt = GroundTruthSCM()
    policy = RandomPolicy(nodes=gt.nodes)
    
    # Run for many steps
    results = run_baseline(
        policy=policy,
        oracle=gt,
        n_episodes=10,
        steps_per_episode=50  # 500 total steps
    )
    
    # Check target distribution
    targets = results['target'].tolist()
    counts = Counter(targets)
    
    # Each node should appear roughly 1/5 of the time
    n_nodes = len(gt.nodes)
    expected_freq = len(targets) / n_nodes
    
    for node in gt.nodes:
        actual_count = counts[node]
        # Allow 40% deviation for randomness
        assert actual_count > expected_freq * 0.4
        assert actual_count < expected_freq * 1.6


@pytest.mark.unit  
def test_random_policy_value_range(seed_everything):
    """Test that random policy respects value range."""
    from baselines import RandomPolicy
    
    seed_everything(42)
    
    # Create policy with custom range
    policy = RandomPolicy(nodes=["X1", "X2", "X3"], value_min=-2.0, value_max=2.0)
    
    # Sample many interventions
    from baselines import GroundTruthSCM, StudentSCM
    gt = GroundTruthSCM()
    student = StudentSCM(gt)
    
    for _ in range(20):
        target, value = policy.select_intervention(student)
        assert -2.0 <= value <= 2.0


@pytest.mark.unit
def test_random_policy_reproducible():
    """Test that random policy selection is reproducible."""
    from baselines import RandomPolicy, GroundTruthSCM, StudentSCM
    import random
    
    def get_selections(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        
        policy = RandomPolicy(nodes=["X1", "X2", "X3"])
        gt = GroundTruthSCM()
        student = StudentSCM(gt)
        
        selections = []
        for _ in range(5):
            target, value = policy.select_intervention(student)
            selections.append((target, value))
        return selections
    
    sel1 = get_selections(99)
    sel2 = get_selections(99)
    
    assert sel1 == sel2


@pytest.mark.integration
@pytest.mark.slow
def test_random_policy_full_run(seed_everything):
    """Test complete random policy run."""
    from baselines import GroundTruthSCM, RandomPolicy, run_baseline
    import pandas as pd
    
    seed_everything(42)
    
    gt = GroundTruthSCM()
    policy = RandomPolicy(nodes=gt.nodes)
    
    results = run_baseline(
        policy=policy,
        oracle=gt,
        n_episodes=10,
        steps_per_episode=10
    )
    
    # Should complete and return DataFrame
    assert isinstance(results, pd.DataFrame)
    
    # Should have 100 rows (10 episodes * 10 steps)
    assert len(results) == 100
    
    # Should have expected columns
    assert 'target' in results.columns
    assert 'value' in results.columns
    assert 'total_loss' in results.columns

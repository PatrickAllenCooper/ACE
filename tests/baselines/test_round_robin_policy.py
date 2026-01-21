"""
Unit tests for Round-Robin baseline policy.

Tests cover:
- Cyclic target selection
- Topological ordering
- Reset functionality
- Deterministic target sequence
"""

import pytest
import torch
from collections import Counter


# =============================================================================
# Round-Robin Policy Tests
# =============================================================================

@pytest.mark.unit
def test_round_robin_policy_initialization():
    """Test RoundRobinPolicy initialization."""
    from baselines import RoundRobinPolicy
    
    nodes = ["X1", "X2", "X3", "X4", "X5"]
    policy = RoundRobinPolicy(nodes)
    
    assert hasattr(policy, 'nodes')
    assert policy.step == 0
    assert policy.name == "Round-Robin"


@pytest.mark.unit
def test_round_robin_cycles_through_nodes(seed_everything):
    """Test that round-robin cycles through all nodes."""
    from baselines import RoundRobinPolicy, GroundTruthSCM, StudentSCM
    
    seed_everything(42)
    
    nodes = ["X1", "X2", "X3", "X4", "X5"]
    policy = RoundRobinPolicy(nodes)
    
    gt = GroundTruthSCM()
    student = StudentSCM(gt)
    
    # Select multiple interventions
    targets = []
    for _ in range(15):  # 3 full cycles
        target, value = policy.select_intervention(student)
        targets.append(target)
    
    # Should cycle through nodes in order
    # Expected order: X1, X4, X2, X5, X3 (topological)
    assert len(targets) == 15
    
    # Each node should appear exactly 3 times
    counts = Counter(targets)
    for node in policy.nodes:
        assert counts[node] == 3


@pytest.mark.unit
def test_round_robin_maintains_topological_order():
    """Test that round-robin uses topological order."""
    from baselines import RoundRobinPolicy, StudentSCM, GroundTruthSCM
    
    nodes = ["X1", "X2", "X3", "X4", "X5"]
    policy = RoundRobinPolicy(nodes)
    
    gt = GroundTruthSCM()
    student = StudentSCM(gt)
    
    # First 5 selections should follow topological order
    targets = []
    for _ in range(5):
        target, _ = policy.select_intervention(student)
        targets.append(target)
    
    # Should match policy.nodes order (topological)
    assert targets == policy.nodes


@pytest.mark.unit
def test_round_robin_reset():
    """Test round-robin reset functionality."""
    from baselines import RoundRobinPolicy, StudentSCM, GroundTruthSCM
    
    nodes = ["X1", "X2", "X3", "X4", "X5"]
    policy = RoundRobinPolicy(nodes)
    
    gt = GroundTruthSCM()
    student = StudentSCM(gt)
    
    # Advance step counter
    for _ in range(7):
        policy.select_intervention(student)
    
    assert policy.step == 7
    
    # Reset
    policy.reset()
    
    assert policy.step == 0
    
    # Next selection should be first node
    target, _ = policy.select_intervention(student)
    assert target == policy.nodes[0]


@pytest.mark.unit
def test_round_robin_values_are_random(seed_everything):
    """Test that round-robin values are still random."""
    from baselines import RoundRobinPolicy, StudentSCM, GroundTruthSCM
    
    seed_everything(42)
    
    nodes = ["X1", "X2", "X3", "X4", "X5"]
    policy = RoundRobinPolicy(nodes)
    
    gt = GroundTruthSCM()
    student = StudentSCM(gt)
    
    # Get values for same target
    values = []
    for _ in range(10):  # 2 full cycles
        target, value = policy.select_intervention(student)
        if target == "X1":
            values.append(value)
    
    # Values should vary (random)
    assert len(values) == 2
    # They should be different (very unlikely to be same)
    if len(values) > 1:
        assert values[0] != values[1]


@pytest.mark.integration
@pytest.mark.slow
def test_round_robin_basic_run(seed_everything):
    """Test basic round-robin policy execution."""
    from baselines import GroundTruthSCM, RoundRobinPolicy, run_baseline
    import pandas as pd
    
    seed_everything(42)
    
    gt = GroundTruthSCM()
    policy = RoundRobinPolicy(nodes=gt.nodes)
    
    results = run_baseline(
        policy=policy,
        oracle=gt,
        n_episodes=3,
        steps_per_episode=10
    )
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 30


@pytest.mark.integration
@pytest.mark.slow
def test_round_robin_uniform_coverage(seed_everything):
    """Test that round-robin provides uniform coverage."""
    from baselines import GroundTruthSCM, RoundRobinPolicy, run_baseline
    
    seed_everything(42)
    
    gt = GroundTruthSCM()
    policy = RoundRobinPolicy(nodes=gt.nodes)
    
    # Run for exact multiple of cycle
    results = run_baseline(
        policy=policy,
        oracle=gt,
        n_episodes=5,
        steps_per_episode=5  # 25 steps = 5 full cycles
    )
    
    targets = results['target'].tolist()
    counts = Counter(targets)
    
    # Each node should appear exactly 5 times
    for node in ["X1", "X4", "X2", "X5", "X3"]:
        assert counts[node] == 5

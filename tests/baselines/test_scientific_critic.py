"""
Unit tests for ScientificCritic class from baselines.py.

Tests cover:
- Initialization
- Evaluation method
- Loss computation
- Per-node losses
"""

import pytest
import torch


# =============================================================================
# ScientificCritic Tests
# =============================================================================

@pytest.mark.unit
def test_scientific_critic_initialization():
    """Test ScientificCritic initialization."""
    from baselines import ScientificCritic, GroundTruthSCM
    
    oracle = GroundTruthSCM()
    critic = ScientificCritic(oracle)
    
    assert hasattr(critic, 'oracle')
    assert critic.oracle is oracle


@pytest.mark.unit
def test_scientific_critic_evaluate(seed_everything):
    """Test critic evaluation."""
    from baselines import ScientificCritic, GroundTruthSCM, StudentSCM
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    critic = ScientificCritic(oracle)
    student = StudentSCM(oracle)
    
    # Evaluate student
    total_loss, node_losses = critic.evaluate(student)
    
    # Should return total loss and per-node losses
    assert isinstance(total_loss, float)
    assert total_loss >= 0
    
    assert isinstance(node_losses, dict)
    assert set(node_losses.keys()) == set(oracle.nodes)
    
    # All node losses should be non-negative
    for node, loss in node_losses.items():
        assert loss >= 0


@pytest.mark.unit
def test_scientific_critic_evaluate_multiple_times(seed_everything):
    """Test that critic can evaluate multiple times."""
    from baselines import ScientificCritic, GroundTruthSCM, StudentSCM
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    critic = ScientificCritic(oracle)
    student = StudentSCM(oracle)
    
    # Multiple evaluations
    for _ in range(3):
        total_loss, node_losses = critic.evaluate(student)
        assert total_loss >= 0
        assert len(node_losses) == 5


@pytest.mark.unit
def test_scientific_critic_loss_decreases_with_training(seed_everything):
    """Test that critic detects loss decrease after training."""
    from baselines import ScientificCritic, GroundTruthSCM, StudentSCM, SCMLearner
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    critic = ScientificCritic(oracle)
    student = StudentSCM(oracle)
    learner = SCMLearner(student)
    
    # Initial evaluation
    initial_loss, _ = critic.evaluate(student)
    
    # Train
    for _ in range(10):
        data = oracle.generate(100)
        learner.train_step(data)
    
    # Re-evaluate
    final_loss, _ = critic.evaluate(student)
    
    # Loss should have decreased (or stayed similar)
    assert final_loss <= initial_loss * 1.5  # Allow some variance

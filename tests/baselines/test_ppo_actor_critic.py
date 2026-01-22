"""
Tests for PPOActorCritic architecture from baselines.py.

Tests cover:
- Initialization
- Forward pass
- State encoding
- Output structure
"""

import pytest
import torch
import torch.nn as nn


# =============================================================================
# PPOActorCritic Tests
# =============================================================================

@pytest.mark.unit
def test_ppo_actor_critic_initialization():
    """Test PPOActorCritic initialization."""
    from baselines import PPOActorCritic
    
    nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    model = PPOActorCritic(nodes)
    
    # Should be nn.Module
    assert isinstance(model, nn.Module)
    
    # Check attributes
    assert hasattr(model, 'nodes')
    assert hasattr(model, 'shared')
    assert hasattr(model, 'actor_node')
    assert hasattr(model, 'actor_value')
    assert hasattr(model, 'critic')


@pytest.mark.unit
def test_ppo_actor_critic_forward_pass():
    """Test PPOActorCritic forward pass."""
    from baselines import PPOActorCritic
    
    nodes = ['X1', 'X2', 'X3']
    model = PPOActorCritic(nodes)
    model.eval()
    
    # Create dummy state (state_dim = n_nodes * 3)
    state_dim = len(nodes) * 3
    state = torch.randn(4, state_dim)
    
    # Forward pass
    node_logits, value_logits, state_value = model(state)
    
    # Check shapes
    assert node_logits.shape == (4, len(nodes))
    assert value_logits.shape[0] == 4
    assert state_value.shape == (4, 1)


@pytest.mark.unit
def test_ppo_actor_critic_encode_state():
    """Test PPOActorCritic state encoding."""
    from baselines import PPOActorCritic
    from collections import Counter
    
    nodes = ['X1', 'X2', 'X3']
    model = PPOActorCritic(nodes)
    
    node_losses = {'X1': 1.0, 'X2': 2.0, 'X3': 0.5}
    intervention_counts = Counter({'X1': 10, 'X2': 5})
    recent_targets = ['X1', 'X2', 'X1', 'X3']
    
    # Encode state
    state = model._encode_state(node_losses, intervention_counts, recent_targets)
    
    # Should return tensor
    assert isinstance(state, torch.Tensor)
    assert state.shape == (1, len(nodes) * 3)


@pytest.mark.unit
def test_ppo_actor_critic_get_action():
    """Test PPOActorCritic action selection."""
    from baselines import PPOActorCritic
    from collections import Counter
    
    nodes = ['X1', 'X2', 'X3']
    model = PPOActorCritic(nodes)
    model.eval()
    
    node_losses = {'X1': 1.0, 'X2': 2.0, 'X3': 0.5}
    intervention_counts = Counter()
    recent_targets = []
    
    # Get action (returns tuple - check actual return signature)
    result = model.get_action(node_losses, intervention_counts, recent_targets)
    
    # Should return a tuple
    assert isinstance(result, tuple)
    assert len(result) >= 2  # At least node_idx and value_idx


@pytest.mark.unit
def test_ppo_actor_critic_trainable():
    """Test that PPOActorCritic has trainable parameters."""
    from baselines import PPOActorCritic
    
    nodes = ['X1', 'X2']
    model = PPOActorCritic(nodes)
    
    # Should have parameters
    params = list(model.parameters())
    assert len(params) > 0
    
    # Can set train/eval
    model.train()
    assert model.training
    
    model.eval()
    assert not model.training


@pytest.mark.unit
def test_ppo_actor_critic_gradient_flow():
    """Test gradients flow through PPOActorCritic."""
    from baselines import PPOActorCritic
    
    nodes = ['X1', 'X2']
    model = PPOActorCritic(nodes)
    model.train()
    
    state = torch.randn(2, len(nodes) * 3)
    
    # Forward
    node_logits, value_logits, state_value = model(state)
    
    # Compute loss and backward
    loss = node_logits.mean() + state_value.mean()
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad

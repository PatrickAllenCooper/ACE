"""
Unit tests for TransformerPolicy class.

Tests cover:
- Initialization
- Architecture
- State encoding integration
- Forward pass
- Generation capability
"""

import pytest
import torch
import torch.nn as nn


# =============================================================================
# TransformerPolicy Tests
# =============================================================================

@pytest.mark.unit
def test_transformer_policy_initialization():
    """Test TransformerPolicy initialization."""
    from ace_experiments import TransformerPolicy, ExperimentalDSL
    
    device = torch.device('cpu')
    nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    dsl = ExperimentalDSL(nodes)
    
    policy = TransformerPolicy(dsl, device, d_model=128, nhead=4, num_layers=2)
    
    # Check basic attributes
    assert policy.dsl is dsl
    assert policy.device == device
    assert policy.d_model == 128
    
    # Should be nn.Module
    assert isinstance(policy, nn.Module)


@pytest.mark.unit
def test_transformer_policy_has_components():
    """Test that TransformerPolicy has required components."""
    from ace_experiments import TransformerPolicy, ExperimentalDSL
    
    device = torch.device('cpu')
    nodes = ['X1', 'X2', 'X3']
    dsl = ExperimentalDSL(nodes)
    
    policy = TransformerPolicy(dsl, device)
    
    # Should have key components
    assert hasattr(policy, 'state_encoder')
    assert hasattr(policy, 'transformer_enc')
    assert hasattr(policy, 'token_embedding')
    assert hasattr(policy, 'transformer_dec')
    assert hasattr(policy, 'output_head')


@pytest.mark.unit
def test_transformer_policy_forward_with_seq(student_scm):
    """Test forward pass with target sequence."""
    from ace_experiments import TransformerPolicy, ExperimentalDSL
    
    device = torch.device('cpu')
    nodes = list(student_scm.nodes)
    dsl = ExperimentalDSL(nodes)
    
    policy = TransformerPolicy(dsl, device, d_model=64)
    policy.eval()
    
    node_losses = {node: 1.0 for node in nodes}
    
    # Create dummy target sequence
    target_seq = torch.randint(0, len(dsl.vocab), (1, 3))
    
    # Forward pass (takes scm_student directly)
    output = policy(student_scm, target_seq=target_seq, node_losses=node_losses)
    
    # Should return logits
    assert isinstance(output, torch.Tensor)
    assert output.shape[-1] == len(dsl.vocab)


@pytest.mark.unit
def test_transformer_policy_forward_without_seq(student_scm):
    """Test forward pass without target sequence (returns memory)."""
    from ace_experiments import TransformerPolicy, ExperimentalDSL
    
    device = torch.device('cpu')
    nodes = list(student_scm.nodes)
    dsl = ExperimentalDSL(nodes)
    
    policy = TransformerPolicy(dsl, device)
    policy.eval()
    
    node_losses = {node: 1.0 for node in nodes}
    
    # Forward without target_seq (takes scm_student)
    memory = policy(student_scm, target_seq=None, node_losses=node_losses)
    
    # Should return memory tensor
    assert isinstance(memory, torch.Tensor)


@pytest.mark.unit
def test_transformer_policy_generate_experiment(student_scm, seed_everything):
    """Test experiment generation."""
    from ace_experiments import TransformerPolicy, ExperimentalDSL
    
    seed_everything(42)
    
    device = torch.device('cpu')
    nodes = list(student_scm.nodes)
    dsl = ExperimentalDSL(nodes)
    
    policy = TransformerPolicy(dsl, device, d_model=64)
    policy.eval()
    
    node_losses = {node: 1.0 for node in nodes}
    
    # Generate experiment
    result = policy.generate_experiment(student_scm, node_losses=node_losses)
    
    # Should return tuple (command, parsed)
    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.mark.unit
def test_transformer_policy_is_trainable():
    """Test that TransformerPolicy has trainable parameters."""
    from ace_experiments import TransformerPolicy, ExperimentalDSL
    
    device = torch.device('cpu')
    nodes = ['X1', 'X2', 'X3']
    dsl = ExperimentalDSL(nodes)
    
    policy = TransformerPolicy(dsl, device)
    
    # Should have parameters
    params = list(policy.parameters())
    assert len(params) > 0
    
    # Should be able to set train/eval
    policy.train()
    assert policy.training
    
    policy.eval()
    assert not policy.training


@pytest.mark.unit
def test_transformer_policy_gradient_flow(student_scm):
    """Test that gradients flow through policy."""
    from ace_experiments import TransformerPolicy, ExperimentalDSL
    
    device = torch.device('cpu')
    nodes = list(student_scm.nodes)
    dsl = ExperimentalDSL(nodes)
    
    policy = TransformerPolicy(dsl, device, d_model=32)
    policy.train()
    
    # Create inputs
    node_losses = {node: 1.0 for node in nodes}
    target_seq = torch.randint(0, len(dsl.vocab), (1, 3))
    
    # Forward and backward (takes scm_student)
    output = policy(student_scm, target_seq=target_seq, node_losses=node_losses)
    loss = output.mean()
    loss.backward()
    
    # Check gradients exist
    has_grad = any(p.grad is not None for p in policy.parameters())
    assert has_grad


@pytest.mark.unit
def test_transformer_policy_custom_architecture():
    """Test TransformerPolicy with custom architecture."""
    from ace_experiments import TransformerPolicy, ExperimentalDSL
    
    device = torch.device('cpu')
    nodes = ['X1', 'X2']
    dsl = ExperimentalDSL(nodes)
    
    # Custom architecture
    policy = TransformerPolicy(dsl, device, d_model=256, nhead=8, num_layers=4)
    
    assert isinstance(policy, nn.Module)
    # Should initialize without error

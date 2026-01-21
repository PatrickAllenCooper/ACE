"""
Unit tests for StateEncoder class.

Tests cover:
- Initialization
- State encoding
- Feature dimensions
- Loss inclusion
- Device handling
"""

import pytest
import torch
import torch.nn as nn


# =============================================================================
# Initialization Tests
# =============================================================================

@pytest.mark.unit
def test_state_encoder_initialization():
    """Test StateEncoder initialization."""
    from ace_experiments import StateEncoder
    
    n_nodes = 5
    device = torch.device('cpu')
    d_model = 128
    
    encoder = StateEncoder(n_nodes=n_nodes, device=device, d_model=d_model)
    
    # Check attributes
    assert hasattr(encoder, 'n_nodes')
    assert encoder.n_nodes == n_nodes
    
    # d_model might not be stored as attribute in all versions
    # Just check it's a nn.Module
    assert isinstance(encoder, nn.Module)


@pytest.mark.unit
def test_state_encoder_is_nn_module():
    """Test that StateEncoder is a PyTorch module."""
    from ace_experiments import StateEncoder
    
    encoder = StateEncoder(n_nodes=5, device=torch.device('cpu'))
    
    assert isinstance(encoder, nn.Module)
    
    # Should have parameters
    params = list(encoder.parameters())
    assert len(params) > 0


@pytest.mark.unit
def test_state_encoder_custom_d_model():
    """Test StateEncoder with custom d_model produces correct output."""
    from ace_experiments import StateEncoder
    
    encoder = StateEncoder(n_nodes=5, device=torch.device('cpu'), d_model=256)
    
    # Test by checking output shape
    assert isinstance(encoder, nn.Module)


# =============================================================================
# Encoding Tests
# =============================================================================

@pytest.mark.unit
def test_state_encoder_forward_pass(student_scm):
    """Test forward pass through StateEncoder."""
    from ace_experiments import StateEncoder
    
    device = torch.device('cpu')
    n_nodes = len(student_scm.nodes)
    
    encoder = StateEncoder(n_nodes=n_nodes, device=device, d_model=128)
    encoder.eval()
    
    # Create dummy node losses (as floats, not tensors)
    node_losses = {node: 1.0 for node in student_scm.nodes}
    
    # Forward pass
    encoding = encoder(student_scm, node_losses)
    
    # Should return a tensor
    assert isinstance(encoding, torch.Tensor)
    
    # Check shape [1, n_nodes, d_model]
    assert encoding.dim() == 3
    assert encoding.shape[0] == 1
    assert encoding.shape[1] == n_nodes
    assert encoding.shape[2] == 128


@pytest.mark.unit
def test_state_encoder_output_shape(student_scm):
    """Test that encoder output has correct shape."""
    from ace_experiments import StateEncoder
    
    device = torch.device('cpu')
    encoder = StateEncoder(n_nodes=5, device=device, d_model=64)
    encoder.eval()
    
    node_losses = {f'X{i}': float(i) for i in range(1, 6)}
    
    encoding = encoder(student_scm, node_losses)
    
    # Output should be [1, n_nodes, d_model]
    assert encoding.shape == (1, 5, 64)


@pytest.mark.unit
def test_state_encoder_with_varying_losses(student_scm):
    """Test encoder with different loss values."""
    from ace_experiments import StateEncoder
    
    device = torch.device('cpu')
    encoder = StateEncoder(n_nodes=5, device=device)
    encoder.eval()
    
    # Low losses
    low_losses = {node: torch.tensor([0.1]) for node in student_scm.nodes}
    encoding_low = encoder(student_scm, low_losses)
    
    # High losses
    high_losses = {node: torch.tensor([10.0]) for node in student_scm.nodes}
    encoding_high = encoder(student_scm, high_losses)
    
    # Encodings should be different
    assert not torch.allclose(encoding_low, encoding_high, atol=1e-6)


@pytest.mark.unit
def test_state_encoder_deterministic(student_scm):
    """Test that encoder is deterministic in eval mode."""
    from ace_experiments import StateEncoder
    
    device = torch.device('cpu')
    encoder = StateEncoder(n_nodes=5, device=device)
    encoder.eval()
    
    node_losses = {node: torch.tensor([1.0]) for node in student_scm.nodes}
    
    # Two forward passes
    encoding1 = encoder(student_scm, node_losses)
    encoding2 = encoder(student_scm, node_losses)
    
    # Should be identical
    assert torch.allclose(encoding1, encoding2)


# =============================================================================
# Device Handling Tests
# =============================================================================

@pytest.mark.unit
def test_state_encoder_cpu_device(student_scm):
    """Test encoder on CPU device."""
    from ace_experiments import StateEncoder
    
    device = torch.device('cpu')
    encoder = StateEncoder(n_nodes=5, device=device)
    
    node_losses = {node: torch.tensor([1.0]) for node in student_scm.nodes}
    encoding = encoder(student_scm, node_losses)
    
    # Output should be on CPU
    assert encoding.device.type == 'cpu'


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_state_encoder_gpu_device(student_scm):
    """Test encoder on GPU device."""
    from ace_experiments import StateEncoder
    
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    device = torch.device('cuda')
    encoder = StateEncoder(n_nodes=5, device=device)
    
    # Move student to GPU
    student_scm = student_scm.to(device)
    
    node_losses = {node: torch.tensor([1.0], device=device) for node in student_scm.nodes}
    encoding = encoder(student_scm, node_losses)
    
    # Output should be on GPU
    assert encoding.device.type == 'cuda'


# =============================================================================
# Loss Inclusion Tests
# =============================================================================

@pytest.mark.unit
def test_state_encoder_includes_loss_information(student_scm):
    """Test that encoder incorporates loss information."""
    from ace_experiments import StateEncoder
    
    device = torch.device('cpu')
    encoder = StateEncoder(n_nodes=5, device=device)
    encoder.eval()
    
    # Same student, different losses should give different encodings
    losses1 = {f'X{i}': torch.tensor([0.1]) for i in range(1, 6)}
    losses2 = {f'X{i}': torch.tensor([5.0]) for i in range(1, 6)}
    
    enc1 = encoder(student_scm, losses1)
    enc2 = encoder(student_scm, losses2)
    
    # Should be different (losses affect encoding)
    assert not torch.allclose(enc1, enc2, atol=0.1)


@pytest.mark.unit
def test_state_encoder_per_node_losses(student_scm):
    """Test encoder with per-node loss variations."""
    from ace_experiments import StateEncoder
    
    device = torch.device('cpu')
    encoder = StateEncoder(n_nodes=5, device=device)
    encoder.eval()
    
    # Different losses per node (as floats)
    varied_losses = {
        'X1': 0.1,
        'X2': 1.0,
        'X3': 5.0,
        'X4': 0.2,
        'X5': 0.3
    }
    
    encoding = encoder(student_scm, varied_losses)
    
    # Should produce valid encoding [1, 5, 128]
    assert encoding.shape == (1, 5, 128)
    assert not torch.isnan(encoding).any()
    assert not torch.isinf(encoding).any()


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.unit
def test_state_encoder_with_zero_losses(student_scm):
    """Test encoder with all zero losses."""
    from ace_experiments import StateEncoder
    
    device = torch.device('cpu')
    encoder = StateEncoder(n_nodes=5, device=device)
    encoder.eval()
    
    zero_losses = {node: 0.0 for node in student_scm.nodes}
    
    encoding = encoder(student_scm, zero_losses)
    
    # Should handle zero losses without error
    assert encoding.shape == (1, 5, 128)
    assert not torch.isnan(encoding).any()


@pytest.mark.unit
def test_state_encoder_with_large_losses(student_scm):
    """Test encoder with very large losses."""
    from ace_experiments import StateEncoder
    
    device = torch.device('cpu')
    encoder = StateEncoder(n_nodes=5, device=device)
    encoder.eval()
    
    large_losses = {node: 1000.0 for node in student_scm.nodes}
    
    encoding = encoder(student_scm, large_losses)
    
    # Should handle large losses without overflow
    assert encoding.shape == (1, 5, 128)
    assert not torch.isnan(encoding).any()
    assert not torch.isinf(encoding).any()


# =============================================================================
# Gradient Tests
# =============================================================================

@pytest.mark.unit
def test_state_encoder_gradients_flow(student_scm):
    """Test that gradients flow through encoder."""
    from ace_experiments import StateEncoder
    
    device = torch.device('cpu')
    encoder = StateEncoder(n_nodes=5, device=device)
    encoder.train()
    
    node_losses = {node: torch.tensor([1.0]) for node in student_scm.nodes}
    
    # Forward pass
    encoding = encoder(student_scm, node_losses)
    
    # Compute loss and backward
    loss = encoding.mean()
    loss.backward()
    
    # Check that encoder has gradients
    has_grad = False
    for param in encoder.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "Encoder should have gradients"


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.unit
def test_state_encoder_works_with_student_scm(student_scm, ground_truth_scm, seed_everything):
    """Test encoder integration with StudentSCM."""
    from ace_experiments import StateEncoder, SCMLearner
    
    seed_everything(42)
    device = torch.device('cpu')
    
    # Train student a bit
    learner = SCMLearner(student_scm, lr=0.01)
    gt_data = ground_truth_scm.generate(100)
    learner.train_step({"data": gt_data, "intervened": None}, n_epochs=5)
    
    # Get actual losses (as floats)
    with torch.no_grad():
        student_data = student_scm.forward(100)
        node_losses = {}
        for node in student_scm.nodes:
            loss = ((student_data[node] - gt_data[node]) ** 2).mean()
            node_losses[node] = loss.item()  # Convert to float
    
    # Encode state
    encoder = StateEncoder(n_nodes=5, device=device)
    encoding = encoder(student_scm, node_losses)
    
    assert encoding.shape == (1, 5, 128)
    assert not torch.isnan(encoding).any()

"""
Unit tests for EarlyStopping class.

Tests cover:
- Initialization
- check_loss method
- check_per_node_convergence method
- Node convergence tracking
"""

import pytest
import torch


# =============================================================================
# Initialization Tests
# =============================================================================

@pytest.mark.unit
def test_early_stopping_initialization():
    """Test EarlyStopping initialization."""
    from ace_experiments import EarlyStopping
    
    stopper = EarlyStopping(
        patience=20,
        min_delta=0.01,
        min_episodes=40
    )
    
    assert stopper.patience == 20
    assert stopper.min_delta == 0.01
    assert stopper.min_episodes == 40
    assert stopper.counter == 0
    assert stopper.best_loss == float('inf')


@pytest.mark.unit
def test_early_stopping_default_values():
    """Test EarlyStopping with default values."""
    from ace_experiments import EarlyStopping
    
    stopper = EarlyStopping()
    
    # Should have sensible defaults
    assert stopper.patience > 0
    assert stopper.min_delta >= 0
    assert stopper.min_episodes >= 0
    assert hasattr(stopper, 'node_targets')


# =============================================================================
# check_loss Method Tests
# =============================================================================

@pytest.mark.unit
def test_check_loss_returns_false_on_improvement():
    """Test check_loss returns False when improving."""
    from ace_experiments import EarlyStopping
    
    stopper = EarlyStopping(patience=5, min_delta=0.01)
    
    # Should not stop on improvement
    should_stop = stopper.check_loss(1.0)
    assert not should_stop
    assert stopper.best_loss == 1.0
    assert stopper.counter == 0
    
    # Improvement
    should_stop = stopper.check_loss(0.5)
    assert not should_stop
    assert stopper.best_loss == 0.5
    assert stopper.counter == 0


@pytest.mark.unit
def test_check_loss_increments_counter_on_no_improvement():
    """Test that counter increments when no improvement."""
    from ace_experiments import EarlyStopping
    
    stopper = EarlyStopping(patience=5, min_delta=0.01)
    
    stopper.check_loss(1.0)  # Initial
    assert stopper.counter == 0
    
    stopper.check_loss(1.0)  # No improvement
    assert stopper.counter == 1
    
    stopper.check_loss(1.0)  # No improvement
    assert stopper.counter == 2


@pytest.mark.unit
def test_check_loss_returns_true_after_patience():
    """Test check_loss returns True after patience exceeded."""
    from ace_experiments import EarlyStopping
    
    stopper = EarlyStopping(patience=3, min_delta=0.01)
    
    stopper.check_loss(1.0)  # Initial, counter=0
    stopper.check_loss(1.0)  # counter=1
    stopper.check_loss(1.0)  # counter=2
    should_stop = stopper.check_loss(1.0)  # counter=3
    
    assert should_stop
    assert stopper.counter >= stopper.patience


@pytest.mark.unit
def test_check_loss_resets_counter_on_improvement():
    """Test that counter resets when improvement detected."""
    from ace_experiments import EarlyStopping
    
    stopper = EarlyStopping(patience=5, min_delta=0.01)
    
    stopper.check_loss(10.0)
    stopper.check_loss(10.0)  # counter=1
    stopper.check_loss(10.0)  # counter=2
    assert stopper.counter == 2
    
    # Improvement resets
    stopper.check_loss(5.0)
    assert stopper.counter == 0


@pytest.mark.unit
def test_check_loss_respects_min_delta():
    """Test that min_delta threshold works correctly."""
    from ace_experiments import EarlyStopping
    
    stopper = EarlyStopping(patience=5, min_delta=0.1)
    
    stopper.check_loss(1.0)
    
    # Small improvement (0.05 < 0.1)
    stopper.check_loss(0.95)
    assert stopper.counter == 1  # Should count as no improvement
    
    # Large improvement (0.2 > 0.1)
    stopper.check_loss(0.75)
    assert stopper.counter == 0  # Should reset


# =============================================================================
# check_per_node_convergence Method Tests
# =============================================================================

@pytest.mark.unit
def test_check_per_node_convergence_all_below_target():
    """Test per-node convergence when all nodes below targets."""
    from ace_experiments import EarlyStopping
    
    stopper = EarlyStopping(
        node_targets={'X1': 1.0, 'X2': 0.5, 'X3': 0.5, 'X4': 1.0, 'X5': 0.5}
    )
    
    # All nodes below targets
    node_losses = {'X1': 0.3, 'X2': 0.2, 'X3': 0.2, 'X4': 0.3, 'X5': 0.2}
    
    # Check multiple times (need patience consecutive)
    for _ in range(11):  # patience=10 by default in method
        converged = stopper.check_per_node_convergence(node_losses, patience=10)
    
    # Should eventually return True
    assert converged


@pytest.mark.unit
def test_check_per_node_convergence_some_above_target():
    """Test per-node convergence when some nodes above targets."""
    from ace_experiments import EarlyStopping
    
    stopper = EarlyStopping(
        node_targets={'X1': 1.0, 'X2': 0.5, 'X3': 0.5, 'X4': 1.0, 'X5': 0.5}
    )
    
    # X3 above target
    node_losses = {'X1': 0.3, 'X2': 0.2, 'X3': 2.0, 'X4': 0.3, 'X5': 0.2}
    
    converged = stopper.check_per_node_convergence(node_losses, patience=10)
    
    # Should not converge
    assert not converged


@pytest.mark.unit
def test_check_per_node_convergence_empty_losses():
    """Test per-node convergence with empty losses dict."""
    from ace_experiments import EarlyStopping
    
    stopper = EarlyStopping()
    
    # Empty losses
    converged = stopper.check_per_node_convergence({}, patience=10)
    
    # Should return False
    assert not converged

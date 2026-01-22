"""
Unit tests for DPOLogger class and get_dpo_logger function.

Tests cover:
- Initialization
- Global logger access
- History tracking
"""

import pytest
import torch


# =============================================================================
# DPOLogger Tests
# =============================================================================

@pytest.mark.unit
def test_dpo_logger_initialization():
    """Test DPOLogger initialization."""
    from ace_experiments import DPOLogger
    
    logger = DPOLogger()
    
    assert hasattr(logger, 'history')
    assert isinstance(logger.history, dict)
    assert hasattr(logger, 'step_count')
    assert logger.step_count == 0


@pytest.mark.unit
def test_dpo_logger_history_structure():
    """Test that logger has expected history structure."""
    from ace_experiments import DPOLogger
    
    logger = DPOLogger()
    
    # Check expected keys
    assert 'loss' in logger.history
    assert 'preference_margin' in logger.history
    assert 'ref_margin' in logger.history
    assert 'kl_from_ref' in logger.history
    assert 'winner_target' in logger.history
    assert 'loser_target' in logger.history
    assert 'sigmoid_input' in logger.history
    
    # All should be empty lists initially
    for key, value in logger.history.items():
        assert isinstance(value, list)
        assert len(value) == 0


# =============================================================================
# get_dpo_logger Tests
# =============================================================================

@pytest.mark.unit
def test_get_dpo_logger_returns_singleton():
    """Test that get_dpo_logger returns same instance."""
    from ace_experiments import get_dpo_logger
    
    logger1 = get_dpo_logger()
    logger2 = get_dpo_logger()
    
    # Should be same instance
    assert logger1 is logger2


@pytest.mark.unit
def test_get_dpo_logger_creates_on_first_call():
    """Test that get_dpo_logger creates logger on first call."""
    from ace_experiments import get_dpo_logger, DPOLogger
    
    logger = get_dpo_logger()
    
    # Should be DPOLogger instance
    assert isinstance(logger, DPOLogger)

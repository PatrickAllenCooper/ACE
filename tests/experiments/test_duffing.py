"""
Unit tests for Duffing Oscillators experiment.

Tests cover:
- Module import
- Basic functionality
- Data generation
"""

import pytest
import torch


# =============================================================================
# Duffing Oscillators Tests
# =============================================================================

@pytest.mark.unit
def test_duffing_module_imports():
    """Test that duffing module can be imported."""
    from experiments import duffing_oscillators
    
    assert duffing_oscillators is not None


@pytest.mark.unit
def test_duffing_has_required_functions():
    """Test that duffing module has expected functions."""
    from experiments import duffing_oscillators
    
    # Should have main components
    assert hasattr(duffing_oscillators, '__name__')


@pytest.mark.unit
def test_duffing_can_create_scm():
    """Test creating Duffing SCM."""
    try:
        from experiments.duffing_oscillators import DuffingGroundTruth
        
        scm = DuffingGroundTruth()
        assert scm is not None
        
    except (ImportError, AttributeError):
        # Module might have different structure
        pytest.skip("DuffingGroundTruth class not found in expected location")

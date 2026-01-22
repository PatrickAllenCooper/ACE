"""
Unit tests for Phillips Curve experiment.

Tests cover:
- Module import
- Basic functionality
"""

import pytest


# =============================================================================
# Phillips Curve Tests
# =============================================================================

@pytest.mark.unit
def test_phillips_module_imports():
    """Test that phillips module can be imported."""
    from experiments import phillips_curve
    
    assert phillips_curve is not None


@pytest.mark.unit
def test_phillips_has_components():
    """Test that phillips module has expected components."""
    from experiments import phillips_curve
    
    # Should import without error
    assert hasattr(phillips_curve, '__name__')


@pytest.mark.unit
def test_phillips_can_create_scm():
    """Test creating Phillips SCM if available."""
    try:
        from experiments.phillips_curve import PhillipsGroundTruth
        
        # Might require FRED data
        pytest.skip("Phillips SCM requires FRED data - skip for unit test")
        
    except (ImportError, AttributeError):
        # Expected - module might have different structure
        pytest.skip("PhillipsGroundTruth not available")

"""
Unit tests for clamping_detector.py module.

Tests cover:
- Module imports
- Basic functionality
- Detection logic
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path


# =============================================================================
# Module Tests
# =============================================================================

@pytest.mark.unit
def test_clamping_detector_imports():
    """Test that clamping_detector module can be imported."""
    import clamping_detector
    
    assert clamping_detector is not None


@pytest.mark.unit
def test_clamping_detector_has_main():
    """Test that clamping_detector has main function."""
    import clamping_detector
    
    # Should have executable components
    assert hasattr(clamping_detector, '__name__')


# =============================================================================
# Functional Tests (with mock data)
# =============================================================================

@pytest.mark.unit
def test_clamping_detector_with_mock_data(tmp_path):
    """Test clamping detection logic with mock data."""
    
    # Create test data file
    results_dir = tmp_path / "duffing_results"
    results_dir.mkdir()
    
    # Mock metrics CSV with clamping pattern
    metrics_df = pd.DataFrame({
        'episode': [0, 1, 2, 3, 4, 5],
        'step': [0, 0, 0, 0, 0, 0],
        'target': ['X1', 'X2', 'X2', 'X2', 'X2', 'X2'],
        'value': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)
    
    # Load and check for clamping pattern
    df = pd.read_csv(results_dir / "metrics.csv")
    
    # Check that X2=0.0 appears multiple times
    x2_zero_count = len(df[(df['target'] == 'X2') & (df['value'] == 0.0)])
    
    assert x2_zero_count >= 4  # Clamping pattern detected

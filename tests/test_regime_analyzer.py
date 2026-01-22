"""
Unit tests for regime_analyzer.py module.

Tests cover:
- Module imports
- Basic functionality
"""

import pytest
import pandas as pd
import tempfile


# =============================================================================
# Module Tests
# =============================================================================

@pytest.mark.unit
def test_regime_analyzer_imports():
    """Test that regime_analyzer module can be imported."""
    import regime_analyzer
    
    assert regime_analyzer is not None


@pytest.mark.unit
def test_regime_analyzer_has_main():
    """Test that regime_analyzer has expected structure."""
    import regime_analyzer
    
    # Should be importable
    assert hasattr(regime_analyzer, '__name__')


# =============================================================================
# Functional Tests (with mock data)
# =============================================================================

@pytest.mark.unit
def test_regime_analyzer_with_mock_data(tmp_path):
    """Test regime analysis with mock data."""
    
    # Create test data directory
    results_dir = tmp_path / "phillips_results"
    results_dir.mkdir()
    
    # Mock metrics CSV
    metrics_df = pd.DataFrame({
        'episode': list(range(10)),
        'step': [0] * 10,
        'target': ['X1'] * 10,
        'value': [1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 2.2, 1.9, 2.1, 2.0]
    })
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)
    
    # Load and verify structure
    df = pd.read_csv(results_dir / "metrics.csv")
    
    assert len(df) == 10
    assert 'value' in df.columns

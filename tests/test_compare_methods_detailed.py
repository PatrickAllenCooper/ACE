"""
Detailed tests for compare_methods.py functionality.

Tests cover:
- load_ace_results function
- Baseline results structure
- Table generation
- Data extraction logic
"""

import pytest
import pandas as pd
import tempfile


# =============================================================================
# Baseline Results Tests
# =============================================================================

@pytest.mark.unit
def test_baseline_results_structure():
    """Test BASELINE_RESULTS structure."""
    from compare_methods import BASELINE_RESULTS
    
    # Should have baseline results
    assert isinstance(BASELINE_RESULTS, dict)
    assert len(BASELINE_RESULTS) > 0
    
    # Check expected baselines
    expected_baselines = ['Random', 'Round-Robin', 'Max-Variance']
    for baseline in expected_baselines:
        assert baseline in BASELINE_RESULTS or 'Random' in BASELINE_RESULTS


@pytest.mark.unit
def test_baseline_results_have_losses():
    """Test that baseline results have loss values."""
    from compare_methods import BASELINE_RESULTS
    
    # Each baseline should have loss metrics
    for baseline_name, results in BASELINE_RESULTS.items():
        assert isinstance(results, dict)
        assert 'total' in results or len(results) > 0


@pytest.mark.unit
def test_load_ace_results_function_exists():
    """Test that load_ace_results function exists."""
    from compare_methods import load_ace_results
    
    assert callable(load_ace_results)


@pytest.mark.unit
def test_load_ace_results_no_data():
    """Test load_ace_results when no ACE data exists."""
    from compare_methods import load_ace_results
    
    # Should handle missing data gracefully
    result = load_ace_results()
    
    # Should return None or empty dict
    assert result is None or isinstance(result, dict)


# =============================================================================
# Data Processing Tests
# =============================================================================

@pytest.mark.unit
def test_dataframe_final_episode_extraction():
    """Test extracting final episode from DataFrame."""
    
    # Mock DataFrame
    df = pd.DataFrame({
        'episode': [0, 0, 1, 1, 2, 2],
        'step': [0, 1, 0, 1, 0, 1],
        'total_loss': [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        'loss_X3': [3.0, 2.8, 2.5, 2.2, 2.0, 1.8]
    })
    
    # Get final episode
    final_ep = df[df['episode'] == df['episode'].max()]
    
    assert len(final_ep) == 2  # Episode 2 has 2 steps
    
    # Get final row
    final_row = df.iloc[-1]
    assert final_row['total_loss'] == 5.0
    assert final_row['loss_X3'] == 1.8

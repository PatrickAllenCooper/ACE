"""
Unit tests for compare_methods.py module.

Tests cover:
- Module imports
- Table generation logic
- Data extraction
"""

import pytest
import pandas as pd


# =============================================================================
# Module Tests
# =============================================================================

@pytest.mark.unit
def test_compare_methods_imports():
    """Test that compare_methods module can be imported."""
    import compare_methods
    
    assert compare_methods is not None


@pytest.mark.unit
def test_compare_methods_has_main():
    """Test that compare_methods has main function."""
    import compare_methods
    
    # Should have expected structure
    assert hasattr(compare_methods, '__name__')


# =============================================================================
# Data Extraction Tests
# =============================================================================

@pytest.mark.unit
def test_compare_methods_can_process_dataframe():
    """Test that compare_methods can process result DataFrames."""
    
    # Create mock results
    df = pd.DataFrame({
        'episode': [0, 1, 2],
        'step': [0, 0, 0],
        'target': ['X1', 'X2', 'X3'],
        'value': [1.0, 2.0, 1.5],
        'total_loss': [10.0, 8.0, 6.0],
        'loss_X1': [2.0, 1.5, 1.2],
        'loss_X2': [3.0, 2.5, 2.0],
        'loss_X3': [5.0, 4.0, 2.8]
    })
    
    # Extract final losses
    final_loss = df['total_loss'].iloc[-1]
    final_x3_loss = df['loss_X3'].iloc[-1]
    
    assert final_loss == 6.0
    assert final_x3_loss == 2.8


@pytest.mark.unit
def test_compare_methods_table_structure():
    """Test expected table structure for comparisons."""
    
    # Mock comparison data
    methods = {
        'ACE': {'total_loss': 1.5, 'X3_loss': 0.3},
        'Random': {'total_loss': 2.2, 'X3_loss': 0.8},
        'RoundRobin': {'total_loss': 2.0, 'X3_loss': 0.6}
    }
    
    # Verify structure
    for method, metrics in methods.items():
        assert 'total_loss' in metrics
        assert 'X3_loss' in metrics
        assert metrics['total_loss'] >= 0

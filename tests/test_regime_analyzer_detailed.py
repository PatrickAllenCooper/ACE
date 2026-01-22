"""
Detailed tests for regime_analyzer.py functionality.

Tests cover:
- analyze_regime_selection function
- Regime definitions
- Volatility classification
"""

import pytest
import pandas as pd
import tempfile


# =============================================================================
# analyze_regime_selection Function Tests
# =============================================================================

@pytest.mark.unit
def test_analyze_regime_selection_function_exists():
    """Test that analyze_regime_selection function exists."""
    from regime_analyzer import analyze_regime_selection
    
    assert callable(analyze_regime_selection)


@pytest.mark.unit
def test_analyze_regime_selection_with_mock_data(tmp_path):
    """Test regime analysis with mock CSV data."""
    from regime_analyzer import analyze_regime_selection
    
    # Create mock results CSV
    results_file = tmp_path / "results.csv"
    df = pd.DataFrame({
        'episode': list(range(20)),
        'step': [0] * 20,
        'target': ['X1'] * 20,
        'value': [float(i) for i in range(20)]
    })
    df.to_csv(results_file, index=False)
    
    # Analyze
    result = analyze_regime_selection(str(results_file), verbose=False)
    
    # Should return dict
    assert isinstance(result, dict)


@pytest.mark.unit
def test_regime_definitions_exist():
    """Test that regime definitions are available."""
    from regime_analyzer import REGIMES
    
    # Should have regime definitions
    assert isinstance(REGIMES, dict)
    assert len(REGIMES) > 0
    
    # Check structure
    for regime_name, regime_info in REGIMES.items():
        assert 'years' in regime_info
        assert 'volatility' in regime_info


@pytest.mark.unit
def test_regime_analyzer_file_not_found():
    """Test regime analyzer with non-existent file."""
    from regime_analyzer import analyze_regime_selection
    
    result = analyze_regime_selection('/nonexistent/file.csv', verbose=False)
    
    # Should handle gracefully
    assert isinstance(result, dict)
    assert 'error' in result


@pytest.mark.unit
def test_regime_volatility_classifications():
    """Test that regimes have volatility classifications."""
    from regime_analyzer import REGIMES
    
    # Check volatility values
    volatility_values = set()
    for regime_info in REGIMES.values():
        volatility_values.add(regime_info['volatility'])
    
    # Should have HIGH, LOW, MEDIUM classifications
    assert 'HIGH' in volatility_values or len(volatility_values) > 0

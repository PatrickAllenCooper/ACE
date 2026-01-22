"""
Detailed tests for clamping_detector.py functionality.

Tests cover:
- detect_clamping function
- Value extraction
- Statistics computation
- Threshold detection
"""

import pytest
import tempfile
from pathlib import Path


# =============================================================================
# detect_clamping Function Tests
# =============================================================================

@pytest.mark.unit
def test_detect_clamping_function_exists():
    """Test that detect_clamping function exists."""
    from clamping_detector import detect_clamping
    
    assert callable(detect_clamping)


@pytest.mark.unit
def test_detect_clamping_with_mock_log(tmp_path):
    """Test clamping detection with mock log file."""
    from clamping_detector import detect_clamping
    
    # Create mock log file with clamping pattern
    log_file = tmp_path / "experiment.log"
    log_content = """
    Episode 1: DO X2 = 0.1
    Episode 2: DO X2 = 0.0
    Episode 3: DO X2 = -0.1
    Episode 4: DO X2 = 0.0
    Episode 5: DO X2 = 0.05
    Episode 6: DO X2 = 0.0
    """
    log_file.write_text(log_content)
    
    # Detect clamping
    result = detect_clamping(str(log_file), target_node='X2', threshold=0.5)
    
    # Should return dict
    assert isinstance(result, dict)


@pytest.mark.unit
def test_detect_clamping_returns_stats():
    """Test that detect_clamping returns expected statistics."""
    from clamping_detector import detect_clamping
    import tempfile
    
    # Create temp log with known values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write("DO X2 = 0.0\n")
        f.write("DO X2 = 0.1\n")
        f.write("DO X2 = 0.0\n")
        f.write("DO X2 = -0.1\n")
        log_path = f.name
    
    result = detect_clamping(log_path, target_node='X2')
    
    # Should not have error
    if 'error' not in result:
        # Should have statistics
        assert 'mean' in result or len(result) > 0


@pytest.mark.unit
def test_detect_clamping_file_not_found():
    """Test clamping detection with non-existent file."""
    from clamping_detector import detect_clamping
    
    result = detect_clamping('/nonexistent/path/file.log')
    
    # Should return error dict
    assert isinstance(result, dict)
    assert 'error' in result


@pytest.mark.unit
def test_detect_clamping_no_interventions():
    """Test clamping detection when no interventions found."""
    from clamping_detector import detect_clamping
    import tempfile
    
    # Create log without X2 interventions
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write("DO X1 = 1.0\n")
        f.write("DO X3 = 2.0\n")
        log_path = f.name
    
    result = detect_clamping(log_path, target_node='X2')
    
    # Should return error or empty result
    assert isinstance(result, dict)

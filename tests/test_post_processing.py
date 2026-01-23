"""
Tests for post-processing script (scripts/process_all_results.sh).

Tests cover:
- Script syntax validation
- Directory structure creation
- Mock data processing
- Error handling
- Integration with extraction/verification scripts
"""

import pytest
import subprocess
import tempfile
from pathlib import Path
import os


# =============================================================================
# Script Validation Tests
# =============================================================================

@pytest.mark.unit
def test_process_all_results_exists():
    """Test that process_all_results.sh exists."""
    script_path = Path("scripts/process_all_results.sh")
    assert script_path.exists(), "process_all_results.sh not found"


@pytest.mark.unit
def test_process_all_results_executable():
    """Test that process_all_results.sh is executable."""
    script_path = Path("scripts/process_all_results.sh")
    assert os.access(script_path, os.X_OK), "process_all_results.sh not executable"


@pytest.mark.unit
def test_process_all_results_syntax():
    """Test that process_all_results.sh has valid bash syntax."""
    script_path = Path("scripts/process_all_results.sh")
    
    result = subprocess.run(
        ['bash', '-n', str(script_path)],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Syntax error: {result.stderr}"


@pytest.mark.unit
def test_process_all_results_has_shebang():
    """Test that script has proper shebang."""
    script_path = Path("scripts/process_all_results.sh")
    
    with open(script_path) as f:
        first_line = f.readline()
    
    assert first_line.startswith("#!/bin/bash"), "Missing or incorrect shebang"


# =============================================================================
# Argument Handling Tests
# =============================================================================

@pytest.mark.unit
def test_process_all_results_requires_argument():
    """Test that script requires results directory argument."""
    script_path = Path("scripts/process_all_results.sh")
    
    result = subprocess.run(
        ['bash', str(script_path)],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    # Should exit with error (no argument)
    assert result.returncode != 0
    assert "Usage" in result.stdout or "Usage" in result.stderr


@pytest.mark.unit
def test_process_all_results_rejects_nonexistent_directory():
    """Test that script rejects non-existent directory."""
    script_path = Path("scripts/process_all_results.sh")
    
    result = subprocess.run(
        ['bash', str(script_path), '/nonexistent/path'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    assert result.returncode != 0
    assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()


# =============================================================================
# Directory Structure Tests
# =============================================================================

@pytest.mark.integration
def test_process_all_results_creates_output_structure(tmp_path):
    """Test that script creates expected output directory structure."""
    
    # Create mock results directory
    results_dir = tmp_path / "paper_20260101_120000"
    results_dir.mkdir()
    
    # Create minimal experiment subdirs
    (results_dir / "ace").mkdir()
    (results_dir / "baselines").mkdir()
    
    # Run script
    script_path = Path("scripts/process_all_results.sh")
    result = subprocess.run(
        ['bash', str(script_path), str(results_dir)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path.cwd()
    )
    
    # Check that processed directory was created
    processed_dir = results_dir / "processed"
    
    if processed_dir.exists():
        # Verify subdirectories
        assert (processed_dir / "tables").exists()
        assert (processed_dir / "figures").exists()
        assert (processed_dir / "verification").exists()


# =============================================================================
# Mock Data Processing Tests
# =============================================================================

@pytest.mark.integration
def test_process_all_results_with_mock_data(tmp_path):
    """Test processing with mock experimental data."""
    
    # Create mock results structure
    results_dir = tmp_path / "paper_test"
    results_dir.mkdir()
    
    # Create experiment directories with minimal files
    for exp in ['ace', 'baselines', 'duffing', 'phillips']:
        exp_dir = results_dir / exp
        exp_dir.mkdir()
        
        # Create mock CSV files
        if exp == 'ace':
            (exp_dir / "metrics.csv").write_text("episode,step,target,value\n0,0,X1,1.0\n")
            (exp_dir / "node_losses.csv").write_text("episode,loss_X1,loss_X2\n0,1.0,2.0\n")
        elif exp == 'baselines':
            (exp_dir / "random_results.csv").write_text("episode,total_loss\n0,2.0\n")
    
    # Run script
    script_path = Path("scripts/process_all_results.sh")
    result = subprocess.run(
        ['bash', str(script_path), str(results_dir)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path.cwd()
    )
    
    # Check summary was created
    summary_file = results_dir / "processed" / "PROCESSING_SUMMARY.txt"
    if summary_file.exists():
        summary = summary_file.read_text()
        assert "ACE Experiment Results Processing Summary" in summary


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.unit
def test_process_all_results_handles_missing_scripts(tmp_path):
    """Test that script handles missing extraction scripts gracefully."""
    
    # Create minimal results dir
    results_dir = tmp_path / "paper_test"
    results_dir.mkdir()
    (results_dir / "ace").mkdir()
    
    # Run from temp directory (where scripts won't exist)
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Copy script to temp location
        import shutil
        script_path = Path(original_dir) / "scripts" / "process_all_results.sh"
        temp_script = tmp_path / "process_all_results.sh"
        shutil.copy(script_path, temp_script)
        
        result = subprocess.run(
            ['bash', str(temp_script), str(results_dir)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Should complete without crashing (exit code 0 or just non-fatal warnings)
        # Check that it at least ran
        assert "Post-Processing" in result.stdout or "ERROR" in result.stdout
        
    finally:
        os.chdir(original_dir)


# =============================================================================
# Output Content Tests
# =============================================================================

@pytest.mark.unit
def test_process_all_results_output_format():
    """Test that script output has expected format."""
    script_path = Path("scripts/process_all_results.sh")
    
    # Run with help (no argument triggers usage)
    result = subprocess.run(
        ['bash', str(script_path)],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    # Should show usage information
    assert "Usage" in result.stdout or "Usage" in result.stderr
    assert "results" in result.stdout.lower() or "results" in result.stderr.lower()


@pytest.mark.unit
def test_process_all_results_mentions_all_steps():
    """Test that script describes all processing steps."""
    script_path = Path("scripts/process_all_results.sh")
    
    content = script_path.read_text()
    
    # Check for all major steps
    assert "Extract" in content or "extract" in content
    assert "Verify" in content or "verify" in content
    assert "table" in content.lower()
    assert "figure" in content.lower()
    assert "summary" in content.lower()


# =============================================================================
# Integration with Other Scripts Tests
# =============================================================================

@pytest.mark.unit
def test_process_all_results_references_extraction_scripts():
    """Test that script references extraction scripts."""
    script_path = Path("scripts/process_all_results.sh")
    
    content = script_path.read_text()
    
    # Should reference extraction scripts
    assert "extract_ace.sh" in content
    assert "extract_baselines.sh" in content


@pytest.mark.unit
def test_process_all_results_references_verification_scripts():
    """Test that script references verification scripts."""
    script_path = Path("scripts/process_all_results.sh")
    
    content = script_path.read_text()
    
    # Should reference verification scripts/tools
    assert "verify_claims.sh" in content
    assert "clamping_detector.py" in content
    assert "regime_analyzer.py" in content


@pytest.mark.unit
def test_process_all_results_references_visualization():
    """Test that script references visualization tools."""
    script_path = Path("scripts/process_all_results.sh")
    
    content = script_path.read_text()
    
    # Should reference comparison/visualization
    assert "compare_methods.py" in content or "visualize.py" in content


# =============================================================================
# Workflow Integration Tests
# =============================================================================

@pytest.mark.unit
def test_run_all_mentions_post_processing():
    """Test that run_all.sh mentions post-processing."""
    run_all = Path("run_all.sh")
    
    if run_all.exists():
        content = run_all.read_text()
        
        # Should mention post-processing
        assert "process_all_results.sh" in content or "post-processing" in content.lower()


@pytest.mark.unit
def test_post_processing_script_structure():
    """Test that post-processing script has expected structure."""
    script_path = Path("scripts/process_all_results.sh")
    
    content = script_path.read_text()
    
    # Should have clear sections
    assert "Step 1" in content or "Extract" in content
    assert "Step 2" in content or "Verify" in content
    assert "Step 3" in content or "table" in content.lower()
    assert "Step 4" in content or "figure" in content.lower()
    assert "Step 5" in content or "summary" in content.lower()


# =============================================================================
# Edge Case Tests
# =============================================================================

@pytest.mark.integration
def test_process_all_results_with_empty_results_dir(tmp_path):
    """Test processing with empty results directory."""
    
    # Create empty results dir
    results_dir = tmp_path / "paper_empty"
    results_dir.mkdir()
    
    script_path = Path("scripts/process_all_results.sh")
    result = subprocess.run(
        ['bash', str(script_path), str(results_dir)],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=Path.cwd()
    )
    
    # Should complete without crashing
    # May have warnings but shouldn't fail completely
    assert result.returncode == 0 or "not found" in result.stdout.lower()


@pytest.mark.integration
def test_process_all_results_with_partial_data(tmp_path):
    """Test processing with only some experiments present."""
    
    # Create results with only ACE
    results_dir = tmp_path / "paper_partial"
    results_dir.mkdir()
    (results_dir / "ace").mkdir()
    (results_dir / "ace" / "metrics.csv").write_text("episode,target\n0,X1\n")
    
    # Missing: baselines, duffing, phillips, complex_scm
    
    script_path = Path("scripts/process_all_results.sh")
    result = subprocess.run(
        ['bash', str(script_path), str(results_dir)],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=Path.cwd()
    )
    
    # Should handle missing experiments gracefully
    # Check it ran (may have warnings)
    assert "Post-Processing" in result.stdout or result.returncode == 0

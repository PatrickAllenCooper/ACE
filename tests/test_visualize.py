"""
Unit tests for visualization functions.

Tests cover:
- Data loading
- Dashboard creation (without display)
- Plot generation
- File saving
- Edge cases
"""

import pytest
import torch
import pandas as pd
import tempfile
import os
from pathlib import Path


# =============================================================================
# Data Loading Tests
# =============================================================================

@pytest.mark.unit
def test_load_run_data_basic(tmp_path):
    """Test loading run data from directory."""
    from visualize import load_run_data
    
    # Create test directory with CSV files
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    
    # Create test CSV files
    node_losses_df = pd.DataFrame({
        'episode': [0, 1, 2],
        'step': [0, 0, 0],
        'loss_X1': [1.0, 0.8, 0.6],
        'loss_X2': [2.0, 1.5, 1.2],
        'loss_X3': [3.0, 2.5, 2.0],
    })
    node_losses_df.to_csv(run_dir / "node_losses.csv", index=False)
    
    metrics_df = pd.DataFrame({
        'episode': [0, 1],
        'step': [0, 0],
        'target': ['X1', 'X2'],
        'value': [1.0, 2.0]
    })
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    
    # Load data
    data = load_run_data(str(run_dir))
    
    # Should load both files
    assert 'node_losses' in data
    assert 'metrics' in data
    assert isinstance(data['node_losses'], pd.DataFrame)
    assert isinstance(data['metrics'], pd.DataFrame)


@pytest.mark.unit
def test_load_run_data_missing_files(tmp_path):
    """Test loading data when some files are missing."""
    from visualize import load_run_data
    
    # Empty directory
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    
    # Should handle gracefully
    data = load_run_data(str(run_dir))
    
    # Should return empty dict or dict with missing keys
    assert isinstance(data, dict)


# =============================================================================
# Success Dashboard Tests (without display)
# =============================================================================

@pytest.mark.unit
def test_create_success_dashboard_basic(tmp_path):
    """Test dashboard creation without displaying."""
    from visualize import create_success_dashboard
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Create test data
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    
    # Minimal node losses data
    node_losses_df = pd.DataFrame({
        'episode': [0, 1, 2],
        'step': [0, 0, 0],
        'loss_X1': [1.0, 0.8, 0.6],
        'loss_X2': [2.0, 1.5, 1.2],
        'loss_X3': [3.0, 2.5, 2.0],
        'loss_X4': [1.0, 0.9, 0.8],
        'loss_X5': [1.5, 1.3, 1.1],
    })
    node_losses_df.to_csv(run_dir / "node_losses.csv", index=False)
    
    metrics_df = pd.DataFrame({
        'episode': [0, 1, 2],
        'step': [0, 0, 0],
        'target': ['X1', 'X2', 'X3'],
        'value': [1.0, 2.0, 1.5]
    })
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    
    # Load and create dashboard
    data = {'node_losses': node_losses_df, 'metrics': metrics_df}
    output_path = tmp_path / "dashboard.png"
    
    # Should create without error
    create_success_dashboard(data, str(run_dir), output_path=str(output_path))
    
    # Output file should be created
    assert output_path.exists()


# =============================================================================
# Utility Function Tests
# =============================================================================

@pytest.mark.unit
def test_visualize_has_required_functions():
    """Test that visualize module has required functions."""
    import visualize
    
    assert hasattr(visualize, 'load_run_data')
    assert hasattr(visualize, 'create_success_dashboard')


@pytest.mark.unit
def test_visualize_imports():
    """Test that visualize module imports correctly."""
    import visualize
    
    # Should import without error
    assert visualize is not None

"""
Unit tests for visualization functions from visualize.py.

Tests cover:
- Mechanism contrast plots
- Print summary
- Additional visualization utilities
"""

import pytest
import torch
import pandas as pd
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


# =============================================================================
# Print Summary Tests
# =============================================================================

@pytest.mark.unit
def test_print_summary_basic():
    """Test print_summary function."""
    from visualize import print_summary
    
    # Create test data
    data = {
        'node_losses': pd.DataFrame({
            'episode': [0, 1, 2],
            'loss_X1': [1.0, 0.8, 0.6],
            'loss_X2': [2.0, 1.5, 1.2],
            'loss_X3': [3.0, 2.5, 2.0],
        }),
        'metrics': pd.DataFrame({
            'episode': [0, 1, 2],
            'target': ['X1', 'X2', 'X3'],
            'value': [1.0, 2.0, 1.5]
        })
    }
    
    # Should run without error
    print_summary(data)


@pytest.mark.unit
def test_print_summary_with_minimal_data():
    """Test print_summary with minimal data."""
    from visualize import print_summary
    
    data = {
        'metrics': pd.DataFrame({
            'episode': [0],
            'target': ['X1'],
            'value': [1.0]
        })
    }
    
    # Should handle minimal data
    print_summary(data)


@pytest.mark.unit
def test_print_summary_with_empty_data():
    """Test print_summary with empty data dict."""
    from visualize import print_summary
    
    data = {}
    
    # Should handle gracefully
    print_summary(data)


# =============================================================================
# Mechanism Contrast Tests
# =============================================================================

@pytest.mark.unit
def test_create_mechanism_contrast_basic(ground_truth_scm, student_scm, tmp_path):
    """Test mechanism contrast visualization."""
    from visualize import create_mechanism_contrast
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Create visualization
    create_mechanism_contrast(ground_truth_scm, student_scm, str(results_dir))
    
    # Should create file
    output_file = results_dir / "mechanism_contrast_improved.png"
    assert output_file.exists()


@pytest.mark.unit
def test_create_mechanism_contrast_custom_filename(ground_truth_scm, student_scm, tmp_path):
    """Test mechanism contrast with custom filename."""
    from visualize import create_mechanism_contrast
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Custom filename
    create_mechanism_contrast(ground_truth_scm, student_scm, str(results_dir), 
                            filename="custom_plot.png")
    
    # Should create custom file
    output_file = results_dir / "custom_plot.png"
    assert output_file.exists()


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
def test_full_visualization_pipeline(ground_truth_scm, student_scm, tmp_path, seed_everything):
    """Test complete visualization pipeline."""
    from visualize import load_run_data, create_success_dashboard, create_mechanism_contrast
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    
    # Create test run directory
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    
    # Train student a bit
    learner = SCMLearner(student_scm, lr=0.01)
    for _ in range(3):
        data = ground_truth_scm.generate(100)
        learner.train_step({"data": data, "intervened": None}, n_epochs=10)
    
    # Create test data files
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
    
    # Load data
    data = load_run_data(str(run_dir))
    
    # Create visualizations
    create_success_dashboard(data, str(run_dir))
    create_mechanism_contrast(ground_truth_scm, student_scm, str(run_dir))
    
    # Should create both files
    assert (run_dir / "success_verification.png").exists()
    assert (run_dir / "mechanism_contrast_improved.png").exists()


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.unit
def test_visualization_handles_empty_metrics():
    """Test visualization with empty metrics."""
    from visualize import print_summary
    
    # Empty data (no metrics)
    data = {}
    
    # Should handle gracefully
    print_summary(data)

"""
Unit tests for plotting utility functions from ace_experiments.py.

Tests cover:
- save_plots
- visualize_contrast_save
- create_emergency_save_handler
"""

import pytest
import torch
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


# =============================================================================
# save_plots Tests
# =============================================================================

@pytest.mark.unit
def test_save_plots_basic(tmp_path):
    """Test save_plots creates figure files."""
    from ace_experiments import save_plots
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Sample data
    loss_history = [10.0, 8.0, 6.0, 5.0, 4.0]
    reward_history = [1.0, 2.0, 3.0, 4.0, 5.0]
    targets = ['X1', 'X2', 'X3', 'X1', 'X2']
    values = [1.0, 2.0, 1.5, 0.5, 2.5]
    nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    
    # Save plots
    save_plots(str(results_dir), loss_history, reward_history, targets, values, nodes)
    
    # Should create training_curves.png
    assert (results_dir / "training_curves.png").exists()


@pytest.mark.unit
def test_save_plots_with_empty_history(tmp_path):
    """Test save_plots with minimal data."""
    from ace_experiments import save_plots
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Minimal data
    loss_history = [10.0]
    reward_history = [1.0]
    targets = ['X1']
    values = [1.0]
    nodes = ['X1', 'X2', 'X3']
    
    # Should handle minimal data
    save_plots(str(results_dir), loss_history, reward_history, targets, values, nodes)
    
    assert (results_dir / "training_curves.png").exists()


@pytest.mark.unit
def test_save_plots_creates_multiple_figures(tmp_path):
    """Test that save_plots creates expected figures."""
    from ace_experiments import save_plots
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Good amount of data
    n = 20
    loss_history = [10.0 - i*0.3 for i in range(n)]
    reward_history = [float(i) for i in range(n)]
    targets = ['X1', 'X2', 'X3', 'X4', 'X5'] * 4
    values = [float(i % 5) for i in range(n)]
    nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    
    save_plots(str(results_dir), loss_history, reward_history, targets, values, nodes)
    
    # Check files created
    assert (results_dir / "training_curves.png").exists()


# =============================================================================
# visualize_contrast_save Tests
# =============================================================================

@pytest.mark.unit
def test_visualize_contrast_save_basic(ground_truth_scm, student_scm, tmp_path, seed_everything):
    """Test visualize_contrast_save creates figure."""
    from ace_experiments import visualize_contrast_save
    
    seed_everything(42)
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Create visualization
    visualize_contrast_save(ground_truth_scm, student_scm, str(results_dir))
    
    # Should create mechanism_contrast.png
    assert (results_dir / "mechanism_contrast.png").exists()


@pytest.mark.unit
def test_visualize_contrast_save_after_training(ground_truth_scm, student_scm, tmp_path, seed_everything):
    """Test visualize_contrast_save after some training."""
    from ace_experiments import visualize_contrast_save, SCMLearner
    
    seed_everything(42)
    
    # Train student a bit
    learner = SCMLearner(student_scm, lr=0.01)
    for _ in range(5):
        data = ground_truth_scm.generate(100)
        learner.train_step({"data": data, "intervened": None}, n_epochs=10)
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Visualize
    visualize_contrast_save(ground_truth_scm, student_scm, str(results_dir))
    
    assert (results_dir / "mechanism_contrast.png").exists()


# =============================================================================
# create_emergency_save_handler Tests
# =============================================================================

@pytest.mark.unit
def test_create_emergency_save_handler(ground_truth_scm, tmp_path):
    """Test emergency save handler creation."""
    from ace_experiments import create_emergency_save_handler
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    history_data = {
        'loss_history': [10.0, 8.0, 6.0],
        'reward_history': [1.0, 2.0, 3.0],
        'targets': ['X1', 'X2', 'X3'],
        'values': [1.0, 2.0, 1.5]
    }
    
    # Create handler
    handler = create_emergency_save_handler(str(results_dir), ground_truth_scm, history_data)
    
    # Should return a callable
    assert callable(handler)


@pytest.mark.unit
def test_emergency_save_handler_creation_only(ground_truth_scm, tmp_path):
    """Test emergency save handler creation (don't execute)."""
    from ace_experiments import create_emergency_save_handler
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    history_data = {
        'loss_history': [10.0, 8.0],
        'reward_history': [1.0, 2.0],
        'target_history': ['X1', 'X2'],  # Correct key name
        'value_history': [(('X1', 1.0)), (('X2', 2.0))]
    }
    
    # Create handler
    handler = create_emergency_save_handler(str(results_dir), ground_truth_scm, history_data)
    
    # Should return a callable
    assert callable(handler)


# =============================================================================
# ScientificCritic Tests (from ace_experiments.py)
# =============================================================================

@pytest.mark.unit
def test_scientific_critic_initialization_ace(ground_truth_scm):
    """Test ScientificCritic from ace_experiments.py."""
    from ace_experiments import ScientificCritic
    
    critic = ScientificCritic(ground_truth_scm)
    
    assert hasattr(critic, 'test_oracle')
    assert critic.test_oracle is ground_truth_scm

"""
Unit tests for utility functions from ace_experiments.py.

Tests cover:
- fit_root_distributions
- visualize_scm_graph
- save_checkpoint
- Helper utilities
"""

import pytest
import torch
import tempfile
from pathlib import Path


# =============================================================================
# fit_root_distributions Tests
# =============================================================================

@pytest.mark.unit
def test_fit_root_distributions_basic(student_scm, ground_truth_scm, seed_everything):
    """Test basic root fitting functionality."""
    from ace_experiments import fit_root_distributions, ScientificCritic
    
    seed_everything(42)
    
    critic = ScientificCritic(ground_truth_scm)
    root_nodes = ['X1', 'X4']
    
    # Fit roots
    fit_root_distributions(
        student_scm=student_scm,
        ground_truth_scm=ground_truth_scm,
        critic=critic,
        root_nodes=root_nodes,
        n_samples=500,
        epochs=50
    )
    
    # Should complete without error


@pytest.mark.unit
@pytest.mark.statistical
def test_fit_root_distributions_improves_roots(student_scm, ground_truth_scm, seed_everything):
    """Test that root fitting improves root node estimates."""
    from ace_experiments import fit_root_distributions, ScientificCritic
    
    seed_everything(42)
    
    critic = ScientificCritic(ground_truth_scm)
    root_nodes = ['X1', 'X4']
    
    # Get initial X1 parameter
    initial_x1_mu = student_scm.mechanisms['X1']['mu'].clone()
    
    # Fit roots
    fit_root_distributions(
        student_scm=student_scm,
        ground_truth_scm=ground_truth_scm,
        critic=critic,
        root_nodes=root_nodes,
        n_samples=1000,
        epochs=100
    )
    
    # X1 mu should have changed
    final_x1_mu = student_scm.mechanisms['X1']['mu']
    
    # Should move toward true mean (0)
    assert abs(final_x1_mu.item()) < abs(initial_x1_mu.item()) + 0.5


@pytest.mark.unit
def test_fit_root_distributions_with_dedicated_learner(student_scm, ground_truth_scm, seed_everything):
    """Test root fitting with dedicated learner."""
    from ace_experiments import fit_root_distributions, ScientificCritic, DedicatedRootLearner
    
    seed_everything(42)
    
    critic = ScientificCritic(ground_truth_scm)
    root_nodes = ['X1', 'X4']
    dedicated_learner = DedicatedRootLearner(root_nodes)  # Only takes root_nodes
    
    # Fit with dedicated learner
    fit_root_distributions(
        student_scm=student_scm,
        ground_truth_scm=ground_truth_scm,
        critic=critic,
        root_nodes=root_nodes,
        n_samples=500,
        epochs=50,
        dedicated_learner=dedicated_learner
    )
    
    # Should complete


# =============================================================================
# visualize_scm_graph Tests (without display)
# =============================================================================

@pytest.mark.unit
def test_visualize_scm_graph_creates_file(ground_truth_scm, tmp_path):
    """Test SCM graph visualization creates file."""
    from ace_experiments import visualize_scm_graph
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Visualize
    visualize_scm_graph(ground_truth_scm, str(results_dir))
    
    # Should create scm_graph.png
    graph_file = results_dir / "scm_graph.png"
    assert graph_file.exists()


@pytest.mark.unit
def test_visualize_scm_graph_with_losses(ground_truth_scm, tmp_path):
    """Test SCM graph visualization with node losses."""
    from ace_experiments import visualize_scm_graph
    import matplotlib
    matplotlib.use('Agg')
    
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    node_losses = {'X1': 0.5, 'X2': 1.0, 'X3': 2.0, 'X4': 0.3, 'X5': 0.8}
    
    # Visualize with losses
    visualize_scm_graph(ground_truth_scm, str(results_dir), node_losses=node_losses)
    
    # Should create file
    assert (results_dir / "scm_graph.png").exists()


# =============================================================================
# save_checkpoint Tests
# =============================================================================

@pytest.mark.unit
def test_save_checkpoint_basic(tmp_path):
    """Test checkpoint saving to checkpoints/ directory."""
    from ace_experiments import save_checkpoint, TransformerPolicy, ExperimentalDSL
    import os
    
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    
    # Create dummy policy
    nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    dsl = ExperimentalDSL(nodes)
    policy = TransformerPolicy(dsl, torch.device('cpu'))
    optimizer = torch.optim.Adam(policy.parameters())
    
    # Change to tmp directory for test isolation
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Save checkpoint
        save_checkpoint(
            run_dir=str(run_dir),
            episode=10,
            policy_net=policy,
            optimizer=optimizer,
            loss_history=[1.0, 0.8, 0.6],
            reward_history=[5.0, 6.0, 7.0],
            recent_actions=['DO X1 = 1.0', 'DO X2 = 2.0']
        )
        
        # Should create checkpoint in checkpoints/run_test/
        checkpoint_dir = tmp_path / "checkpoints" / "run_test"
        assert checkpoint_dir.exists()
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoint_files) > 0
    finally:
        os.chdir(original_dir)


# =============================================================================
# ScientificCritic Tests (from ace_experiments.py)
# =============================================================================

@pytest.mark.unit
def test_scientific_critic_from_ace_initialization(ground_truth_scm):
    """Test ScientificCritic from ace_experiments.py initialization."""
    from ace_experiments import ScientificCritic
    
    critic = ScientificCritic(ground_truth_scm)
    
    assert hasattr(critic, 'test_oracle')
    assert critic.test_oracle is ground_truth_scm

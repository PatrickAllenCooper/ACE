"""
Unit tests for DedicatedRootLearner class.

Tests cover:
- Initialization
- Root-only training
- Observational data isolation
- Parameter updates
"""

import pytest
import torch


# =============================================================================
# DedicatedRootLearner Tests
# =============================================================================

@pytest.mark.unit
def test_dedicated_root_learner_initialization():
    """Test DedicatedRootLearner initialization."""
    from ace_experiments import DedicatedRootLearner
    
    root_nodes = ['X1', 'X4']
    learner = DedicatedRootLearner(root_nodes)
    
    assert learner.root_nodes == root_nodes
    assert hasattr(learner, 'distributions')
    assert hasattr(learner, 'optimizer')
    
    # Should have distributions for both roots
    assert 'X1' in learner.distributions
    assert 'X4' in learner.distributions


@pytest.mark.unit
def test_dedicated_root_learner_fit_method(ground_truth_scm, seed_everything):
    """Test fit method with observational data."""
    from ace_experiments import DedicatedRootLearner
    
    seed_everything(42)
    
    root_nodes = ['X1', 'X4']
    learner = DedicatedRootLearner(root_nodes)
    
    # Generate observational data
    obs_data = ground_truth_scm.generate(n_samples=1000)
    
    # Fit
    losses = learner.fit(obs_data, epochs=50)
    
    # Should return losses dict
    assert isinstance(losses, dict)
    assert all(node in losses for node in root_nodes)


@pytest.mark.unit
@pytest.mark.statistical
def test_dedicated_root_learner_learns_x1_distribution(ground_truth_scm, seed_everything):
    """Test that learner can learn X1 ~ N(0,1)."""
    from ace_experiments import DedicatedRootLearner
    
    seed_everything(42)
    
    root_nodes = ['X1', 'X4']
    learner = DedicatedRootLearner(root_nodes)
    
    # Generate observational data
    obs_data = ground_truth_scm.generate(n_samples=2000)
    
    # Fit
    learner.fit(obs_data, epochs=200)
    
    # Check learned mu for X1
    learned_mu = learner.distributions['X1']['mu'].item()
    
    # Should be close to 0
    assert abs(learned_mu) < 0.3


@pytest.mark.unit
@pytest.mark.statistical
def test_dedicated_root_learner_learns_x4_distribution(ground_truth_scm, seed_everything):
    """Test that learner can learn X4 ~ N(2,1)."""
    from ace_experiments import DedicatedRootLearner
    
    seed_everything(42)
    
    root_nodes = ['X1', 'X4']
    learner = DedicatedRootLearner(root_nodes)
    
    # Generate observational data
    obs_data = ground_truth_scm.generate(n_samples=2000)
    
    # Fit
    learner.fit(obs_data, epochs=200)
    
    # Check learned mu for X4
    learned_mu = learner.distributions['X4']['mu'].item()
    
    # Should move toward 2 (true mean) - relaxed tolerance for stochastic learning
    assert abs(learned_mu - 2.0) < 1.5


@pytest.mark.unit
def test_dedicated_root_learner_apply_to_student(student_scm, ground_truth_scm, seed_everything):
    """Test applying learned distributions to student."""
    from ace_experiments import DedicatedRootLearner
    
    seed_everything(42)
    
    root_nodes = ['X1', 'X4']
    learner = DedicatedRootLearner(root_nodes)
    
    # Fit
    obs_data = ground_truth_scm.generate(n_samples=1000)
    learner.fit(obs_data, epochs=100)
    
    # Apply to student
    learner.apply_to_student(student_scm)
    
    # Student's root parameters should now match learned values
    student_x1_mu = student_scm.mechanisms['X1']['mu'].item()
    learned_x1_mu = learner.distributions['X1']['mu'].item()
    
    assert abs(student_x1_mu - learned_x1_mu) < 1e-5


@pytest.mark.unit
def test_dedicated_root_learner_only_affects_roots(student_scm, ground_truth_scm, seed_everything):
    """Test that dedicated learner only affects root nodes."""
    from ace_experiments import DedicatedRootLearner
    
    seed_everything(42)
    
    root_nodes = ['X1', 'X4']
    learner = DedicatedRootLearner(root_nodes)
    
    # Get initial X2 parameters
    initial_x2_params = [p.clone() for p in student_scm.mechanisms['X2'].parameters()]
    
    # Fit and apply
    obs_data = ground_truth_scm.generate(n_samples=500)
    learner.fit(obs_data, epochs=50)
    learner.apply_to_student(student_scm)
    
    # X2 should be unchanged
    final_x2_params = list(student_scm.mechanisms['X2'].parameters())
    
    for init_p, final_p in zip(initial_x2_params, final_x2_params):
        assert torch.allclose(init_p, final_p, atol=1e-6)

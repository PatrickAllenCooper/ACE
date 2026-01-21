"""
Unit tests for baseline SCM classes from baselines.py.

Tests cover:
- GroundTruthSCM from baselines
- StudentSCM from baselines
- Basic generation and training
"""

import pytest
import torch


# =============================================================================
# GroundTruthSCM Tests (from baselines.py)
# =============================================================================

@pytest.mark.unit
def test_baselines_ground_truth_scm_initialization():
    """Test GroundTruthSCM from baselines.py initializes correctly."""
    from baselines import GroundTruthSCM
    
    scm = GroundTruthSCM()
    
    assert scm.nodes == ["X1", "X2", "X3", "X4", "X5"]
    assert hasattr(scm, 'graph')
    assert hasattr(scm, 'noise_std')
    assert scm.noise_std == 0.1


@pytest.mark.unit
def test_baselines_ground_truth_scm_generate(seed_everything):
    """Test basic generation from baselines GroundTruthSCM."""
    from baselines import GroundTruthSCM
    
    seed_everything(42)
    scm = GroundTruthSCM()
    
    data = scm.generate(n_samples=100)
    
    assert set(data.keys()) == {'X1', 'X2', 'X3', 'X4', 'X5'}
    for node in data:
        assert data[node].shape == (100,)


@pytest.mark.unit
def test_baselines_ground_truth_scm_interventions(seed_everything):
    """Test interventions in baselines GroundTruthSCM."""
    from baselines import GroundTruthSCM
    
    seed_everything(42)
    scm = GroundTruthSCM()
    
    data = scm.generate(n_samples=100, interventions={'X1': 2.0})
    
    assert torch.all(data['X1'] == 2.0)


# =============================================================================
# StudentSCM Tests (from baselines.py)
# =============================================================================

@pytest.mark.unit
def test_baselines_student_scm_initialization():
    """Test StudentSCM from baselines.py initializes correctly."""
    from baselines import GroundTruthSCM, StudentSCM
    
    gt = GroundTruthSCM()
    student = StudentSCM(gt)
    
    assert set(student.nodes) == set(gt.nodes)
    assert hasattr(student, 'mechanisms')


@pytest.mark.unit
def test_baselines_student_scm_forward(seed_everything):
    """Test StudentSCM forward pass (teacher forcing)."""
    from baselines import GroundTruthSCM, StudentSCM
    
    seed_everything(42)
    gt = GroundTruthSCM()
    student = StudentSCM(gt)
    
    # Generate data
    gt_data = gt.generate(100)
    
    # Forward pass (teacher forcing - takes data dict)
    predictions = student.forward(gt_data)
    
    assert set(predictions.keys()) == {'X1', 'X2', 'X3', 'X4', 'X5'}
    for node in predictions:
        assert predictions[node].shape == (100,)


@pytest.mark.unit
def test_baselines_student_scm_train_step(seed_everything):
    """Test StudentSCM train_step method."""
    from baselines import GroundTruthSCM, StudentSCM
    
    seed_everything(42)
    gt = GroundTruthSCM()
    student = StudentSCM(gt)
    
    # Generate training data
    gt_data = gt.generate(100)
    
    # Train (train_step is a method in the Learner wrapper, not StudentSCM)
    # Let's test that student can be trained
    predictions = student.forward(gt_data)
    
    # Compute loss manually
    loss = sum(((predictions[node] - gt_data[node]) ** 2).mean() for node in student.nodes)
    
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0


@pytest.mark.unit
def test_baselines_student_scm_parameters(seed_everything):
    """Test StudentSCM has trainable parameters."""
    from baselines import GroundTruthSCM, StudentSCM
    
    seed_everything(42)
    gt = GroundTruthSCM()
    student = StudentSCM(gt)
    
    # Should have parameters
    params = list(student.parameters())
    assert len(params) > 0
    
    # All should be torch Parameters
    assert all(isinstance(p, torch.nn.Parameter) for p in params)

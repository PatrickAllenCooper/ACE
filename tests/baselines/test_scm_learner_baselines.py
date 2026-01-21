"""
Unit tests for SCMLearner class from baselines.py.

Tests cover:
- Initialization
- Training step
- Observational training
- Intervention masking
- Buffer management
"""

import pytest
import torch


# =============================================================================
# SCMLearner Tests (from baselines.py)
# =============================================================================

@pytest.mark.unit
def test_scm_learner_initialization():
    """Test SCMLearner initialization from baselines."""
    from baselines import GroundTruthSCM, StudentSCM, SCMLearner
    
    oracle = GroundTruthSCM()
    student = StudentSCM(oracle)
    learner = SCMLearner(student)
    
    assert hasattr(learner, 'student')
    assert learner.student is student
    assert hasattr(learner, 'optimizer')


@pytest.mark.unit
def test_scm_learner_train_step(seed_everything):
    """Test train_step method."""
    from baselines import GroundTruthSCM, StudentSCM, SCMLearner
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    student = StudentSCM(oracle)
    learner = SCMLearner(student)
    
    # Generate data
    data = oracle.generate(100)
    
    # Train
    loss = learner.train_step(data, n_epochs=10)
    
    assert isinstance(loss, float)
    assert loss >= 0


@pytest.mark.unit
def test_scm_learner_train_with_intervention(seed_everything):
    """Test training with intervention masking."""
    from baselines import GroundTruthSCM, StudentSCM, SCMLearner
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    student = StudentSCM(oracle)
    learner = SCMLearner(student)
    
    # Generate interventional data
    data = oracle.generate(100, interventions={'X1': 2.0})
    
    # Train with intervention mask
    loss = learner.train_step(data, intervened='X1', n_epochs=10)
    
    assert isinstance(loss, float)
    assert loss >= 0


@pytest.mark.unit
def test_scm_learner_observational_train(seed_everything):
    """Test observational training method."""
    from baselines import GroundTruthSCM, StudentSCM, SCMLearner
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    student = StudentSCM(oracle)
    learner = SCMLearner(student)
    
    # Observational training
    learner.observational_train(oracle, n_samples=100, n_epochs=10)
    
    # Should complete without error
    # (method doesn't return value)


@pytest.mark.unit
def test_scm_learner_parameters_update(seed_everything):
    """Test that parameters update during training."""
    from baselines import GroundTruthSCM, StudentSCM, SCMLearner
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    student = StudentSCM(oracle)
    learner = SCMLearner(student)
    
    # Get initial parameters
    initial_params = [p.clone() for p in student.parameters()]
    
    # Train
    data = oracle.generate(100)
    learner.train_step(data, n_epochs=20)
    
    # Parameters should have changed
    final_params = list(student.parameters())
    
    changed = False
    for init_p, final_p in zip(initial_params, final_params):
        if not torch.allclose(init_p, final_p, atol=1e-5):
            changed = True
            break
    
    assert changed


@pytest.mark.unit
def test_scm_learner_multiple_train_steps(seed_everything):
    """Test multiple training steps."""
    from baselines import GroundTruthSCM, StudentSCM, SCMLearner
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    student = StudentSCM(oracle)
    learner = SCMLearner(student)
    
    # Multiple train steps
    for _ in range(5):
        data = oracle.generate(100)
        loss = learner.train_step(data, n_epochs=5)
        assert isinstance(loss, float)


@pytest.mark.unit
def test_scm_learner_intervention_masking_prevents_training(seed_everything):
    """Test that intervention masking works."""
    from baselines import GroundTruthSCM, StudentSCM, SCMLearner
    
    seed_everything(42)
    
    oracle = GroundTruthSCM()
    student = StudentSCM(oracle)
    learner = SCMLearner(student)
    
    # Get initial root parameters (X1)
    initial_mu = student.mechanisms['X1']['mu'].clone()
    
    # Train only with DO(X1) interventions
    for _ in range(10):
        data = oracle.generate(100, interventions={'X1': 5.0})
        learner.train_step(data, intervened='X1', n_epochs=10)
    
    # X1 parameters should not have changed much (masked)
    final_mu = student.mechanisms['X1']['mu']
    
    # Might change slightly, but should be similar
    assert torch.allclose(initial_mu, final_mu, atol=1.0)

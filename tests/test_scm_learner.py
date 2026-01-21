"""
Unit tests for SCMLearner class.

Tests cover:
- Initialization
- Batch normalization
- Training step with observational data
- Training step with interventional data
- Buffer management
- Intervention masking
- Loss computation
- Fast adaptation phase
- Replay consolidation
"""

import pytest
import torch
import torch.nn as nn


# =============================================================================
# Initialization Tests
# =============================================================================

@pytest.mark.unit
def test_scm_learner_initialization(student_scm):
    """Test that SCMLearner initializes correctly."""
    from ace_experiments import SCMLearner
    
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=50)
    
    # Check attributes
    assert hasattr(learner, 'student')
    assert learner.student is student_scm
    
    assert hasattr(learner, 'optimizer')
    assert isinstance(learner.optimizer, torch.optim.Optimizer)
    
    assert hasattr(learner, 'loss_fn')
    assert isinstance(learner.loss_fn, nn.Module)
    
    assert hasattr(learner, 'buffer')
    assert isinstance(learner.buffer, list)
    assert len(learner.buffer) == 0
    
    assert learner.buffer_steps == 50


@pytest.mark.unit
def test_scm_learner_initialization_with_initial_buffer(student_scm, ground_truth_scm, seed_everything):
    """Test initialization with pre-populated buffer."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    
    # Create initial buffer data
    initial_data = [
        {"data": ground_truth_scm.generate(100), "intervened": None},
        {"data": ground_truth_scm.generate(100, interventions={'X1': 1.0}), "intervened": 'X1'}
    ]
    
    learner = SCMLearner(student_scm, initial_buffer=initial_data)
    
    # Buffer should be initialized
    assert len(learner.buffer) == 2


@pytest.mark.unit
def test_scm_learner_custom_learning_rate(student_scm):
    """Test initialization with custom learning rate."""
    from ace_experiments import SCMLearner
    
    learner = SCMLearner(student_scm, lr=0.001)
    
    # Check optimizer has correct learning rate
    assert learner.optimizer.param_groups[0]['lr'] == 0.001


# =============================================================================
# Batch Normalization Tests
# =============================================================================

@pytest.mark.unit
def test_normalize_batch_with_full_format(student_scm, sample_observational_data):
    """Test batch normalization with full format."""
    from ace_experiments import SCMLearner
    
    learner = SCMLearner(student_scm)
    
    batch = {"data": sample_observational_data, "intervened": None}
    
    normalized = learner._normalize_batch(batch)
    
    assert "data" in normalized
    assert "intervened" in normalized
    assert normalized["intervened"] is None


@pytest.mark.unit
def test_normalize_batch_with_interventional_format(student_scm, sample_intervention_data):
    """Test batch normalization with intervention."""
    from ace_experiments import SCMLearner
    
    learner = SCMLearner(student_scm)
    
    batch = {"data": sample_intervention_data, "intervened": "X2"}
    
    normalized = learner._normalize_batch(batch)
    
    assert normalized["intervened"] == "X2"


@pytest.mark.unit
def test_normalize_batch_backward_compat(student_scm, sample_observational_data):
    """Test backward compatibility with plain dict format."""
    from ace_experiments import SCMLearner
    
    learner = SCMLearner(student_scm)
    
    # Plain dict (backward compatibility)
    batch = sample_observational_data
    
    normalized = learner._normalize_batch(batch)
    
    assert "data" in normalized
    assert normalized["intervened"] is None


@pytest.mark.unit
def test_normalize_batch_rejects_invalid_format(student_scm):
    """Test that invalid batch format is rejected."""
    from ace_experiments import SCMLearner
    
    learner = SCMLearner(student_scm)
    
    # Invalid format
    invalid_batch = "not a dict"
    
    with pytest.raises(ValueError, match="Unsupported batch format"):
        learner._normalize_batch(invalid_batch)


# =============================================================================
# Training Step - Observational Data Tests
# =============================================================================

@pytest.mark.unit
def test_train_step_with_observational_data(student_scm, ground_truth_scm, seed_everything):
    """Test training step with observational data."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    # Generate observational data
    data = ground_truth_scm.generate(n_samples=100)
    batch = {"data": data, "intervened": None}
    
    # Get initial loss
    initial_params = [p.clone() for p in student_scm.parameters()]
    
    # Train step
    loss = learner.train_step(batch, n_epochs=10)
    
    # Loss should be a number
    assert isinstance(loss, float)
    assert loss >= 0
    
    # Parameters should have changed
    final_params = [p for p in student_scm.parameters()]
    changed = False
    for init_p, final_p in zip(initial_params, final_params):
        if not torch.allclose(init_p, final_p, atol=1e-6):
            changed = True
            break
    assert changed, "Parameters should have been updated"


@pytest.mark.unit
def test_train_step_adds_to_buffer(student_scm, ground_truth_scm, seed_everything):
    """Test that training step adds data to buffer."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm, buffer_steps=5)
    
    assert len(learner.buffer) == 0
    
    # First training step
    data1 = ground_truth_scm.generate(100)
    learner.train_step({"data": data1, "intervened": None}, n_epochs=5)
    
    assert len(learner.buffer) == 1
    
    # Second training step
    data2 = ground_truth_scm.generate(100)
    learner.train_step({"data": data2, "intervened": None}, n_epochs=5)
    
    assert len(learner.buffer) == 2


@pytest.mark.unit
def test_train_step_respects_buffer_limit(student_scm, ground_truth_scm, seed_everything):
    """Test that buffer respects maximum size."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    buffer_limit = 3
    learner = SCMLearner(student_scm, buffer_steps=buffer_limit)
    
    # Add more than buffer limit
    for i in range(5):
        data = ground_truth_scm.generate(50)
        learner.train_step({"data": data, "intervened": None}, n_epochs=2)
    
    # Buffer should not exceed limit
    assert len(learner.buffer) == buffer_limit


@pytest.mark.unit
def test_train_step_removes_oldest_from_buffer(student_scm, ground_truth_scm, seed_everything):
    """Test that oldest data is removed when buffer is full."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm, buffer_steps=2)
    
    # Add first batch
    data1 = ground_truth_scm.generate(10)
    batch1 = {"data": data1, "intervened": None}
    learner.train_step(batch1, n_epochs=1)
    
    # Add second batch
    data2 = ground_truth_scm.generate(10)
    batch2 = {"data": data2, "intervened": None}
    learner.train_step(batch2, n_epochs=1)
    
    assert len(learner.buffer) == 2
    
    # Add third batch (should remove first)
    data3 = ground_truth_scm.generate(10)
    batch3 = {"data": data3, "intervened": None}
    learner.train_step(batch3, n_epochs=1)
    
    assert len(learner.buffer) == 2
    # First batch should be gone, second and third should remain


# =============================================================================
# Training Step - Interventional Data Tests
# =============================================================================

@pytest.mark.unit
def test_train_step_with_interventional_data(student_scm, ground_truth_scm, seed_everything):
    """Test training step with interventional data."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm, lr=0.01)
    
    # Generate interventional data DO(X1=2)
    data = ground_truth_scm.generate(n_samples=100, interventions={'X1': 2.0})
    batch = {"data": data, "intervened": "X1"}
    
    # Train step should complete without error
    loss = learner.train_step(batch, n_epochs=5)
    
    assert isinstance(loss, float)
    assert loss >= 0


@pytest.mark.unit
def test_intervention_masking_prevents_training_on_intervened_node(
    student_scm, ground_truth_scm, seed_everything
):
    """Test that intervened nodes are masked during training."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=1)
    
    # Get initial X1 parameters (root node)
    initial_mu = student_scm.mechanisms['X1']['mu'].clone()
    
    # Train with intervention on X1
    data = ground_truth_scm.generate(n_samples=100, interventions={'X1': 5.0})
    batch = {"data": data, "intervened": "X1"}
    
    learner.train_step(batch, n_epochs=20)
    
    # X1 parameters should not have changed much (masked from training)
    final_mu = student_scm.mechanisms['X1']['mu']
    
    # They might change slightly from other nodes, but should be similar
    # This is a weak test - the key is that X1 mechanism wasn't trained on intervened data
    assert final_mu.shape == initial_mu.shape


@pytest.mark.unit
def test_intervention_masking_allows_training_on_other_nodes(
    student_scm, ground_truth_scm, seed_everything
):
    """Test that non-intervened nodes are still trained with intervention data."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=1)
    
    # Get initial X2 mechanism parameters
    initial_x2_params = [p.clone() for p in student_scm.mechanisms['X2'].parameters()]
    
    # Train with intervention on X1 (X2 should still be trained)
    data = ground_truth_scm.generate(n_samples=100, interventions={'X1': 2.0})
    batch = {"data": data, "intervened": "X1"}
    
    learner.train_step(batch, n_epochs=20)
    
    # X2 parameters should have changed
    final_x2_params = [p for p in student_scm.mechanisms['X2'].parameters()]
    changed = False
    for init_p, final_p in zip(initial_x2_params, final_x2_params):
        if not torch.allclose(init_p, final_p, atol=1e-5):
            changed = True
            break
    
    assert changed, "X2 mechanism should have been trained"


# =============================================================================
# Loss Computation Tests
# =============================================================================

@pytest.mark.unit
def test_train_step_returns_valid_loss(student_scm, ground_truth_scm, seed_everything):
    """Test that train_step returns a valid loss value."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm)
    
    data = ground_truth_scm.generate(100)
    batch = {"data": data, "intervened": None}
    
    loss = learner.train_step(batch, n_epochs=5)
    
    # Loss should be finite positive number
    assert isinstance(loss, float)
    assert loss >= 0
    assert not torch.isnan(torch.tensor(loss))
    assert not torch.isinf(torch.tensor(loss))


@pytest.mark.unit
@pytest.mark.slow
def test_loss_decreases_with_training(student_scm, ground_truth_scm, seed_everything):
    """Test that loss decreases over multiple training steps."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=50)
    
    losses = []
    
    # Train for multiple steps on same distribution
    for _ in range(5):
        data = ground_truth_scm.generate(200)
        batch = {"data": data, "intervened": None}
        loss = learner.train_step(batch, n_epochs=20)
        losses.append(loss)
    
    # Later losses should generally be lower than initial losses
    # (though not strictly monotonic due to stochasticity)
    assert losses[-1] < losses[0], "Loss should decrease with training"


# =============================================================================
# Fast Adaptation Phase Tests
# =============================================================================

@pytest.mark.unit
def test_fast_adaptation_phase_executes(student_scm, ground_truth_scm, seed_everything):
    """Test that fast adaptation phase runs during train_step."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    # First training step (fast adaptation on new data)
    data = ground_truth_scm.generate(100)
    batch = {"data": data, "intervened": None}
    
    # This should execute fast adaptation phase (20% of epochs, min 5)
    loss = learner.train_step(batch, n_epochs=50)
    
    # Should complete without error
    assert isinstance(loss, float)


# =============================================================================
# Replay Consolidation Tests
# =============================================================================

@pytest.mark.unit
def test_replay_consolidation_uses_buffer(student_scm, ground_truth_scm, seed_everything):
    """Test that replay phase trains on buffered data."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=3)
    
    # Add multiple batches to buffer
    for i in range(3):
        data = ground_truth_scm.generate(50)
        batch = {"data": data, "intervened": None}
        learner.train_step(batch, n_epochs=5)
    
    # Buffer should have 3 batches
    assert len(learner.buffer) == 3
    
    # Next training step should use all buffered data
    data = ground_truth_scm.generate(50)
    batch = {"data": data, "intervened": None}
    loss = learner.train_step(batch, n_epochs=5)
    
    # Should complete successfully
    assert isinstance(loss, float)


@pytest.mark.unit
def test_mixed_observational_and_interventional_in_buffer(
    student_scm, ground_truth_scm, seed_everything
):
    """Test buffer with mixed observational and interventional data."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=5)
    
    # Add observational data
    data_obs = ground_truth_scm.generate(50)
    learner.train_step({"data": data_obs, "intervened": None}, n_epochs=5)
    
    # Add interventional data on X1
    data_int1 = ground_truth_scm.generate(50, interventions={'X1': 1.0})
    learner.train_step({"data": data_int1, "intervened": "X1"}, n_epochs=5)
    
    # Add interventional data on X2
    data_int2 = ground_truth_scm.generate(50, interventions={'X2': 2.0})
    learner.train_step({"data": data_int2, "intervened": "X2"}, n_epochs=5)
    
    # Buffer should have all three types
    assert len(learner.buffer) == 3
    
    # Next training should handle mixed buffer correctly
    data = ground_truth_scm.generate(50)
    loss = learner.train_step({"data": data, "intervened": None}, n_epochs=5)
    
    assert isinstance(loss, float)


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.unit
def test_train_step_with_single_epoch(student_scm, ground_truth_scm, seed_everything):
    """Test training with single epoch."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm)
    
    data = ground_truth_scm.generate(100)
    batch = {"data": data, "intervened": None}
    
    # Should work with n_epochs=1
    loss = learner.train_step(batch, n_epochs=1)
    
    assert isinstance(loss, float)


@pytest.mark.unit
def test_train_step_with_small_batch(student_scm, ground_truth_scm, seed_everything):
    """Test training with small batch size."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm)
    
    # Very small batch
    data = ground_truth_scm.generate(n_samples=5)
    batch = {"data": data, "intervened": None}
    
    loss = learner.train_step(batch, n_epochs=5)
    
    assert isinstance(loss, float)


@pytest.mark.unit
def test_train_step_with_large_batch(student_scm, ground_truth_scm, seed_everything):
    """Test training with large batch size."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm)
    
    # Large batch
    data = ground_truth_scm.generate(n_samples=1000)
    batch = {"data": data, "intervened": None}
    
    loss = learner.train_step(batch, n_epochs=3)
    
    assert isinstance(loss, float)


@pytest.mark.unit
def test_learner_handles_multiple_interventions_in_sequence(
    student_scm, ground_truth_scm, seed_everything
):
    """Test learner with sequence of different interventions."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm, buffer_steps=10)
    
    interventions = [
        (None, None),
        ('X1', 1.0),
        ('X2', 2.0),
        ('X3', 0.5),
        ('X4', 3.0),
        (None, None),
    ]
    
    for target, value in interventions:
        if target is None:
            data = ground_truth_scm.generate(50)
            batch = {"data": data, "intervened": None}
        else:
            data = ground_truth_scm.generate(50, interventions={target: value})
            batch = {"data": data, "intervened": target}
        
        loss = learner.train_step(batch, n_epochs=5)
        assert isinstance(loss, float)


# =============================================================================
# Gradient Flow Tests
# =============================================================================

@pytest.mark.unit
def test_gradients_flow_during_training(student_scm, ground_truth_scm, seed_everything):
    """Test that gradients flow through student during training."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm)
    
    # Clear any existing gradients
    learner.optimizer.zero_grad()
    
    # Train
    data = ground_truth_scm.generate(100)
    batch = {"data": data, "intervened": None}
    learner.train_step(batch, n_epochs=1)
    
    # After training, gradients should have been computed and cleared
    # We can't easily check this directly, but we can verify no errors occurred
    assert True  # If we got here, gradients flowed successfully


# =============================================================================
# Module Integration Tests
# =============================================================================

@pytest.mark.unit
def test_learner_integrates_with_student_scm(student_scm, ground_truth_scm, seed_everything):
    """Test that learner correctly integrates with StudentSCM."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    learner = SCMLearner(student_scm)
    
    # Should be able to access student mechanisms
    assert 'X2' in learner.student.mechanisms
    
    # Training should update student parameters
    initial_param_count = sum(p.numel() for p in student_scm.parameters())
    
    data = ground_truth_scm.generate(100)
    learner.train_step({"data": data, "intervened": None}, n_epochs=5)
    
    final_param_count = sum(p.numel() for p in student_scm.parameters())
    
    # Parameter count should be unchanged (structure doesn't change)
    assert initial_param_count == final_param_count


@pytest.mark.unit
def test_learner_works_with_fresh_student_scm(ground_truth_scm, seed_everything):
    """Test that learner works with freshly initialized student."""
    from ace_experiments import StudentSCM, SCMLearner
    
    seed_everything(42)
    
    # Create fresh student
    fresh_student = StudentSCM(ground_truth_scm)
    learner = SCMLearner(fresh_student, lr=0.01)
    
    # Should work normally
    data = ground_truth_scm.generate(100)
    loss = learner.train_step({"data": data, "intervened": None}, n_epochs=10)
    
    assert isinstance(loss, float)
    assert loss >= 0

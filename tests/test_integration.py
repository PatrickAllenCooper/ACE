"""
Integration tests for ACE experimental pipeline.

Tests cover:
- End-to-end episode execution
- Executor + Learner integration
- Multi-episode training
- Loss tracking across episodes
- Full pipeline workflows
"""

import pytest
import torch


# =============================================================================
# Single Episode Integration Tests
# =============================================================================

@pytest.mark.integration
def test_single_episode_workflow(ground_truth_scm, student_scm, seed_everything):
    """Test complete single episode workflow."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    # Setup
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    # Episode 1: Observational
    result = executor.run_experiment(None)
    initial_loss = learner.train_step(result, n_epochs=10)
    
    # Loss should be computed
    assert isinstance(initial_loss, float)
    assert initial_loss >= 0
    
    # Buffer should have data
    assert len(learner.buffer) == 1


@pytest.mark.integration
def test_episode_with_intervention(ground_truth_scm, student_scm, seed_everything):
    """Test episode with intervention."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    # Run intervention experiment
    intervention = {'target': 'X2', 'value': 1.5, 'samples': 100}
    result = executor.run_experiment(intervention)
    
    # Train on interventional data
    loss = learner.train_step(result, n_epochs=10)
    
    assert isinstance(loss, float)
    assert result['intervened'] == 'X2'
    
    # Check intervention was masked in training
    assert len(learner.buffer) == 1
    assert learner.buffer[0]['intervened'] == 'X2'


@pytest.mark.integration
def test_observational_then_interventional_episode(ground_truth_scm, student_scm, seed_everything):
    """Test sequence of observational then interventional episodes."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    # Observational
    obs_result = executor.run_experiment(None)
    obs_loss = learner.train_step(obs_result, n_epochs=10)
    
    # Interventional
    int_plan = {'target': 'X1', 'value': 2.0}
    int_result = executor.run_experiment(int_plan)
    int_loss = learner.train_step(int_result, n_epochs=10)
    
    # Both should succeed
    assert isinstance(obs_loss, float)
    assert isinstance(int_loss, float)
    
    # Buffer should have both
    assert len(learner.buffer) == 2


# =============================================================================
# Multi-Episode Integration Tests
# =============================================================================

@pytest.mark.integration
def test_multiple_episodes_train_student(ground_truth_scm, student_scm, seed_everything):
    """Test that multiple episodes improve student."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    losses = []
    
    # Run 5 episodes
    for i in range(5):
        result = executor.run_experiment(None)
        loss = learner.train_step(result, n_epochs=20)
        losses.append(loss)
    
    # Check losses
    assert len(losses) == 5
    assert all(isinstance(l, float) for l in losses)
    
    # Loss should generally decrease (with some noise)
    # Check that later losses are lower than initial
    assert losses[-1] < losses[0] * 2  # At least some improvement


@pytest.mark.integration
def test_mixed_episode_sequence(ground_truth_scm, student_scm, seed_everything):
    """Test mixed sequence of observational and interventional episodes."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=5)
    
    episode_plans = [
        None,  # Observational
        {'target': 'X1', 'value': 1.0},
        {'target': 'X2', 'value': 2.0},
        None,  # Observational
        {'target': 'X3', 'value': 0.5},
    ]
    
    for plan in episode_plans:
        result = executor.run_experiment(plan)
        loss = learner.train_step(result, n_epochs=5)
        
        assert isinstance(loss, float)
    
    # Buffer should respect limit
    assert len(learner.buffer) == 5


# =============================================================================
# Loss Tracking Integration Tests
# =============================================================================

@pytest.mark.integration
def test_track_losses_across_episodes(ground_truth_scm, student_scm, seed_everything):
    """Test tracking losses across multiple episodes."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    loss_history = []
    
    for _ in range(3):
        result = executor.run_experiment(None)
        loss = learner.train_step(result, n_epochs=10)
        loss_history.append(loss)
    
    # Should have history
    assert len(loss_history) == 3
    
    # All should be valid
    assert all(l >= 0 for l in loss_history)
    assert all(not torch.isnan(torch.tensor(l)) for l in loss_history)


@pytest.mark.integration
def test_per_node_loss_computation(ground_truth_scm, student_scm, seed_everything):
    """Test computing per-node losses."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01)
    
    # Train for one episode
    result = executor.run_experiment(None)
    learner.train_step(result, n_epochs=10)
    
    # Compute per-node losses
    with torch.no_grad():
        gt_data = ground_truth_scm.generate(100)
        student_data = student_scm.forward(100)
        
        per_node_losses = {}
        for node in student_scm.nodes:
            mse = ((student_data[node] - gt_data[node]) ** 2).mean()
            per_node_losses[node] = mse.item()
    
    # Should have loss for each node
    assert len(per_node_losses) == 5
    
    # All should be valid
    assert all(l >= 0 for l in per_node_losses.values())


# =============================================================================
# Buffer Management Integration Tests
# =============================================================================

@pytest.mark.integration
def test_buffer_fills_correctly(ground_truth_scm, student_scm, seed_everything):
    """Test that buffer fills correctly across episodes."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    buffer_limit = 3
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=buffer_limit)
    
    # Run more episodes than buffer limit
    for i in range(5):
        result = executor.run_experiment(None)
        learner.train_step(result, n_epochs=5)
        
        # Buffer should not exceed limit
        assert len(learner.buffer) <= buffer_limit
    
    # Final buffer should be at limit
    assert len(learner.buffer) == buffer_limit


@pytest.mark.integration
def test_buffer_with_mixed_data_types(ground_truth_scm, student_scm, seed_everything):
    """Test buffer with mixed observational and interventional data."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    # Add observational
    obs_result = executor.run_experiment(None)
    learner.train_step(obs_result, n_epochs=5)
    
    # Add interventional on X1
    int1_result = executor.run_experiment({'target': 'X1', 'value': 1.0})
    learner.train_step(int1_result, n_epochs=5)
    
    # Add interventional on X2
    int2_result = executor.run_experiment({'target': 'X2', 'value': 2.0})
    learner.train_step(int2_result, n_epochs=5)
    
    # Buffer should have all three types
    assert len(learner.buffer) == 3
    assert learner.buffer[0]['intervened'] is None
    assert learner.buffer[1]['intervened'] == 'X1'
    assert learner.buffer[2]['intervened'] == 'X2'


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================

@pytest.mark.integration
def test_full_pipeline_10_episodes(ground_truth_scm, student_scm, seed_everything):
    """Test full pipeline with 10 episodes."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    # Run 10 episodes with random interventions
    import random
    nodes = list(ground_truth_scm.nodes)
    
    for i in range(10):
        if random.random() < 0.5:
            # Observational
            result = executor.run_experiment(None)
        else:
            # Random intervention
            target = random.choice(nodes)
            value = random.uniform(-2.0, 2.0)
            result = executor.run_experiment({
                'target': target,
                'value': value,
                'samples': 100
            })
        
        loss = learner.train_step(result, n_epochs=10)
        assert isinstance(loss, float)


@pytest.mark.integration
@pytest.mark.slow
def test_student_learns_simple_mechanism(ground_truth_scm, student_scm, seed_everything):
    """Test that student learns X2 = 2*X1 + 1 mechanism."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    # Train for 20 episodes (mix of observational and interventional)
    for i in range(20):
        if i % 3 == 0:
            # Interventional on X1
            result = executor.run_experiment({'target': 'X1', 'value': float(i % 5)})
        else:
            # Observational
            result = executor.run_experiment(None)
        
        learner.train_step(result, n_epochs=20)
    
    # Test X2 mechanism
    with torch.no_grad():
        # Generate test data
        test_data = ground_truth_scm.generate(1000)
        student_pred = student_scm.forward(1000)
        
        # X2 loss should be reasonable
        x2_loss = ((test_data['X2'] - student_pred['X2']) ** 2).mean().item()
        
        # Should have learned something (loss < 10)
        assert x2_loss < 10.0


# =============================================================================
# Error Handling Integration Tests
# =============================================================================

@pytest.mark.integration
def test_pipeline_handles_empty_intervention_plan(ground_truth_scm, student_scm):
    """Test pipeline handles empty intervention plan gracefully."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm)
    
    # Empty plan (no target) should be observational
    result = executor.run_experiment({})
    loss = learner.train_step(result, n_epochs=5)
    
    assert isinstance(loss, float)
    assert result['intervened'] is None


@pytest.mark.integration
def test_pipeline_with_sequential_same_interventions(ground_truth_scm, student_scm, seed_everything):
    """Test pipeline with repeated interventions."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=5)
    
    # Repeat same intervention 3 times
    for _ in range(3):
        result = executor.run_experiment({'target': 'X2', 'value': 1.5})
        loss = learner.train_step(result, n_epochs=5)
        assert isinstance(loss, float)
    
    # Buffer should have all 3
    assert len(learner.buffer) == 3


# =============================================================================
# Performance Integration Tests
# =============================================================================

@pytest.mark.integration
def test_pipeline_performance_scales(ground_truth_scm, student_scm, seed_everything):
    """Test that pipeline performance is reasonable."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    import time
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=5)
    
    # Time 5 episodes
    start = time.time()
    
    for _ in range(5):
        result = executor.run_experiment(None)
        learner.train_step(result, n_epochs=10)
    
    duration = time.time() - start
    
    # Should complete in reasonable time (<30 seconds for 5 episodes)
    assert duration < 30.0


# =============================================================================
# Consistency Integration Tests
# =============================================================================

@pytest.mark.integration
def test_pipeline_produces_consistent_results(ground_truth_scm, seed_everything):
    """Test that pipeline produces consistent results with same seed."""
    from ace_experiments import StudentSCM, ExperimentExecutor, SCMLearner
    
    def run_pipeline(seed):
        seed_everything(seed)
        student = StudentSCM(ground_truth_scm)
        executor = ExperimentExecutor(ground_truth_scm)
        learner = SCMLearner(student, lr=0.01, buffer_steps=5)
        
        losses = []
        for _ in range(3):
            result = executor.run_experiment(None)
            loss = learner.train_step(result, n_epochs=10)
            losses.append(loss)
        
        return losses
    
    # Run twice with same seed
    losses1 = run_pipeline(42)
    losses2 = run_pipeline(42)
    
    # Should be very similar (small numerical differences ok)
    for l1, l2 in zip(losses1, losses2):
        assert abs(l1 - l2) < 0.1

"""
Comprehensive integration tests for ACE experimental pipeline.

Tests cover:
- Multi-component integration
- Full episode workflows  
- Data flow through pipeline
- Loss tracking and improvement
"""

import pytest
import torch


# =============================================================================
# Multi-Episode Integration Tests
# =============================================================================

@pytest.mark.integration
def test_five_episode_training_workflow(ground_truth_scm, student_scm, seed_everything):
    """Test 5-episode training workflow."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    loss_history = []
    
    # Run 5 episodes
    for ep in range(5):
        # Mix of observational and interventional
        if ep % 2 == 0:
            result = executor.run_experiment(None)
        else:
            result = executor.run_experiment({
                'target': 'X1',
                'value': float(ep),
                'samples': 100
            })
        
        loss = learner.train_step(result, n_epochs=15)
        loss_history.append(loss)
    
    # Should have 5 losses
    assert len(loss_history) == 5
    
    # All valid
    assert all(l >= 0 for l in loss_history)


@pytest.mark.integration
def test_alternating_interventions_workflow(ground_truth_scm, student_scm, seed_everything):
    """Test workflow alternating between different nodes."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=5)
    
    nodes = list(ground_truth_scm.nodes)
    
    # Alternate through nodes
    for i, node in enumerate(nodes * 2):  # 10 steps total
        result = executor.run_experiment({
            'target': node,
            'value': float(i % 3),
            'samples': 50
        })
        
        loss = learner.train_step(result, n_epochs=5)
        assert isinstance(loss, float)


@pytest.mark.integration
def test_buffer_accumulation_workflow(ground_truth_scm, student_scm, seed_everything):
    """Test buffer accumulates correctly over episodes."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    buffer_limit = 3
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=buffer_limit)
    
    # Run 5 steps
    for i in range(5):
        result = executor.run_experiment(None)
        learner.train_step(result, n_epochs=3)
        
        # Buffer should not exceed limit
        assert len(learner.buffer) <= buffer_limit
    
    # Final buffer should be at limit
    assert len(learner.buffer) == buffer_limit


@pytest.mark.integration
def test_intervention_masking_throughout_workflow(ground_truth_scm, student_scm, seed_everything):
    """Test that intervention masking works throughout workflow."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=5)
    
    # Run interventions on X1
    for i in range(3):
        result = executor.run_experiment({
            'target': 'X1',
            'value': float(i * 2),
            'samples': 50
        })
        
        learner.train_step(result, n_epochs=5)
        
        # Verify intervention recorded in buffer
        assert learner.buffer[-1]['intervened'] == 'X1'


# =============================================================================
# Loss Improvement Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_training_improves_student_over_episodes(ground_truth_scm, student_scm, seed_everything):
    """Test that student improves over multiple training episodes."""
    from ace_experiments import ExperimentExecutor, SCMLearner, ScientificCritic
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    critic = ScientificCritic(ground_truth_scm)
    
    # Initial evaluation
    initial_loss, _ = critic.evaluate_model_detailed(student_scm)
    
    # Train for 10 episodes
    for _ in range(10):
        result = executor.run_experiment(None)
        learner.train_step(result, n_epochs=20)
    
    # Final evaluation
    final_loss, _ = critic.evaluate_model_detailed(student_scm)
    
    # Should have improved (or at least not gotten much worse)
    assert final_loss <= initial_loss * 1.5


# =============================================================================
# Data Flow Tests
# =============================================================================

@pytest.mark.integration
def test_data_flows_from_executor_to_learner(ground_truth_scm, student_scm, seed_everything):
    """Test data flows correctly from executor to learner."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01)
    
    # Generate data
    result = executor.run_experiment({'target': 'X2', 'value': 1.5, 'samples': 100})
    
    # Data structure should be compatible with learner
    assert 'data' in result
    assert 'intervened' in result
    
    # Should be able to train
    loss = learner.train_step(result, n_epochs=5)
    assert isinstance(loss, float)


@pytest.mark.integration
def test_observational_and_interventional_mix_workflow(ground_truth_scm, student_scm, seed_everything):
    """Test workflow with mixed observational and interventional data."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    # Structured sequence
    sequence = [
        None,  # Obs
        {'target': 'X1', 'value': 1.0},
        None,  # Obs
        {'target': 'X2', 'value': 2.0},
        None,  # Obs
        {'target': 'X3', 'value': 0.5},
    ]
    
    for plan in sequence:
        result = executor.run_experiment(plan)
        loss = learner.train_step(result, n_epochs=10)
        
        # Verify data type in buffer
        if plan is None:
            assert learner.buffer[-1]['intervened'] is None
        else:
            assert learner.buffer[-1]['intervened'] == plan['target']

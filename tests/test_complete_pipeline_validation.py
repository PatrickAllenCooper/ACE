"""
Complete pipeline validation tests.

Tests the entire ACE experimental workflow end-to-end.
"""

import pytest
import torch


# =============================================================================
# Complete Pipeline Tests
# =============================================================================

@pytest.mark.integration
def test_complete_ace_pipeline_observational_only(ground_truth_scm, student_scm, seed_everything):
    """Test complete pipeline with only observational data."""
    from ace_experiments import ExperimentExecutor, SCMLearner, ScientificCritic
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=5)
    critic = ScientificCritic(ground_truth_scm)
    
    # Run 5 observational episodes
    for _ in range(5):
        result = executor.run_experiment(None)
        learner.train_step(result, n_epochs=10)
    
    # Evaluate
    final_loss, node_losses = critic.evaluate_model_detailed(student_scm)
    
    # Should have trained
    assert final_loss >= 0
    assert len(node_losses) == 5


@pytest.mark.integration
def test_complete_ace_pipeline_with_interventions(ground_truth_scm, student_scm, seed_everything):
    """Test complete pipeline with systematic interventions."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    nodes = list(ground_truth_scm.nodes)
    
    # Systematic interventions on each node
    for node in nodes:
        result = executor.run_experiment({
            'target': node,
            'value': 1.0,
            'samples': 100
        })
        loss = learner.train_step(result, n_epochs=15)
        assert isinstance(loss, float)


@pytest.mark.integration
def test_pipeline_with_state_encoder(ground_truth_scm, student_scm, seed_everything):
    """Test pipeline integrating StateEncoder."""
    from ace_experiments import ExperimentExecutor, SCMLearner, StateEncoder, ScientificCritic
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01)
    critic = ScientificCritic(ground_truth_scm)
    encoder = StateEncoder(len(ground_truth_scm.nodes), torch.device('cpu'))
    
    # Train a bit
    for _ in range(3):
        result = executor.run_experiment(None)
        learner.train_step(result, n_epochs=10)
    
    # Evaluate and encode state
    _, node_losses = critic.evaluate_model_detailed(student_scm)
    state = encoder(student_scm, node_losses)
    
    # State should be valid
    assert isinstance(state, torch.Tensor)
    assert not torch.isnan(state).any()


@pytest.mark.integration
def test_pipeline_with_early_stopping(ground_truth_scm, student_scm, seed_everything):
    """Test pipeline with EarlyStopping integration."""
    from ace_experiments import ExperimentExecutor, SCMLearner, ScientificCritic, EarlyStopping
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01)
    critic = ScientificCritic(ground_truth_scm)
    stopper = EarlyStopping(patience=3, min_episodes=0)
    
    # Train until early stop
    for ep in range(20):
        result = executor.run_experiment(None)
        learner.train_step(result, n_epochs=5)
        
        total_loss, _ = critic.evaluate_model_detailed(student_scm)
        
        if stopper.check_loss(total_loss):
            break
    
    # Should have run at least a few episodes
    assert ep > 0


@pytest.mark.integration
def test_pipeline_with_dedicated_root_learner(ground_truth_scm, student_scm, seed_everything):
    """Test pipeline with DedicatedRootLearner."""
    from ace_experiments import ExperimentExecutor, SCMLearner, DedicatedRootLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01)
    root_learner = DedicatedRootLearner(['X1', 'X4'])
    
    # Train main learner
    for _ in range(3):
        result = executor.run_experiment(None)
        learner.train_step(result, n_epochs=10)
    
    # Train roots separately
    obs_data = ground_truth_scm.generate(1000)
    root_learner.fit(obs_data, epochs=50)
    
    # Apply to student
    root_learner.apply_to_student(student_scm)
    
    # Should complete without error


@pytest.mark.integration
def test_full_workflow_with_all_components(ground_truth_scm, seed_everything):
    """Test complete workflow with all major components."""
    from ace_experiments import (
        StudentSCM, ExperimentExecutor, SCMLearner,
        ScientificCritic, EarlyStopping, StateEncoder,
        DedicatedRootLearner
    )
    
    seed_everything(42)
    
    # Initialize all components
    student = StudentSCM(ground_truth_scm)
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student, lr=0.01, buffer_steps=5)
    critic = ScientificCritic(ground_truth_scm)
    stopper = EarlyStopping(patience=5, min_episodes=2)
    encoder = StateEncoder(5, torch.device('cpu'))
    root_learner = DedicatedRootLearner(['X1', 'X4'])
    
    # Run 5 episodes
    for ep in range(5):
        # Experiment
        result = executor.run_experiment(None)
        
        # Train
        learner.train_step(result, n_epochs=10)
        
        # Evaluate
        total_loss, node_losses = critic.evaluate_model_detailed(student)
        
        # Encode state
        state = encoder(student, node_losses)
        
        # Check early stopping
        if ep >= 2 and stopper.check_loss(total_loss):
            break
    
    # Should complete without errors
    assert True

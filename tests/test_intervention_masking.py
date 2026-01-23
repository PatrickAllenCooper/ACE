"""
Critical tests for intervention masking correctness.

These tests ensure that when we do(X₂ = v), the X₂ mechanism
is NOT trained on that data (preserves causal semantics).
"""

import pytest
import torch


# =============================================================================
# Intervention Masking Correctness Tests
# =============================================================================

@pytest.mark.unit
def test_intervention_mask_prevents_training(ground_truth_scm, student_scm, seed_everything):
    """CRITICAL: Test that intervened nodes are not trained."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=5)
    
    # Get initial X2 mechanism parameters
    initial_x2_params = [p.clone().detach() for p in student_scm.mechanisms['X2'].parameters()]
    
    # Train ONLY with do(X2) interventions (X2 should NOT be trained)
    for _ in range(10):
        data = ground_truth_scm.generate(n_samples=100, interventions={'X2': 2.0})
        learner.train_step({"data": data, "intervened": "X2"}, n_epochs=20)
    
    # X2 mechanism should be UNCHANGED (was masked)
    final_x2_params = [p.detach() for p in student_scm.mechanisms['X2'].parameters()]
    
    # Check if parameters changed significantly
    max_change = max((p1 - p2).abs().max().item() 
                     for p1, p2 in zip(initial_x2_params, final_x2_params))
    
    # Should have minimal change (only from other mechanisms in buffer)
    # If masking works, change should be < 0.1
    # If masking fails, change would be >> 1.0
    assert max_change < 0.5, f"X2 mechanism changed too much ({max_change:.4f}) - masking may have failed"


@pytest.mark.unit
def test_non_intervened_nodes_are_trained(ground_truth_scm, student_scm, seed_everything):
    """Test that non-intervened nodes ARE trained during interventions."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=5)
    
    # Get initial X3 mechanism parameters (not intervened)
    initial_x3_params = [p.clone().detach() for p in student_scm.mechanisms['X3'].parameters()]
    
    # Train with do(X1) interventions (X3 should be trained)
    for _ in range(10):
        data = ground_truth_scm.generate(n_samples=100, interventions={'X1': 2.0})
        learner.train_step({"data": data, "intervened": "X1"}, n_epochs=20)
    
    # X3 mechanism SHOULD change (was not masked)
    final_x3_params = [p.detach() for p in student_scm.mechanisms['X3'].parameters()]
    
    max_change = max((p1 - p2).abs().max().item() 
                     for p1, p2 in zip(initial_x3_params, final_x3_params))
    
    # Should have significant change (being trained)
    assert max_change > 0.01, f"X3 mechanism didn't change enough ({max_change:.4f}) - should be trained"


@pytest.mark.unit
def test_observational_trains_all_mechanisms(ground_truth_scm, student_scm, seed_everything):
    """Test that observational data trains all mechanisms."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=5)
    
    # Get initial parameters for all mechanisms
    initial_params = {
        node: [p.clone().detach() for p in student_scm.mechanisms[node].parameters()]
        for node in ['X2', 'X3', 'X5']  # Mechanisms with parents
    }
    
    # Train with ONLY observational data (no interventions)
    for _ in range(10):
        data = ground_truth_scm.generate(n_samples=100, interventions=None)
        learner.train_step({"data": data, "intervened": None}, n_epochs=20)
    
    # ALL mechanisms should change
    for node in ['X2', 'X3', 'X5']:
        final_params = [p.detach() for p in student_scm.mechanisms[node].parameters()]
        
        max_change = max((p1 - p2).abs().max().item() 
                         for p1, p2 in zip(initial_params[node], final_params))
        
        assert max_change > 0.01, f"{node} mechanism didn't change with observational data"


@pytest.mark.unit
def test_mask_preserved_in_buffer(ground_truth_scm, student_scm, seed_everything):
    """Test that intervention information is preserved in buffer."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    # Add interventional data
    int_data = ground_truth_scm.generate(n_samples=50, interventions={'X1': 1.0})
    learner.train_step({"data": int_data, "intervened": "X1"}, n_epochs=5)
    
    # Add observational data
    obs_data = ground_truth_scm.generate(n_samples=50, interventions=None)
    learner.train_step({"data": obs_data, "intervened": None}, n_epochs=5)
    
    # Check buffer preserved intervention information
    assert len(learner.buffer) == 2
    assert learner.buffer[0]["intervened"] == "X1"
    assert learner.buffer[1]["intervened"] is None


@pytest.mark.unit
def test_mask_application_in_replay(ground_truth_scm, student_scm, seed_everything):
    """Test that masks are correctly applied during replay phase."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=5)
    
    # Fill buffer with mixed data
    for i in range(5):
        if i % 2 == 0:
            data = ground_truth_scm.generate(n_samples=50, interventions={'X2': float(i)})
            learner.train_step({"data": data, "intervened": "X2"}, n_epochs=5)
        else:
            data = ground_truth_scm.generate(n_samples=50, interventions=None)
            learner.train_step({"data": data, "intervened": None}, n_epochs=5)
    
    # Buffer should have mixed intervention markers
    intervened_count = sum(1 for b in learner.buffer if b["intervened"] is not None)
    assert intervened_count == 3  # Episodes 0, 2, 4
    
    # All training should complete without errors (masks working)
    assert len(learner.buffer) == 5


@pytest.mark.integration
def test_intervention_masking_full_workflow(ground_truth_scm, seed_everything):
    """Integration test: Full workflow with intervention masking."""
    from ace_experiments import StudentSCM, ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    student = StudentSCM(ground_truth_scm)
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student, lr=0.01, buffer_steps=10)
    
    # Run 10 steps with various interventions
    for i in range(10):
        if i % 3 == 0:
            result = executor.run_experiment(None)
        else:
            target = ['X1', 'X2', 'X3'][i % 3]
            result = executor.run_experiment({'target': target, 'value': float(i)})
        
        # This should apply masking correctly
        loss = learner.train_step(result, n_epochs=10)
        
        # Should complete without assertion errors
        assert isinstance(loss, float)

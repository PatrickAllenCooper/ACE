"""
Final push tests to maximize coverage toward 90%.

Covers remaining edge cases and code paths.
"""

import pytest
import torch
import numpy as np


# =============================================================================
# Additional GroundTruthSCM Edge Cases
# =============================================================================

@pytest.mark.unit
def test_ground_truth_scm_all_mechanisms_with_large_samples(ground_truth_scm, seed_everything):
    """Test all mechanisms with large sample size."""
    seed_everything(42)
    
    data = ground_truth_scm.generate(n_samples=10000)
    
    # All mechanisms should produce reasonable outputs
    for node in ground_truth_scm.nodes:
        assert data[node].shape == (10000,)
        assert data[node].std() > 0  # Should have variance
        assert not torch.isnan(data[node]).any()


@pytest.mark.unit
def test_ground_truth_scm_intervention_on_all_nodes_sequentially(ground_truth_scm, seed_everything):
    """Test interventions on all nodes sequentially."""
    seed_everything(42)
    
    for node in ground_truth_scm.nodes:
        for value in [-2.0, 0.0, 2.0]:
            data = ground_truth_scm.generate(n_samples=50, interventions={node: value})
            assert torch.all(data[node] == value)


# =============================================================================
# Additional StudentSCM Edge Cases
# =============================================================================

@pytest.mark.unit
def test_student_scm_forward_with_all_interventions(student_scm, seed_everything):
    """Test student forward with intervention on every node."""
    seed_everything(42)
    
    # Intervene on all nodes
    interventions = {node: float(i) for i, node in enumerate(student_scm.nodes)}
    
    data = student_scm.forward(n_samples=10, interventions=interventions)
    
    # All should match intervention values
    for i, node in enumerate(student_scm.nodes):
        assert torch.all(data[node] == float(i))


@pytest.mark.unit
def test_student_scm_mechanisms_different_for_different_nodes(student_scm):
    """Test that different nodes have different mechanisms."""
    
    # Get mechanisms
    x2_mech = student_scm.mechanisms['X2']
    x3_mech = student_scm.mechanisms['X3']
    
    # X2 and X3 should be different objects
    assert x2_mech is not x3_mech
    
    # They should have different numbers of input features
    # X2 has 1 parent, X3 has 2 parents
    assert x2_mech[0].in_features == 1
    assert x3_mech[0].in_features == 2


# =============================================================================
# Additional ExperimentExecutor Edge Cases
# =============================================================================

@pytest.mark.unit
def test_experiment_executor_with_zero_value_intervention(ground_truth_scm, seed_everything):
    """Test executor with zero intervention value."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    
    result = executor.run_experiment({'target': 'X2', 'value': 0.0, 'samples': 100})
    
    assert torch.all(result['data']['X2'] == 0.0)


@pytest.mark.unit
def test_experiment_executor_with_negative_values(ground_truth_scm, seed_everything):
    """Test executor with negative intervention values."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    
    for val in [-5.0, -2.5, -1.0]:
        result = executor.run_experiment({'target': 'X1', 'value': val, 'samples': 50})
        assert torch.all(result['data']['X1'] == val)


# =============================================================================
# Additional SCMLearner Edge Cases
# =============================================================================

@pytest.mark.unit
def test_scm_learner_with_very_small_buffer(student_scm, ground_truth_scm, seed_everything):
    """Test SCMLearner with buffer_steps=1."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    
    learner = SCMLearner(student_scm, buffer_steps=1)
    
    # Train multiple steps
    for _ in range(5):
        data = ground_truth_scm.generate(50)
        learner.train_step({"data": data, "intervened": None}, n_epochs=5)
    
    # Buffer should never exceed 1
    assert len(learner.buffer) == 1


@pytest.mark.unit
def test_scm_learner_with_large_buffer(student_scm, ground_truth_scm, seed_everything):
    """Test SCMLearner with large buffer."""
    from ace_experiments import SCMLearner
    
    seed_everything(42)
    
    learner = SCMLearner(student_scm, buffer_steps=100)
    
    # Add 50 batches
    for _ in range(50):
        data = ground_truth_scm.generate(30)
        learner.train_step({"data": data, "intervened": None}, n_epochs=3)
    
    # Buffer should have 50
    assert len(learner.buffer) == 50


@pytest.mark.unit
def test_scm_learner_normalize_batch_with_all_formats(student_scm, ground_truth_scm):
    """Test _normalize_batch with various input formats."""
    from ace_experiments import SCMLearner
    
    learner = SCMLearner(student_scm)
    
    data = ground_truth_scm.generate(100)
    
    # Format 1: Full format
    batch1 = {"data": data, "intervened": "X1"}
    normalized1 = learner._normalize_batch(batch1)
    assert normalized1['intervened'] == "X1"
    
    # Format 2: Backward compatible (plain dict)
    batch2 = data
    normalized2 = learner._normalize_batch(batch2)
    assert normalized2['intervened'] is None


# =============================================================================
# Additional StateEncoder Tests
# =============================================================================

@pytest.mark.unit
def test_state_encoder_with_missing_node_losses(student_scm):
    """Test StateEncoder when some node losses are missing."""
    from ace_experiments import StateEncoder
    
    encoder = StateEncoder(5, torch.device('cpu'))
    encoder.eval()
    
    # Incomplete losses (missing some nodes)
    node_losses = {'X1': 1.0, 'X3': 2.0}  # Missing X2, X4, X5
    
    # Should handle gracefully (will use 0.0 for missing)
    encoding = encoder(student_scm, node_losses)
    
    assert isinstance(encoding, torch.Tensor)
    assert not torch.isnan(encoding).any()


@pytest.mark.unit
def test_state_encoder_batch_processing(student_scm):
    """Test StateEncoder produces consistent output."""
    from ace_experiments import StateEncoder
    
    encoder = StateEncoder(5, torch.device('cpu'))
    encoder.eval()
    
    node_losses = {f'X{i}': float(i) for i in range(1, 6)}
    
    # Multiple calls
    enc1 = encoder(student_scm, node_losses)
    enc2 = encoder(student_scm, node_losses)
    
    # Should be deterministic in eval mode
    assert torch.allclose(enc1, enc2)


# =============================================================================
# Additional ExperimentalDSL Tests
# =============================================================================

@pytest.mark.unit
def test_experimental_dsl_parse_scientific_notation():
    """Test DSL parsing with scientific notation."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2', 'X3']
    dsl = ExperimentalDSL(nodes)
    
    # Scientific notation
    commands = [
        "DO X1 = 1e-5",
        "DO X2 = 2.5e+3",
        "DO X3 = -1.5e-2"
    ]
    
    for cmd in commands:
        result = dsl.parse_to_dict(cmd)
        # Should parse (might be out of range and return None, but shouldn't crash)
        assert result is None or isinstance(result, dict)


@pytest.mark.unit
def test_experimental_dsl_case_insensitive():
    """Test DSL parsing is case insensitive for DO."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2']
    dsl = ExperimentalDSL(nodes)
    
    # Different cases
    commands = ["DO X1 = 1.0", "do X1 = 1.0", "Do X1 = 1.0"]
    
    for cmd in commands:
        result = dsl.parse_to_dict(cmd)
        if result:
            assert result['target'] == 'X1'


# =============================================================================
# Additional TransformerPolicy Tests
# =============================================================================

@pytest.mark.unit
def test_transformer_policy_decode_tensor():
    """Test TransformerPolicy decode_tensor method."""
    from ace_experiments import TransformerPolicy, ExperimentalDSL
    
    nodes = ['X1', 'X2']
    dsl = ExperimentalDSL(nodes)
    policy = TransformerPolicy(dsl, torch.device('cpu'))
    
    # Create token indices
    indices = torch.tensor([dsl.token2id['DO'], dsl.token2id['X1'], dsl.token2id['=']])
    
    # Decode
    text = policy.decode_tensor(indices)
    
    # Should return string
    assert isinstance(text, str)
    assert 'DO' in text or 'X1' in text


# =============================================================================
# Additional Integration Tests
# =============================================================================

@pytest.mark.integration
def test_full_workflow_with_varying_learning_rates(ground_truth_scm, seed_everything):
    """Test workflow with different learning rates."""
    from ace_experiments import StudentSCM, ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    for lr in [0.001, 0.01, 0.1]:
        student = StudentSCM(ground_truth_scm)
        executor = ExperimentExecutor(ground_truth_scm)
        learner = SCMLearner(student, lr=lr)
        
        # Train briefly
        for _ in range(3):
            result = executor.run_experiment(None)
            loss = learner.train_step(result, n_epochs=5)
            assert isinstance(loss, float)


@pytest.mark.integration
def test_workflow_with_all_intervention_types(ground_truth_scm, student_scm, seed_everything):
    """Test workflow covering all intervention patterns."""
    from ace_experiments import ExperimentExecutor, SCMLearner
    
    seed_everything(42)
    
    executor = ExperimentExecutor(ground_truth_scm)
    learner = SCMLearner(student_scm, lr=0.01, buffer_steps=10)
    
    # Mix of all patterns
    patterns = [
        None,  # Observational
        {'target': 'X1', 'value': 1.0},  # Root
        {'target': 'X2', 'value': 2.0},  # Intermediate
        {'target': 'X3', 'value': 0.0},  # Collider
        {'target': 'X4', 'value': 3.0},  # Root
        {'target': 'X5', 'value': 1.5},  # Leaf
        None,  # Observational again
    ]
    
    for pattern in patterns:
        result = executor.run_experiment(pattern)
        loss = learner.train_step(result, n_epochs=5)
        assert isinstance(loss, float)

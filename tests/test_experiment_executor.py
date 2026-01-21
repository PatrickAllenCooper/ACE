"""
Unit tests for ExperimentExecutor class.

Tests cover:
- Initialization
- Observational experiment execution
- Interventional experiment execution
- Intervention plan parsing
- Data structure validation
- Edge cases
"""

import pytest
import torch


# =============================================================================
# Initialization Tests
# =============================================================================

@pytest.mark.unit
def test_experiment_executor_initialization(ground_truth_scm):
    """Test that ExperimentExecutor initializes correctly."""
    from ace_experiments import ExperimentExecutor
    
    executor = ExperimentExecutor(ground_truth_scm)
    
    # Should have env attribute
    assert hasattr(executor, 'env')
    assert executor.env is ground_truth_scm


# =============================================================================
# Observational Experiment Tests
# =============================================================================

@pytest.mark.unit
def test_run_observational_experiment(ground_truth_scm, seed_everything):
    """Test running observational experiment (no intervention)."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    # Run with None intervention plan
    result = executor.run_experiment(None)
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'data' in result
    assert 'intervened' in result
    
    # Check data
    assert isinstance(result['data'], dict)
    assert set(result['data'].keys()) == {'X1', 'X2', 'X3', 'X4', 'X5'}
    
    # Check intervened is None (observational)
    assert result['intervened'] is None
    
    # Default should be 100 samples
    for node in result['data']:
        assert result['data'][node].shape == (100,)


@pytest.mark.unit
def test_observational_experiment_generates_valid_data(ground_truth_scm, seed_everything):
    """Test that observational experiment generates valid data."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    result = executor.run_experiment(None)
    data = result['data']
    
    # Check no NaN/Inf
    for node in data:
        assert not torch.isnan(data[node]).any()
        assert not torch.isinf(data[node]).any()
    
    # Check X1 distribution (should be ~N(0,1))
    assert data['X1'].mean() == pytest.approx(0.0, abs=0.3)
    assert data['X1'].std() == pytest.approx(1.0, abs=0.3)


# =============================================================================
# Interventional Experiment Tests
# =============================================================================

@pytest.mark.unit
def test_run_intervention_experiment_on_x1(ground_truth_scm, seed_everything):
    """Test running intervention on X1."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    # Intervention plan
    intervention_plan = {
        'target': 'X1',
        'value': 2.0,
        'samples': 50
    }
    
    result = executor.run_experiment(intervention_plan)
    
    # Check structure
    assert 'data' in result
    assert 'intervened' in result
    
    # Check intervened node is marked
    assert result['intervened'] == 'X1'
    
    # Check X1 is set to intervention value
    assert torch.all(result['data']['X1'] == 2.0)
    
    # Check sample size
    assert result['data']['X1'].shape == (50,)


@pytest.mark.unit
def test_run_intervention_experiment_on_x2(ground_truth_scm, seed_everything):
    """Test running intervention on X2."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    intervention_plan = {
        'target': 'X2',
        'value': 1.5
    }
    
    result = executor.run_experiment(intervention_plan)
    
    # Check intervened node
    assert result['intervened'] == 'X2'
    
    # Check X2 is set to intervention value
    assert torch.all(result['data']['X2'] == 1.5)
    
    # Default samples is 100
    assert result['data']['X2'].shape == (100,)


@pytest.mark.unit
def test_intervention_affects_downstream_nodes(ground_truth_scm, seed_everything):
    """Test that intervention on X1 affects downstream nodes."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    intervention_plan = {
        'target': 'X1',
        'value': 5.0,
        'samples': 100
    }
    
    result = executor.run_experiment(intervention_plan)
    data = result['data']
    
    # X1 is fixed
    assert torch.all(data['X1'] == 5.0)
    
    # X2 should be affected (X2 = 2*X1 + 1)
    expected_x2_mean = 2.0 * 5.0 + 1.0
    assert data['X2'].mean() == pytest.approx(expected_x2_mean, abs=0.3)
    
    # X4 should be unaffected (independent subgraph)
    assert data['X4'].mean() == pytest.approx(2.0, abs=0.5)


@pytest.mark.unit
def test_intervention_does_not_affect_upstream_nodes(ground_truth_scm, seed_everything):
    """Test that intervention does not affect upstream nodes."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    intervention_plan = {
        'target': 'X2',
        'value': 3.0
    }
    
    result = executor.run_experiment(intervention_plan)
    data = result['data']
    
    # X2 is fixed
    assert torch.all(data['X2'] == 3.0)
    
    # X1 is upstream, should be unaffected
    assert data['X1'].mean() == pytest.approx(0.0, abs=0.3)
    assert data['X1'].std() > 0.5  # Should vary


# =============================================================================
# Intervention Plan Parsing Tests
# =============================================================================

@pytest.mark.unit
def test_intervention_plan_with_default_samples(ground_truth_scm, seed_everything):
    """Test intervention plan uses default 100 samples."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    # Plan without 'samples' key
    intervention_plan = {
        'target': 'X3',
        'value': 0.0
    }
    
    result = executor.run_experiment(intervention_plan)
    
    # Should use default 100 samples
    assert result['data']['X3'].shape == (100,)


@pytest.mark.unit
def test_intervention_plan_with_custom_samples(ground_truth_scm, seed_everything):
    """Test intervention plan with custom sample size."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    intervention_plan = {
        'target': 'X4',
        'value': 1.0,
        'samples': 200
    }
    
    result = executor.run_experiment(intervention_plan)
    
    # Should use specified samples
    assert result['data']['X4'].shape == (200,)


@pytest.mark.unit
def test_intervention_plan_without_target_is_observational(ground_truth_scm, seed_everything):
    """Test that intervention plan without target becomes observational."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    # Plan with no target
    intervention_plan = {
        'value': 2.0,
        'samples': 50
    }
    
    result = executor.run_experiment(intervention_plan)
    
    # Should be observational
    assert result['intervened'] is None
    
    # Data should vary naturally
    assert result['data']['X1'].std() > 0.5


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.unit
def test_intervention_with_extreme_values(ground_truth_scm, seed_everything):
    """Test intervention with extreme values."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    extreme_values = [-1000.0, -10.0, 0.0, 10.0, 1000.0]
    
    for val in extreme_values:
        intervention_plan = {
            'target': 'X1',
            'value': val,
            'samples': 10
        }
        
        result = executor.run_experiment(intervention_plan)
        
        # Should complete without error
        assert result['intervened'] == 'X1'
        assert torch.all(result['data']['X1'] == val)
        
        # No NaN or Inf
        for node in result['data']:
            assert not torch.isnan(result['data'][node]).any()
            assert not torch.isinf(result['data'][node]).any()


@pytest.mark.unit
def test_intervention_on_each_node(ground_truth_scm, seed_everything):
    """Test that intervention works on all nodes."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    
    for node in nodes:
        intervention_plan = {
            'target': node,
            'value': 1.0,
            'samples': 20
        }
        
        result = executor.run_experiment(intervention_plan)
        
        # Check intervention applied
        assert result['intervened'] == node
        assert torch.all(result['data'][node] == 1.0)


@pytest.mark.unit
def test_intervention_with_single_sample(ground_truth_scm, seed_everything):
    """Test intervention with single sample."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    intervention_plan = {
        'target': 'X2',
        'value': 2.5,
        'samples': 1
    }
    
    result = executor.run_experiment(intervention_plan)
    
    # Should work with single sample
    assert result['data']['X2'].shape == (1,)
    assert result['data']['X2'][0] == 2.5


@pytest.mark.unit
def test_intervention_with_large_sample_size(ground_truth_scm, seed_everything):
    """Test intervention with large sample size."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    intervention_plan = {
        'target': 'X3',
        'value': 0.5,
        'samples': 5000
    }
    
    result = executor.run_experiment(intervention_plan)
    
    # Should handle large samples
    assert result['data']['X3'].shape == (5000,)
    assert torch.all(result['data']['X3'] == 0.5)


# =============================================================================
# Data Structure Validation Tests
# =============================================================================

@pytest.mark.unit
def test_result_structure_is_consistent(ground_truth_scm, seed_everything):
    """Test that result structure is consistent across calls."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    # Observational
    result1 = executor.run_experiment(None)
    assert set(result1.keys()) == {'data', 'intervened'}
    
    # Interventional
    result2 = executor.run_experiment({'target': 'X1', 'value': 1.0})
    assert set(result2.keys()) == {'data', 'intervened'}
    
    # Both have same data keys
    assert set(result1['data'].keys()) == set(result2['data'].keys())


@pytest.mark.unit
def test_data_is_dict_of_tensors(ground_truth_scm, seed_everything):
    """Test that data is always dict of tensors."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    result = executor.run_experiment(None)
    data = result['data']
    
    # Should be dict
    assert isinstance(data, dict)
    
    # All values should be tensors
    for node, values in data.items():
        assert isinstance(values, torch.Tensor)
        assert values.dim() == 1  # 1D tensor


@pytest.mark.unit
def test_intervened_field_is_string_or_none(ground_truth_scm, seed_everything):
    """Test that intervened field is string or None."""
    from ace_experiments import ExperimentExecutor
    
    seed_everything(42)
    executor = ExperimentExecutor(ground_truth_scm)
    
    # Observational - should be None
    result1 = executor.run_experiment(None)
    assert result1['intervened'] is None
    
    # Interventional - should be string
    result2 = executor.run_experiment({'target': 'X2', 'value': 1.0})
    assert isinstance(result2['intervened'], str)
    assert result2['intervened'] == 'X2'


# =============================================================================
# Reproducibility Tests
# =============================================================================

@pytest.mark.unit
def test_deterministic_with_seed_observational(ground_truth_scm, seed_everything):
    """Test observational experiments are reproducible with same seed."""
    from ace_experiments import ExperimentExecutor
    
    executor = ExperimentExecutor(ground_truth_scm)
    
    # First run
    seed_everything(12345)
    result1 = executor.run_experiment(None)
    
    # Second run with same seed
    seed_everything(12345)
    result2 = executor.run_experiment(None)
    
    # Should be identical
    for node in result1['data']:
        assert torch.allclose(result1['data'][node], result2['data'][node])


@pytest.mark.unit
def test_deterministic_with_seed_interventional(ground_truth_scm, seed_everything):
    """Test interventional experiments are reproducible with same seed."""
    from ace_experiments import ExperimentExecutor
    
    executor = ExperimentExecutor(ground_truth_scm)
    
    intervention_plan = {'target': 'X1', 'value': 3.0, 'samples': 100}
    
    # First run
    seed_everything(54321)
    result1 = executor.run_experiment(intervention_plan)
    
    # Second run with same seed
    seed_everything(54321)
    result2 = executor.run_experiment(intervention_plan)
    
    # Should be identical
    for node in result1['data']:
        assert torch.allclose(result1['data'][node], result2['data'][node])

"""
Unit tests for GroundTruthSCM class.

Tests cover:
- Graph structure and topology
- Individual mechanism equations
- Observational data generation
- Interventional data generation
- Statistical properties of distributions
- Edge cases and invariants
"""

import pytest
import torch
import numpy as np
from scipy import stats


# =============================================================================
# Graph Structure Tests
# =============================================================================

@pytest.mark.unit
def test_ground_truth_scm_initialization(ground_truth_scm):
    """Test that GroundTruthSCM initializes with correct structure."""
    scm = ground_truth_scm
    
    # Check nodes
    assert set(scm.nodes) == {'X1', 'X2', 'X3', 'X4', 'X5'}
    assert len(scm.nodes) == 5
    
    # Check graph is DAG
    import networkx as nx
    assert nx.is_directed_acyclic_graph(scm.graph)


@pytest.mark.unit
def test_graph_edges(ground_truth_scm):
    """Test that graph has correct edges."""
    scm = ground_truth_scm
    edges = list(scm.graph.edges())
    
    expected_edges = [
        ('X1', 'X2'),
        ('X2', 'X3'),
        ('X1', 'X3'),
        ('X4', 'X5')
    ]
    
    assert len(edges) == 4
    for edge in expected_edges:
        assert edge in edges


@pytest.mark.unit
def test_parent_relationships(ground_truth_scm):
    """Test parent-child relationships."""
    scm = ground_truth_scm
    
    # Root nodes (no parents)
    assert scm.get_parents('X1') == []
    assert scm.get_parents('X4') == []
    
    # Intermediate nodes
    assert scm.get_parents('X2') == ['X1']
    assert scm.get_parents('X5') == ['X4']
    
    # Collider (two parents)
    parents_x3 = scm.get_parents('X3')
    assert len(parents_x3) == 2
    assert 'X1' in parents_x3
    assert 'X2' in parents_x3


@pytest.mark.unit
def test_topological_ordering(ground_truth_scm):
    """Test that topological ordering is valid."""
    scm = ground_truth_scm
    topo_order = scm.topo_order
    
    # Check all nodes present
    assert len(topo_order) == 5
    assert set(topo_order) == {'X1', 'X2', 'X3', 'X4', 'X5'}
    
    # Check ordering constraints
    # X1 must come before X2 and X3
    assert topo_order.index('X1') < topo_order.index('X2')
    assert topo_order.index('X1') < topo_order.index('X3')
    
    # X2 must come before X3
    assert topo_order.index('X2') < topo_order.index('X3')
    
    # X4 must come before X5
    assert topo_order.index('X4') < topo_order.index('X5')


@pytest.mark.unit
def test_collider_identification(ground_truth_scm):
    """Test that X3 is correctly identified as a collider."""
    scm = ground_truth_scm
    
    # X3 has two parents (makes it a collider)
    parents_x3 = scm.get_parents('X3')
    assert len(parents_x3) == 2
    
    # All other nodes are not colliders
    assert len(scm.get_parents('X1')) == 0
    assert len(scm.get_parents('X2')) == 1
    assert len(scm.get_parents('X4')) == 0
    assert len(scm.get_parents('X5')) == 1


# =============================================================================
# Mechanism Tests - Individual Equations
# =============================================================================

@pytest.mark.unit
@pytest.mark.statistical
def test_x1_root_mechanism(ground_truth_scm, seed_everything):
    """Test X1 ~ N(0, 1) root mechanism."""
    seed_everything(42)
    scm = ground_truth_scm
    
    n_samples = 10000
    data = scm.generate(n_samples, interventions=None)
    x1_samples = data['X1'].numpy()
    
    # Test mean ≈ 0
    assert x1_samples.mean() == pytest.approx(0.0, abs=0.1)
    
    # Test std ≈ 1
    assert x1_samples.std() == pytest.approx(1.0, abs=0.1)
    
    # Kolmogorov-Smirnov test for normality
    _, p_value = stats.kstest(x1_samples, 'norm', args=(0, 1))
    assert p_value > 0.01  # Should not reject normality


@pytest.mark.unit
@pytest.mark.statistical
def test_x4_root_mechanism(ground_truth_scm, seed_everything):
    """Test X4 ~ N(2, 1) root mechanism."""
    seed_everything(42)
    scm = ground_truth_scm
    
    n_samples = 10000
    data = scm.generate(n_samples, interventions=None)
    x4_samples = data['X4'].numpy()
    
    # Test mean ≈ 2
    assert x4_samples.mean() == pytest.approx(2.0, abs=0.1)
    
    # Test std ≈ 1
    assert x4_samples.std() == pytest.approx(1.0, abs=0.1)
    
    # Kolmogorov-Smirnov test for normality around mean=2
    _, p_value = stats.kstest(x4_samples, 'norm', args=(2, 1))
    assert p_value > 0.01


@pytest.mark.unit
@pytest.mark.statistical
def test_x2_linear_mechanism(ground_truth_scm, seed_everything):
    """Test X2 = 2.0 * X1 + 1.0 + ε mechanism."""
    seed_everything(42)
    scm = ground_truth_scm
    
    n_samples = 10000
    data = scm.generate(n_samples, interventions=None)
    
    x1 = data['X1']
    x2 = data['X2']
    
    # Compute expected X2 (without noise)
    expected_x2 = 2.0 * x1 + 1.0
    
    # Residuals should be approximately normal with std ≈ 0.1
    residuals = (x2 - expected_x2).numpy()
    
    # Mean residual should be near 0
    assert residuals.mean() == pytest.approx(0.0, abs=0.05)
    
    # Std of residuals should be ≈ 0.1 (noise std)
    assert residuals.std() == pytest.approx(0.1, abs=0.02)


@pytest.mark.unit
@pytest.mark.statistical
def test_x3_collider_mechanism(ground_truth_scm, seed_everything):
    """Test X3 = 0.5 * X1 - X2 + sin(X2) + ε mechanism."""
    seed_everything(42)
    scm = ground_truth_scm
    
    n_samples = 10000
    data = scm.generate(n_samples, interventions=None)
    
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    
    # Compute expected X3 (without noise)
    expected_x3 = 0.5 * x1 - x2 + torch.sin(x2)
    
    # Residuals should be approximately normal with std ≈ 0.1
    residuals = (x3 - expected_x3).numpy()
    
    # Mean residual should be near 0
    assert residuals.mean() == pytest.approx(0.0, abs=0.05)
    
    # Std of residuals should be ≈ 0.1 (noise std)
    assert residuals.std() == pytest.approx(0.1, abs=0.02)


@pytest.mark.unit
@pytest.mark.statistical
def test_x5_quadratic_mechanism(ground_truth_scm, seed_everything):
    """Test X5 = 0.2 * X4^2 + ε mechanism."""
    seed_everything(42)
    scm = ground_truth_scm
    
    n_samples = 10000
    data = scm.generate(n_samples, interventions=None)
    
    x4 = data['X4']
    x5 = data['X5']
    
    # Compute expected X5 (without noise)
    expected_x5 = 0.2 * (x4 ** 2)
    
    # Residuals should be approximately normal with std ≈ 0.1
    residuals = (x5 - expected_x5).numpy()
    
    # Mean residual should be near 0
    assert residuals.mean() == pytest.approx(0.0, abs=0.05)
    
    # Std of residuals should be ≈ 0.1 (noise std)
    assert residuals.std() == pytest.approx(0.1, abs=0.02)


@pytest.mark.unit
def test_noise_variance(ground_truth_scm, seed_everything):
    """Test that noise variance is consistent (0.1) across mechanisms."""
    seed_everything(42)
    scm = ground_truth_scm
    
    n_samples = 10000
    data = scm.generate(n_samples, interventions=None)
    
    # X2 residuals
    x2_residuals = (data['X2'] - (2.0 * data['X1'] + 1.0)).numpy()
    assert x2_residuals.std() == pytest.approx(0.1, abs=0.02)
    
    # X3 residuals
    x3_residuals = (data['X3'] - (0.5 * data['X1'] - data['X2'] + torch.sin(data['X2']))).numpy()
    assert x3_residuals.std() == pytest.approx(0.1, abs=0.02)
    
    # X5 residuals
    x5_residuals = (data['X5'] - 0.2 * (data['X4'] ** 2)).numpy()
    assert x5_residuals.std() == pytest.approx(0.1, abs=0.02)


# =============================================================================
# Generation Tests - Observational
# =============================================================================

@pytest.mark.unit
def test_observational_generation_basic(ground_truth_scm, seed_everything):
    """Test basic observational data generation."""
    seed_everything(42)
    scm = ground_truth_scm
    
    n_samples = 100
    data = scm.generate(n_samples, interventions=None)
    
    # Check all nodes present
    assert set(data.keys()) == {'X1', 'X2', 'X3', 'X4', 'X5'}
    
    # Check correct shapes
    for node in data:
        assert data[node].shape == (n_samples,)


@pytest.mark.unit
def test_observational_generation_different_sample_sizes(ground_truth_scm, seed_everything):
    """Test generation with different sample sizes."""
    seed_everything(42)
    scm = ground_truth_scm
    
    for n in [1, 10, 100, 1000]:
        data = scm.generate(n_samples=n, interventions=None)
        
        for node in data:
            assert data[node].shape == (n,)


@pytest.mark.unit
def test_observational_samples_independent(ground_truth_scm, seed_everything):
    """Test that generated samples are independent."""
    seed_everything(42)
    scm = ground_truth_scm
    
    # Generate two batches
    data1 = scm.generate(n_samples=100, interventions=None)
    data2 = scm.generate(n_samples=100, interventions=None)
    
    # They should be different (not identical)
    for node in data1:
        assert not torch.allclose(data1[node], data2[node])


@pytest.mark.unit
def test_deterministic_with_seed(ground_truth_scm, seed_everything):
    """Test that generation is deterministic with same seed."""
    scm = ground_truth_scm
    
    # First run
    seed_everything(12345)
    data1 = scm.generate(n_samples=100, interventions=None)
    
    # Second run with same seed
    seed_everything(12345)
    data2 = scm.generate(n_samples=100, interventions=None)
    
    # Should be identical
    for node in data1:
        assert torch.allclose(data1[node], data2[node])


# =============================================================================
# Generation Tests - Interventional
# =============================================================================

@pytest.mark.unit
def test_intervention_on_x1(ground_truth_scm, seed_everything):
    """Test DO(X1=value) intervention."""
    seed_everything(42)
    scm = ground_truth_scm
    
    intervention_value = 5.0
    n_samples = 100
    
    data = scm.generate(n_samples, interventions={'X1': intervention_value})
    
    # X1 should be exactly the intervention value
    assert torch.all(data['X1'] == intervention_value)
    
    # X2 should be affected (X2 = 2*X1 + 1)
    expected_x2_mean = 2.0 * intervention_value + 1.0
    assert data['X2'].mean() == pytest.approx(expected_x2_mean, abs=0.5)
    
    # X4 should be unaffected (independent subgraph)
    assert data['X4'].mean() == pytest.approx(2.0, abs=0.5)


@pytest.mark.unit
def test_intervention_on_x2(ground_truth_scm, seed_everything):
    """Test DO(X2=value) intervention."""
    seed_everything(42)
    scm = ground_truth_scm
    
    intervention_value = 1.5
    n_samples = 100
    
    data = scm.generate(n_samples, interventions={'X2': intervention_value})
    
    # X2 should be exactly the intervention value
    assert torch.all(data['X2'] == intervention_value)
    
    # X1 should be unaffected (X1 is upstream)
    assert data['X1'].mean() == pytest.approx(0.0, abs=0.5)
    
    # X3 should be affected by the intervention
    # X3 = 0.5*X1 - X2 + sin(X2) + noise
    # With X2=1.5, the -X2 + sin(X2) part is constant
    assert data['X3'].std() > 0  # Should still have variance from X1 and noise


@pytest.mark.unit
def test_intervention_on_x4(ground_truth_scm, seed_everything):
    """Test DO(X4=value) intervention."""
    seed_everything(42)
    scm = ground_truth_scm
    
    intervention_value = 3.0
    n_samples = 100
    
    data = scm.generate(n_samples, interventions={'X4': intervention_value})
    
    # X4 should be exactly the intervention value
    assert torch.all(data['X4'] == intervention_value)
    
    # X5 should be affected (X5 = 0.2 * X4^2)
    expected_x5_mean = 0.2 * (intervention_value ** 2)
    assert data['X5'].mean() == pytest.approx(expected_x5_mean, abs=0.5)
    
    # X1, X2, X3 should be unaffected (independent subgraph)
    assert data['X1'].mean() == pytest.approx(0.0, abs=0.5)


@pytest.mark.unit
def test_intervention_overrides_mechanism(ground_truth_scm, seed_everything):
    """Test that intervention completely overrides natural mechanism."""
    seed_everything(42)
    scm = ground_truth_scm
    
    # Intervene on X2 (which normally depends on X1)
    intervention_value = 10.0
    data = scm.generate(n_samples=100, interventions={'X2': intervention_value})
    
    # X2 should be exactly intervention_value, regardless of X1
    assert torch.all(data['X2'] == intervention_value)
    
    # Verify X1 still varies (not affected by intervention on X2)
    assert data['X1'].std() > 0.5


@pytest.mark.unit
def test_multiple_interventions(ground_truth_scm, seed_everything):
    """Test multiple simultaneous interventions."""
    seed_everything(42)
    scm = ground_truth_scm
    
    interventions = {'X1': 2.0, 'X4': 3.0}
    data = scm.generate(n_samples=100, interventions=interventions)
    
    # Both interventions should be applied
    assert torch.all(data['X1'] == 2.0)
    assert torch.all(data['X4'] == 3.0)
    
    # Downstream nodes should be affected
    expected_x2_mean = 2.0 * 2.0 + 1.0
    assert data['X2'].mean() == pytest.approx(expected_x2_mean, abs=0.5)
    
    expected_x5_mean = 0.2 * (3.0 ** 2)
    assert data['X5'].mean() == pytest.approx(expected_x5_mean, abs=0.5)


@pytest.mark.unit
def test_intervention_breaks_correlation(ground_truth_scm, seed_everything):
    """Test that DO(X1) breaks natural correlation between X1 and X2."""
    seed_everything(42)
    scm = ground_truth_scm
    
    # Observational: X1 and X2 are correlated
    obs_data = scm.generate(n_samples=1000, interventions=None)
    obs_corr = np.corrcoef(obs_data['X1'].numpy(), obs_data['X2'].numpy())[0, 1]
    assert obs_corr > 0.9  # Strong positive correlation
    
    # Interventional: DO(X1=v) makes X1 constant, breaking correlation
    int_data = scm.generate(n_samples=1000, interventions={'X1': 1.0})
    # X1 is constant, so correlation is undefined/near-zero
    assert int_data['X1'].std() == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# Edge Cases and Invariants
# =============================================================================

@pytest.mark.unit
def test_no_nan_values(ground_truth_scm, seed_everything):
    """Test that generation never produces NaN values."""
    seed_everything(42)
    scm = ground_truth_scm
    
    # Test observational
    data = scm.generate(n_samples=1000, interventions=None)
    for node in data:
        assert not torch.any(torch.isnan(data[node]))
    
    # Test interventional
    data = scm.generate(n_samples=1000, interventions={'X2': 0.0})
    for node in data:
        assert not torch.any(torch.isnan(data[node]))


@pytest.mark.unit
def test_no_inf_values(ground_truth_scm, seed_everything):
    """Test that generation never produces infinite values."""
    seed_everything(42)
    scm = ground_truth_scm
    
    data = scm.generate(n_samples=1000, interventions=None)
    for node in data:
        assert not torch.any(torch.isinf(data[node]))


@pytest.mark.unit
def test_output_shape_consistency(ground_truth_scm, seed_everything):
    """Test that all nodes produce same number of samples."""
    seed_everything(42)
    scm = ground_truth_scm
    
    for n_samples in [1, 10, 100, 500]:
        data = scm.generate(n_samples=n_samples)
        
        shapes = [data[node].shape[0] for node in data]
        assert all(s == n_samples for s in shapes)


@pytest.mark.unit
def test_intervention_with_extreme_values(ground_truth_scm, seed_everything):
    """Test interventions with extreme values don't break generation."""
    seed_everything(42)
    scm = ground_truth_scm
    
    extreme_values = [-1000.0, -10.0, 0.0, 10.0, 1000.0]
    
    for val in extreme_values:
        data = scm.generate(n_samples=10, interventions={'X1': val})
        
        # Should complete without error
        assert data['X1'][0] == val
        
        # No NaN or inf
        for node in data:
            assert not torch.any(torch.isnan(data[node]))
            assert not torch.any(torch.isinf(data[node]))


@pytest.mark.unit
def test_intervention_on_collider(ground_truth_scm, seed_everything):
    """Test that intervening on collider X3 blocks backtracking."""
    seed_everything(42)
    scm = ground_truth_scm
    
    # DO(X3=value) should set X3 to exact value
    intervention_value = 5.0
    data = scm.generate(n_samples=100, interventions={'X3': intervention_value})
    
    assert torch.all(data['X3'] == intervention_value)
    
    # X1 and X2 should be unaffected (upstream nodes)
    assert data['X1'].mean() == pytest.approx(0.0, abs=0.5)
    assert data['X2'].mean() == pytest.approx(1.0, abs=0.5)  # 2*0 + 1


# =============================================================================
# Property-Based Tests (using hypothesis)
# =============================================================================

from hypothesis import given, strategies as st

@pytest.mark.unit
@given(intervention_value=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
def test_intervention_always_overrides_property(intervention_value):
    """Property test: Any intervention value should override the mechanism."""
    from ace_experiments import GroundTruthSCM
    
    torch.manual_seed(42)
    scm = GroundTruthSCM()
    
    data = scm.generate(n_samples=10, interventions={'X1': intervention_value})
    
    # Check all values match intervention value (within floating point precision)
    assert torch.allclose(data['X1'], torch.tensor(intervention_value), atol=1e-6)


@pytest.mark.unit
@given(n_samples=st.integers(min_value=1, max_value=1000))
def test_sample_size_invariance(n_samples):
    """Property test: Generation should work for any valid sample size."""
    from ace_experiments import GroundTruthSCM
    
    torch.manual_seed(42)
    scm = GroundTruthSCM()
    
    data = scm.generate(n_samples=n_samples)
    
    # All nodes should have correct shape
    for node in data:
        assert data[node].shape == (n_samples,)


@pytest.mark.unit
@given(seed=st.integers(min_value=0, max_value=10000))
def test_seed_reproducibility_property(seed):
    """Property test: Same seed always produces same output."""
    from ace_experiments import GroundTruthSCM
    
    scm = GroundTruthSCM()
    
    # First generation
    torch.manual_seed(seed)
    np.random.seed(seed)
    data1 = scm.generate(n_samples=10)
    
    # Second generation with same seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    data2 = scm.generate(n_samples=10)
    
    # Should be identical
    for node in data1:
        assert torch.allclose(data1[node], data2[node])


# =============================================================================
# Integration with CausalModel Base Class
# =============================================================================

@pytest.mark.unit
def test_inherits_from_causal_model(ground_truth_scm):
    """Test that GroundTruthSCM properly inherits from CausalModel."""
    from ace_experiments import CausalModel
    
    assert isinstance(ground_truth_scm, CausalModel)


@pytest.mark.unit
def test_has_required_attributes(ground_truth_scm):
    """Test that GroundTruthSCM has all required attributes."""
    scm = ground_truth_scm
    
    # From CausalModel
    assert hasattr(scm, 'graph')
    assert hasattr(scm, 'nodes')
    assert hasattr(scm, 'topo_order')
    assert hasattr(scm, 'get_parents')
    
    # Own methods
    assert hasattr(scm, 'mechanisms')
    assert hasattr(scm, 'generate')


@pytest.mark.unit
def test_get_parents_method(ground_truth_scm):
    """Test get_parents method works correctly."""
    scm = ground_truth_scm
    
    # Test returns list
    assert isinstance(scm.get_parents('X1'), list)
    
    # Test correct parents
    assert scm.get_parents('X3') == ['X1', 'X2'] or scm.get_parents('X3') == ['X2', 'X1']

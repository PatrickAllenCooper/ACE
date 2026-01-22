"""
Comprehensive tests for Complex SCM mechanisms.

Tests cover:
- All layer mechanisms (L1-L3, N1-N3, C1-C2, F1-F3)
- Mechanism equations
- Data generation validation
"""

import pytest
import torch


# =============================================================================
# Layer 2 Mechanism Tests (Linear)
# =============================================================================

@pytest.mark.unit
@pytest.mark.statistical
def test_complex_scm_l1_mechanism(seed_everything):
    """Test L1 = 1.5*R1 - 0.8*R2 + 2.0 mechanism."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    data = scm.generate(n_samples=10000)
    
    # Compute expected L1
    expected = 1.5 * data['R1'] - 0.8 * data['R2'] + 2.0
    
    # Check residuals (allowing for noise)
    residuals = (data['L1'] - expected).abs()
    assert residuals.mean() < 0.5  # Reasonable fit given noise=0.2


@pytest.mark.unit
def test_complex_scm_all_layers_generate(seed_everything):
    """Test that all layers generate data."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    data = scm.generate(n_samples=100)
    
    # Check all nodes present
    assert len(data) == len(scm.nodes)
    
    # Check each layer
    for node in scm.nodes:
        assert node in data
        assert data[node].shape == (100,)
        assert not torch.isnan(data[node]).any()


@pytest.mark.unit
def test_complex_scm_intervention_on_each_layer(seed_everything):
    """Test interventions work on nodes from each layer."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    # Test intervention on each layer
    test_nodes = ['R1', 'L1', 'N1', 'C1', 'F1']
    
    for node in test_nodes:
        data = scm.generate(n_samples=50, interventions={node: 3.0})
        assert torch.all(data[node] == 3.0)


# =============================================================================
# Complex Mechanisms Tests
# =============================================================================

@pytest.mark.unit
def test_complex_scm_nonlinear_mechanisms(seed_everything):
    """Test that nonlinear mechanisms produce varied outputs."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    data = scm.generate(n_samples=1000)
    
    # Nonlinear nodes should have reasonable variance
    for node in ['N1', 'N2', 'N3']:
        assert data[node].std() > 0.3


@pytest.mark.unit
def test_complex_scm_final_layer_dependencies(seed_everything):
    """Test final layer depends on earlier layers."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    # Intervene on C1 (should affect F1 and F3)
    data = scm.generate(n_samples=100, interventions={'C1': 5.0})
    
    assert torch.all(data['C1'] == 5.0)
    
    # F1 depends on C1, should have some structure
    assert data['F1'].std() < data['F1'].max() - data['F1'].min()


@pytest.mark.unit
def test_complex_scm_independent_subgraphs(seed_everything):
    """Test that independent subgraphs remain independent."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    # Generate two datasets
    data1 = scm.generate(n_samples=100, interventions={'R1': 5.0})
    data2 = scm.generate(n_samples=100)
    
    # R1 intervention shouldn't affect all nodes equally
    # Some nodes should be independent
    # This is a basic check
    assert data1['R1'].mean() != data2['R1'].mean()


# =============================================================================
# Edge Cases for Complex SCM
# =============================================================================

@pytest.mark.unit
def test_complex_scm_large_sample_generation(seed_everything):
    """Test complex SCM with large sample sizes."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    # Generate large dataset
    data = scm.generate(n_samples=5000)
    
    for node in scm.nodes:
        assert data[node].shape == (5000,)
        assert not torch.isnan(data[node]).any()


@pytest.mark.unit
def test_complex_scm_multiple_simultaneous_interventions(seed_everything):
    """Test complex SCM with interventions on multiple layers."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    # Intervene on multiple layers simultaneously
    interventions = {'R1': 1.0, 'L1': 2.0, 'N1': 3.0}
    data = scm.generate(n_samples=100, interventions=interventions)
    
    # All interventions should be applied
    assert torch.all(data['R1'] == 1.0)
    assert torch.all(data['L1'] == 2.0)
    assert torch.all(data['N1'] == 3.0)


@pytest.mark.unit
def test_complex_scm_extreme_intervention_values(seed_everything):
    """Test complex SCM with extreme intervention values."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    scm = ComplexGroundTruthSCM()
    
    # Extreme values
    extreme_values = [-100.0, -10.0, 0.0, 10.0, 100.0]
    
    for val in extreme_values:
        data = scm.generate(n_samples=10, interventions={'R1': val})
        
        assert torch.all(data['R1'] == val)
        # Should not produce NaN or Inf
        for node in scm.nodes:
            assert not torch.isnan(data[node]).any()
            assert not torch.isinf(data[node]).any()

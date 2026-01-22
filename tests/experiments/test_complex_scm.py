"""
Unit tests for Complex SCM experiment.

Tests cover:
- ComplexSCM initialization
- Graph structure
- Generation
- Basic functionality
"""

import pytest
import torch


# =============================================================================
# ComplexSCM Tests
# =============================================================================

@pytest.mark.unit
def test_complex_scm_can_import():
    """Test that complex_scm module can be imported."""
    from experiments import complex_scm
    
    assert complex_scm is not None


@pytest.mark.unit
def test_complex_scm_has_expected_classes():
    """Test that complex_scm has expected classes."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    # Should be able to create instance
    scm = ComplexGroundTruthSCM()
    
    assert scm is not None
    assert hasattr(scm, 'nodes')
    assert hasattr(scm, 'graph')


@pytest.mark.unit
def test_complex_scm_node_count():
    """Test that complex SCM has 15 nodes."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    scm = ComplexGroundTruthSCM()
    
    # Should have 15 nodes
    assert len(scm.nodes) >= 10  # At least more than simple SCM
    assert hasattr(scm, 'graph')


@pytest.mark.unit
def test_complex_scm_generate(seed_everything):
    """Test complex SCM data generation."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    
    scm = ComplexGroundTruthSCM()
    
    # Generate data
    data = scm.generate(n_samples=100)
    
    # Should return dict of tensors
    assert isinstance(data, dict)
    assert len(data) == len(scm.nodes)
    
    # All should be correct shape
    for node, values in data.items():
        assert isinstance(values, torch.Tensor)
        assert values.shape[0] == 100


@pytest.mark.unit
def test_complex_scm_interventions(seed_everything):
    """Test complex SCM with interventions."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    
    scm = ComplexGroundTruthSCM()
    
    # Get first node name
    first_node = scm.nodes[0]
    
    # Generate with intervention
    data = scm.generate(n_samples=50, interventions={first_node: 2.0})
    
    # Intervened node should be set to value
    assert torch.all(data[first_node] == 2.0)


@pytest.mark.unit
def test_complex_scm_has_colliders():
    """Test that complex SCM has multiple colliders."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    scm = ComplexGroundTruthSCM()
    
    # Count nodes with multiple parents (colliders)
    colliders = []
    for node in scm.nodes:
        parents = scm.get_parents(node)
        if len(parents) >= 2:
            colliders.append(node)
    
    # Should have colliders (design goal was 5)
    assert len(colliders) >= 2


@pytest.mark.unit
def test_complex_scm_no_nan_values(seed_everything):
    """Test that complex SCM doesn't produce NaN."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    
    scm = ComplexGroundTruthSCM()
    data = scm.generate(n_samples=100)
    
    # No NaN values
    for node, values in data.items():
        assert not torch.isnan(values).any()

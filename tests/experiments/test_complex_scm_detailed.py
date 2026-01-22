"""
Detailed tests for Complex SCM mechanisms and structure.

Tests cover:
- Individual layer mechanisms
- Collider structures
- Nested colliders
- Hierarchical dependencies
"""

import pytest
import torch


# =============================================================================
# Complex SCM Structure Tests
# =============================================================================

@pytest.mark.unit
def test_complex_scm_15_nodes():
    """Test that complex SCM has 14-15 nodes."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    scm = ComplexGroundTruthSCM()
    
    # Should have 14-15 nodes (verify implementation)
    assert len(scm.nodes) >= 14
    assert len(scm.nodes) <= 15


@pytest.mark.unit
def test_complex_scm_layer_structure():
    """Test hierarchical layer structure."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    scm = ComplexGroundTruthSCM()
    
    # Layer 1: Roots (R1, R2, R3)
    roots = [n for n in scm.nodes if n.startswith('R')]
    assert len(roots) == 3
    
    # Layer 2: Linear (L1, L2, L3)
    linear = [n for n in scm.nodes if n.startswith('L')]
    assert len(linear) == 3
    
    # Layer 3: Nonlinear (N1, N2, N3)
    nonlinear = [n for n in scm.nodes if n.startswith('N')]
    assert len(nonlinear) == 3
    
    # Layer 4: Complex (C1, C2)
    complex_nodes = [n for n in scm.nodes if n.startswith('C')]
    assert len(complex_nodes) == 2
    
    # Layer 5: Final (F1, F2, F3)
    final = [n for n in scm.nodes if n.startswith('F')]
    assert len(final) == 3


@pytest.mark.unit
def test_complex_scm_collider_count():
    """Test that complex SCM has multiple colliders."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    scm = ComplexGroundTruthSCM()
    
    # Count colliders (nodes with 2+ parents)
    colliders = []
    for node in scm.nodes:
        parents = scm.get_parents(node)
        if len(parents) >= 2:
            colliders.append(node)
    
    # Should have 4-5 colliders as designed
    assert len(colliders) >= 4
    
    # Specific colliders: L1, N1, C1, C2, F3
    assert 'L1' in colliders  # R1, R2
    assert 'N1' in colliders  # L1, L2 (nested)
    assert 'C1' in colliders  # N1, N2
    assert 'C2' in colliders  # N2, N3
    assert 'F3' in colliders  # C1, C2


@pytest.mark.unit
def test_complex_scm_root_mechanisms(seed_everything):
    """Test root node mechanisms."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    
    scm = ComplexGroundTruthSCM()
    data = scm.generate(n_samples=10000)
    
    # R1 ~ N(0, 1)
    assert data['R1'].mean() == pytest.approx(0.0, abs=0.1)
    assert data['R1'].std() == pytest.approx(1.0, abs=0.1)
    
    # R2 ~ N(1, 1.5)
    assert data['R2'].mean() == pytest.approx(1.0, abs=0.2)
    assert data['R2'].std() == pytest.approx(1.5, abs=0.2)
    
    # R3 ~ N(-0.5, 0.8)
    assert data['R3'].mean() == pytest.approx(-0.5, abs=0.1)
    assert data['R3'].std() == pytest.approx(0.8, abs=0.1)


@pytest.mark.unit
def test_complex_scm_intervention_propagation(seed_everything):
    """Test intervention propagates through layers."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    
    scm = ComplexGroundTruthSCM()
    
    # Intervene on R1
    data = scm.generate(n_samples=100, interventions={'R1': 5.0})
    
    # R1 should be fixed
    assert torch.all(data['R1'] == 5.0)
    
    # Downstream should be affected (L1, L2, N1, etc.)
    # R2 should be unaffected (independent root)
    assert data['R2'].std() > 0.5


@pytest.mark.unit
def test_complex_scm_nested_collider():
    """Test nested collider structure (N1 depends on L1 which is a collider)."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    scm = ComplexGroundTruthSCM()
    
    # N1 parents
    n1_parents = scm.get_parents('N1')
    assert len(n1_parents) == 2
    assert 'L1' in n1_parents
    assert 'L2' in n1_parents
    
    # L1 itself is a collider
    l1_parents = scm.get_parents('L1')
    assert len(l1_parents) == 2
    assert 'R1' in l1_parents
    assert 'R2' in l1_parents


@pytest.mark.unit
def test_complex_scm_no_cycles():
    """Test that complex SCM is a DAG (no cycles)."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    import networkx as nx
    
    scm = ComplexGroundTruthSCM()
    
    # Build networkx graph
    G = nx.DiGraph()
    for node in scm.nodes:
        parents = scm.get_parents(node)
        for parent in parents:
            G.add_edge(parent, node)
    
    # Should be DAG
    assert nx.is_directed_acyclic_graph(G)


@pytest.mark.unit
def test_complex_scm_higher_noise():
    """Test that complex SCM uses higher noise (0.2 vs 0.1)."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    scm = ComplexGroundTruthSCM()
    
    assert scm.noise_std == 0.2


@pytest.mark.unit
def test_complex_scm_multiple_interventions(seed_everything):
    """Test complex SCM with multiple simultaneous interventions."""
    from experiments.complex_scm import ComplexGroundTruthSCM
    
    seed_everything(42)
    
    scm = ComplexGroundTruthSCM()
    
    interventions = {'R1': 2.0, 'R3': -1.0}
    data = scm.generate(n_samples=50, interventions=interventions)
    
    # Both interventions applied
    assert torch.all(data['R1'] == 2.0)
    assert torch.all(data['R3'] == -1.0)
    
    # R2 unaffected
    assert data['R2'].std() > 0.5

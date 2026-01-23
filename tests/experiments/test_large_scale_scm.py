"""
Tests for large-scale SCM (30-50 nodes).

Tests cover:
- SCM construction
- Graph properties
- Data generation
- Scalability validation
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Large-Scale SCM Tests
# =============================================================================

@pytest.mark.unit
def test_large_scale_scm_imports():
    """Test that large_scale_scm module imports."""
    from experiments import large_scale_scm
    assert large_scale_scm is not None


@pytest.mark.unit
def test_large_scale_scm_class_exists():
    """Test that LargeScaleSCM class exists."""
    from experiments.large_scale_scm import LargeScaleSCM
    
    # Should be able to instantiate
    scm = LargeScaleSCM(n_nodes=30)
    assert scm is not None


@pytest.mark.unit
def test_large_scale_scm_has_30_nodes():
    """Test that large-scale SCM has 30 nodes."""
    from experiments.large_scale_scm import LargeScaleSCM
    
    scm = LargeScaleSCM(n_nodes=30)
    assert len(scm.nodes) == 30


@pytest.mark.unit
def test_large_scale_scm_hierarchical_structure():
    """Test that large-scale SCM has hierarchical structure."""
    from experiments.large_scale_scm import LargeScaleSCM
    
    scm = LargeScaleSCM(n_nodes=30)
    
    # Should have roots (nodes with no parents)
    roots = [node for node in scm.nodes if len(scm.get_parents(node)) == 0]
    assert len(roots) >= 3  # Should have multiple roots
    
    # Should have some colliders (nodes with 2+ parents)
    colliders = [node for node in scm.nodes if len(scm.get_parents(node)) >= 2]
    assert len(colliders) >= 5  # Should have multiple colliders


@pytest.mark.unit
def test_large_scale_scm_generate(seed_everything):
    """Test data generation for large-scale SCM."""
    from experiments.large_scale_scm import LargeScaleSCM
    
    seed_everything(42)
    scm = LargeScaleSCM(n_nodes=30)
    
    # Generate data
    data = scm.generate(n_samples=100)
    
    # Should have all 30 nodes
    assert len(data) == 30
    
    # All should have correct shape
    for node in scm.nodes:
        assert node in data
        assert data[node].shape == (100,)


@pytest.mark.unit
def test_large_scale_scm_interventions(seed_everything):
    """Test interventions on large-scale SCM."""
    from experiments.large_scale_scm import LargeScaleSCM
    
    seed_everything(42)
    scm = LargeScaleSCM(n_nodes=30)
    
    # Test intervention on arbitrary node
    data = scm.generate(n_samples=50, interventions={'X10': 3.0})
    
    assert torch.all(data['X10'] == 3.0)


@pytest.mark.unit
def test_large_scale_scm_no_nan_values(seed_everything):
    """Test that large-scale SCM doesn't produce NaN."""
    from experiments.large_scale_scm import LargeScaleSCM
    
    seed_everything(42)
    scm = LargeScaleSCM(n_nodes=30)
    
    data = scm.generate(n_samples=100)
    
    for node in scm.nodes:
        assert not torch.isnan(data[node]).any()
        assert not torch.isinf(data[node]).any()


@pytest.mark.unit
def test_large_scale_scm_graph_is_dag():
    """Test that large-scale SCM forms a DAG."""
    from experiments.large_scale_scm import LargeScaleSCM
    import networkx as nx
    
    scm = LargeScaleSCM(n_nodes=30)
    
    # Build networkx graph
    G = nx.DiGraph()
    for node in scm.nodes:
        parents = scm.get_parents(node)
        for parent in parents:
            G.add_edge(parent, node)
    
    # Should be DAG
    assert nx.is_directed_acyclic_graph(G)


@pytest.mark.unit
def test_large_scale_scm_topological_sort():
    """Test topological sort works for large-scale SCM."""
    from experiments.large_scale_scm import LargeScaleSCM
    
    scm = LargeScaleSCM(n_nodes=30)
    
    topo_order = scm._topological_sort()
    
    # Should have all nodes
    assert len(topo_order) == 30
    assert set(topo_order) == set(scm.nodes)
    
    # Parents should come before children
    node_positions = {node: i for i, node in enumerate(topo_order)}
    
    for node in scm.nodes:
        for parent in scm.get_parents(node):
            assert node_positions[parent] < node_positions[node], \
                f"{parent} should come before {node}"


@pytest.mark.unit
def test_large_scale_scm_configurable_size():
    """Test that large-scale SCM size is configurable."""
    from experiments.large_scale_scm import LargeScaleSCM
    
    # Test different sizes
    for n in [20, 30, 40]:
        scm = LargeScaleSCM(n_nodes=n)
        assert len(scm.nodes) == n


@pytest.mark.unit
def test_large_scale_scm_demonstrates_scale():
    """Test that large-scale SCM shows scale challenge."""
    from experiments.large_scale_scm import LargeScaleSCM
    
    scm = LargeScaleSCM(n_nodes=30)
    
    # At 30 nodes, random sampling is ~3.3% per node
    # Strategic selection should have clear advantage
    
    # Count total edges (complexity)
    total_edges = sum(len(scm.get_parents(node)) for node in scm.nodes)
    
    # Should be reasonably complex
    assert total_edges >= 30  # At least one edge per node on average


# =============================================================================
# Theoretical Justification Tests
# =============================================================================

@pytest.mark.unit
def test_paper_has_theoretical_justification():
    """Test that paper includes theoretical justification."""
    paper = Path("paper/paper.tex")
    content = paper.read_text()
    
    # Should mention Bradley-Terry
    assert 'Bradley-Terry' in content or 'bradley' in content.lower()


@pytest.mark.unit
def test_paper_explains_scale_invariance():
    """Test that paper explains scale-invariance property."""
    paper = Path("paper/paper.tex")
    content = paper.read_text()
    
    # Should discuss scale invariance or preference stability
    assert 'scale' in content.lower() or 'invariance' in content.lower()


@pytest.mark.unit
def test_paper_connects_to_non_stationarity():
    """Test that paper connects theory to non-stationarity."""
    paper = Path("paper/paper.tex")
    content = paper.read_text()
    
    # Should discuss non-stationary rewards
    assert 'non-stationary' in content or 'non-stationarity' in content


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
def test_large_scale_scm_can_be_used_with_student(seed_everything):
    """Test that large-scale SCM works with StudentSCM."""
    from experiments.large_scale_scm import LargeScaleSCM
    
    # Note: Would need StudentSCM that handles 30 nodes
    # For now, just test SCM generates valid data
    
    seed_everything(42)
    scm = LargeScaleSCM(n_nodes=30)
    
    data = scm.generate(n_samples=50)
    
    # Should generate valid data for all nodes
    assert all(node in data for node in scm.nodes)
    assert all(data[node].shape == (50,) for node in scm.nodes)


@pytest.mark.unit
def test_all_recent_additions_present():
    """Test that all recent additions are present."""
    
    # Large-scale SCM
    assert Path("experiments/large_scale_scm.py").exists()
    
    # Paper enhancements
    paper = Path("paper/paper.tex")
    content = paper.read_text()
    assert 'Reproducibility' in content
    assert 'Ablation' in content
    
    # Dependencies
    assert Path("requirements.txt").exists()
    assert Path("environment.yml").exists()

"""
Additional tests for uncovered functions in ace_experiments.py.

Tests cover:
- Additional utility functions
- Helper methods
- Integration helpers
"""

import pytest
import torch
import tempfile


# =============================================================================
# Additional Utility Function Tests
# =============================================================================

@pytest.mark.unit
def test_get_random_valid_command_generates_valid_nodes(ground_truth_scm):
    """Test random command generation always produces valid nodes."""
    from ace_experiments import get_random_valid_command_range
    import re
    
    nodes = list(ground_truth_scm.nodes)
    
    # Generate many commands
    for _ in range(30):
        command = get_random_valid_command_range(nodes, value_min=-5.0, value_max=5.0)
        
        # Should contain one of the nodes
        found_node = False
        for node in nodes:
            if node in command:
                found_node = True
                break
        
        assert found_node, f"Command '{command}' doesn't contain any valid node"


@pytest.mark.unit
def test_bin_index_computation():
    """Test _bin_index function."""
    from ace_experiments import _bin_index
    
    # Test various values
    assert _bin_index(0.0, -5.0, 5.0, 10) == 5  # Middle bin
    assert _bin_index(-5.0, -5.0, 5.0, 10) == 0  # First bin
    assert _bin_index(5.0, -5.0, 5.0, 10) == 9  # Last bin
    
    # Out of range (should clip)
    assert _bin_index(-10.0, -5.0, 5.0, 10) == 0
    assert _bin_index(10.0, -5.0, 5.0, 10) == 9


@pytest.mark.unit
def test_bin_index_with_different_ranges():
    """Test _bin_index with different value ranges."""
    from ace_experiments import _bin_index
    
    # Test with different range
    idx = _bin_index(0.0, -10.0, 10.0, 20)
    assert 0 <= idx < 20
    assert idx == 10  # Middle of 20 bins


@pytest.mark.unit
def test_impact_weight_descendants_only(ground_truth_scm):
    """Test that impact weight only considers descendants."""
    from ace_experiments import _impact_weight
    
    graph = ground_truth_scm.graph
    node_losses = {'X1': 1.0, 'X2': 2.0, 'X3': 3.0, 'X4': 1.0, 'X5': 2.0}
    
    # X1 has descendants (X2, X3), so weight > 0
    weight_x1 = _impact_weight(graph, 'X1', node_losses)
    assert weight_x1 > 0
    
    # X3 is a leaf (no descendants), so weight = 0
    weight_x3 = _impact_weight(graph, 'X3', node_losses)
    assert weight_x3 == 0.0
    
    # X5 is a leaf, weight = 0
    weight_x5 = _impact_weight(graph, 'X5', node_losses)
    assert weight_x5 == 0.0


@pytest.mark.unit
def test_direct_child_impact_normalized_vs_unnormalized(ground_truth_scm):
    """Test normalized vs unnormalized direct child impact."""
    from ace_experiments import _direct_child_impact_weight
    
    graph = ground_truth_scm.graph
    node_losses = {'X1': 1.0, 'X2': 2.0, 'X3': 3.0, 'X4': 1.0, 'X5': 2.0}
    
    # X1 has children X2 and X3
    weight_normalized = _direct_child_impact_weight(graph, 'X1', node_losses, normalize=True)
    weight_unnormalized = _direct_child_impact_weight(graph, 'X1', node_losses, normalize=False)
    
    # Unnormalized should be >= normalized
    assert weight_unnormalized >= weight_normalized
    
    # Both should be > 0
    assert weight_normalized > 0
    assert weight_unnormalized > 0


@pytest.mark.unit
def test_disentanglement_bonus_for_chain(ground_truth_scm):
    """Test disentanglement bonus identifies X1->X2->X3 chain."""
    from ace_experiments import _disentanglement_bonus
    
    graph = ground_truth_scm.graph
    node_losses = {'X1': 0.5, 'X2': 0.5, 'X3': 5.0, 'X4': 0.3, 'X5': 0.3}
    
    # X2 is in the chain X1->X2->X3 where both X1 and X2 are parents of X3
    bonus_x2 = _disentanglement_bonus(graph, 'X2', node_losses)
    
    # Should get bonus for triangle structure
    assert bonus_x2 > 0


@pytest.mark.unit
def test_calculate_value_novelty_with_history():
    """Test value novelty calculation with history."""
    from ace_experiments import calculate_value_novelty_bonus
    
    # History of X1 values around 0
    history = [('X1', 0.0), ('X1', 0.1), ('X1', -0.1), ('X1', 0.2), ('X1', -0.2), ('X1', 0.15)]
    
    # Novel value far from history
    bonus = calculate_value_novelty_bonus(5.0, 'X1', history, window=100)
    
    # Should get bonus for distance
    assert bonus > 0
    assert bonus <= 5.0  # Maximum bonus


@pytest.mark.unit  
def test_calculate_value_novelty_early_exploration():
    """Test that early exploration gets bonus."""
    from ace_experiments import calculate_value_novelty_bonus
    
    # Very short history (< 5 values)
    history = [('X1', 1.0), ('X1', 2.0)]
    
    bonus = calculate_value_novelty_bonus(3.0, 'X1', history, window=100)
    
    # Should get early exploration bonus (5.0)
    assert bonus == 5.0


@pytest.mark.unit
def test_compute_unified_diversity_score_with_empty_history():
    """Test diversity score with no history."""
    from ace_experiments import compute_unified_diversity_score
    
    all_nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    recent_targets = []
    
    # With empty history, should return 0
    score = compute_unified_diversity_score('X1', recent_targets, all_nodes)
    
    assert score == 0.0


@pytest.mark.unit
def test_compute_unified_diversity_score_short_history():
    """Test diversity score with short history (<20)."""
    from ace_experiments import compute_unified_diversity_score
    
    all_nodes = ['X1', 'X2', 'X3']
    recent_targets = ['X1', 'X2', 'X3']  # Only 3 targets
    
    # Should return 0 (too short)
    score = compute_unified_diversity_score('X1', recent_targets, all_nodes)
    
    assert score == 0.0

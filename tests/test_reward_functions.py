"""
Unit tests for reward and scoring utility functions.

Tests cover:
- Impact weight computation
- Direct child impact weight
- Disentanglement bonus
- Value novelty bonus
- Unified diversity score
- Teacher command generation
"""

import pytest
import torch
import networkx as nx


# =============================================================================
# Impact Weight Tests
# =============================================================================

@pytest.mark.unit
def test_impact_weight_basic(ground_truth_scm):
    """Test basic impact weight computation."""
    from ace_experiments import _impact_weight
    
    graph = ground_truth_scm.graph
    node_losses = {'X1': 0.5, 'X2': 1.0, 'X3': 2.0, 'X4': 0.3, 'X5': 0.8}
    
    # X1 affects X2 and X3 (descendants)
    weight_x1 = _impact_weight(graph, 'X1', node_losses)
    
    # X4 affects only X5
    weight_x4 = _impact_weight(graph, 'X4', node_losses)
    
    # Both should be positive
    assert weight_x1 > 0
    assert weight_x4 > 0
    
    # X1 affects more nodes with higher losses, so should have higher weight
    assert weight_x1 > weight_x4


@pytest.mark.unit
def test_impact_weight_considers_descendants(ground_truth_scm):
    """Test that impact weight considers all descendants."""
    from ace_experiments import _impact_weight
    
    graph = ground_truth_scm.graph
    
    # High loss on X3 (descendant of X1 and X2)
    node_losses = {'X1': 0.1, 'X2': 0.1, 'X3': 10.0, 'X4': 0.1, 'X5': 0.1}
    
    # X1 and X2 both affect X3, should have high impact
    weight_x1 = _impact_weight(graph, 'X1', node_losses)
    weight_x2 = _impact_weight(graph, 'X2', node_losses)
    
    # X4 doesn't affect X3
    weight_x4 = _impact_weight(graph, 'X4', node_losses)
    
    assert weight_x1 > weight_x4
    assert weight_x2 > weight_x4


@pytest.mark.unit
def test_impact_weight_leaf_node(ground_truth_scm):
    """Test impact weight for leaf node (no descendants)."""
    from ace_experiments import _impact_weight
    
    graph = ground_truth_scm.graph
    node_losses = {'X1': 1.0, 'X2': 1.0, 'X3': 1.0, 'X4': 1.0, 'X5': 1.0}
    
    # X3 and X5 are leaves (no descendants)
    weight_x3 = _impact_weight(graph, 'X3', node_losses)
    weight_x5 = _impact_weight(graph, 'X5', node_losses)
    
    # Impact weight only considers descendants, leaves have none
    assert weight_x3 == 0.0
    assert weight_x5 == 0.0


# =============================================================================
# Direct Child Impact Weight Tests
# =============================================================================

@pytest.mark.unit
def test_direct_child_impact_weight_basic(ground_truth_scm):
    """Test direct child impact weight computation."""
    from ace_experiments import _direct_child_impact_weight
    
    graph = ground_truth_scm.graph
    node_losses = {'X1': 0.5, 'X2': 1.0, 'X3': 2.0, 'X4': 0.3, 'X5': 0.8}
    
    # X1 has direct children X2, X3
    weight_x1 = _direct_child_impact_weight(graph, 'X1', node_losses)
    
    # X2 has direct child X3
    weight_x2 = _direct_child_impact_weight(graph, 'X2', node_losses)
    
    # Both should be positive
    assert weight_x1 > 0
    assert weight_x2 > 0


@pytest.mark.unit
def test_direct_child_impact_weight_normalized(ground_truth_scm):
    """Test normalized direct child impact weight."""
    from ace_experiments import _direct_child_impact_weight
    
    graph = ground_truth_scm.graph
    node_losses = {'X1': 0.5, 'X2': 1.0, 'X3': 2.0, 'X4': 0.3, 'X5': 0.8}
    
    # With normalization (default)
    weight_norm = _direct_child_impact_weight(graph, 'X1', node_losses, normalize=True)
    
    # Without normalization
    weight_raw = _direct_child_impact_weight(graph, 'X1', node_losses, normalize=False)
    
    # Normalized should be <= raw (divided by number of children)
    assert weight_norm <= weight_raw


@pytest.mark.unit
def test_direct_child_impact_weight_no_children(ground_truth_scm):
    """Test direct child impact for node with no children."""
    from ace_experiments import _direct_child_impact_weight
    
    graph = ground_truth_scm.graph
    node_losses = {'X1': 1.0, 'X2': 1.0, 'X3': 1.0, 'X4': 1.0, 'X5': 1.0}
    
    # X3 and X5 have no children
    weight_x3 = _direct_child_impact_weight(graph, 'X3', node_losses)
    weight_x5 = _direct_child_impact_weight(graph, 'X5', node_losses)
    
    # Should return 0 (no children to impact)
    assert weight_x3 == 0
    assert weight_x5 == 0


# =============================================================================
# Disentanglement Bonus Tests
# =============================================================================

@pytest.mark.unit
def test_disentanglement_bonus_for_collider_parent(ground_truth_scm):
    """Test disentanglement bonus for parent of collider."""
    from ace_experiments import _disentanglement_bonus
    
    graph = ground_truth_scm.graph
    node_losses = {'X1': 0.5, 'X2': 0.5, 'X3': 5.0, 'X4': 0.3, 'X5': 0.3}
    
    # X2 depends on X1, both affect X3
    # Intervening on X2 breaks X1->X2->X3 chain
    bonus_x2 = _disentanglement_bonus(graph, 'X2', node_losses)
    
    # X2 should get bonus (has X1->X2 edge and both are parents of X3)
    assert bonus_x2 > 0
    
    # X1 doesn't get bonus (no parent points to it before X3)
    bonus_x1 = _disentanglement_bonus(graph, 'X1', node_losses)
    assert bonus_x1 == 0.0


@pytest.mark.unit
def test_disentanglement_bonus_for_non_collider_parent(ground_truth_scm):
    """Test disentanglement bonus for nodes not involved in colliders."""
    from ace_experiments import _disentanglement_bonus
    
    graph = ground_truth_scm.graph
    node_losses = {'X1': 0.5, 'X2': 0.5, 'X3': 0.5, 'X4': 0.3, 'X5': 5.0}
    
    # X4 -> X5 is not a collider
    bonus_x4 = _disentanglement_bonus(graph, 'X4', node_losses)
    
    # Should be 0 (X5 only has one parent)
    assert bonus_x4 == 0


@pytest.mark.unit
def test_disentanglement_bonus_scales_with_collider_loss(ground_truth_scm):
    """Test that disentanglement bonus scales with collider loss."""
    from ace_experiments import _disentanglement_bonus
    
    graph = ground_truth_scm.graph
    
    # X2 gets the disentanglement bonus (X1->X2->X3 structure)
    # Low collider loss
    node_losses_low = {'X1': 0.5, 'X2': 0.5, 'X3': 1.0, 'X4': 0.3, 'X5': 0.3}
    bonus_low = _disentanglement_bonus(graph, 'X2', node_losses_low)
    
    # High collider loss
    node_losses_high = {'X1': 0.5, 'X2': 0.5, 'X3': 10.0, 'X4': 0.3, 'X5': 0.3}
    bonus_high = _disentanglement_bonus(graph, 'X2', node_losses_high)
    
    # Higher collider loss should give higher bonus
    assert bonus_high > bonus_low


# =============================================================================
# Value Novelty Bonus Tests
# =============================================================================

@pytest.mark.unit
def test_value_novelty_bonus_for_novel_value():
    """Test novelty bonus for completely novel value."""
    from ace_experiments import calculate_value_novelty_bonus
    
    # History as list of (target, value) tuples - values around 0 for X1
    value_history = [
        ('X1', 0.0), ('X1', 0.1), ('X1', -0.1), ('X1', 0.2), ('X1', -0.2),
        ('X1', 0.15), ('X2', 1.0)  # Also some for other nodes
    ]
    
    # Novel value far from history
    bonus = calculate_value_novelty_bonus(5.0, 'X1', value_history, window=100)
    
    # Should get bonus for novelty (distance > 0)
    assert bonus > 0


@pytest.mark.unit
def test_value_novelty_bonus_for_repeated_value():
    """Test novelty bonus for repeated value."""
    from ace_experiments import calculate_value_novelty_bonus
    
    # History with repeated value for X1
    value_history = [
        ('X1', 1.0), ('X1', 1.0), ('X1', 1.0), ('X1', 1.0), ('X1', 1.0), ('X1', 1.0)
    ]
    
    # Same value again
    bonus = calculate_value_novelty_bonus(1.0, 'X1', value_history, window=100)
    
    # Should have low or zero bonus (distance = 0)
    assert bonus == 0


@pytest.mark.unit
def test_value_novelty_bonus_with_empty_history():
    """Test novelty bonus with no history."""
    from ace_experiments import calculate_value_novelty_bonus
    
    value_history = []
    
    # First value for X1 (less than 5 values in history)
    bonus = calculate_value_novelty_bonus(2.0, 'X1', value_history, window=100)
    
    # Should get early exploration bonus
    assert bonus == 5.0


@pytest.mark.unit
def test_value_novelty_bonus_respects_window():
    """Test that novelty bonus only considers recent window."""
    from ace_experiments import calculate_value_novelty_bonus
    
    # Long history: old values are 5.0, recent values are 0.0
    value_history = [('X1', 5.0)] * 50 + [('X1', 0.0)] * 50
    
    # Value of 5.0 is old, should get novelty bonus relative to recent window
    bonus = calculate_value_novelty_bonus(5.0, 'X1', value_history, window=50)
    
    # Should have bonus (5.0 far from recent 0.0 values)
    assert bonus > 0


# =============================================================================
# Unified Diversity Score Tests  
# =============================================================================

@pytest.mark.unit
def test_unified_diversity_score_balanced_distribution():
    """Test diversity score with balanced target distribution."""
    from ace_experiments import compute_unified_diversity_score
    
    all_nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    
    # Balanced recent targets
    recent_targets = ['X1', 'X2', 'X3', 'X4', 'X5', 'X1', 'X2', 'X3', 'X4', 'X5']
    
    for target in all_nodes:
        score = compute_unified_diversity_score(target, recent_targets, all_nodes)
        
        # All should have similar (good) scores
        assert score > -1.0  # Not heavily penalized


@pytest.mark.unit
def test_unified_diversity_score_concentrated_distribution():
    """Test diversity score with concentrated distribution."""
    from ace_experiments import compute_unified_diversity_score
    
    all_nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    
    # Heavily concentrated on X1
    recent_targets = ['X1'] * 80 + ['X2'] * 5 + ['X3'] * 5
    
    # X1 should be penalized
    score_x1 = compute_unified_diversity_score('X1', recent_targets, all_nodes)
    
    # X4 should be rewarded (undersampled)
    score_x4 = compute_unified_diversity_score('X4', recent_targets, all_nodes)
    
    assert score_x4 > score_x1  # Undersampled gets better score


@pytest.mark.unit
def test_unified_diversity_score_with_max_concentration():
    """Test diversity score respects max concentration parameter."""
    from ace_experiments import compute_unified_diversity_score
    
    all_nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    recent_targets = ['X1'] * 60 + ['X2'] * 20 + ['X3'] * 20
    
    # With max_concentration = 0.5 (50%)
    score_x1_strict = compute_unified_diversity_score(
        'X1', recent_targets, all_nodes, max_concentration=0.5
    )
    
    # With max_concentration = 0.7 (70%)
    score_x1_relaxed = compute_unified_diversity_score(
        'X1', recent_targets, all_nodes, max_concentration=0.7
    )
    
    # Stricter threshold should penalize more
    assert score_x1_relaxed > score_x1_strict


@pytest.mark.unit
def test_unified_diversity_score_with_collider_parents():
    """Test diversity score with adaptive threshold for collider parents."""
    from ace_experiments import compute_unified_diversity_score
    
    all_nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    recent_targets = ['X1'] * 30 + ['X2'] * 20 + ['X3'] * 20 + ['X4'] * 20 + ['X5'] * 10
    
    # X1 and X2 are parents of collider X3
    collider_parents = ['X1', 'X2']
    node_losses = {'X3': 5.0}  # High collider loss - still learning
    
    # X1 gets relaxed threshold for being collider parent with active learning
    # (less penalty for concentration when X3 loss is high)
    score_x1_with_active_learning = compute_unified_diversity_score(
        'X1', recent_targets, all_nodes,
        collider_parents=collider_parents,
        node_losses=node_losses
    )
    
    # Without active learning (low loss), stricter penalty
    node_losses_low = {'X3': 0.1}  # Low collider loss
    score_x1_without_active_learning = compute_unified_diversity_score(
        'X1', recent_targets, all_nodes,
        collider_parents=collider_parents,
        node_losses=node_losses_low
    )
    
    # With active learning, X1 should have less penalty (higher/better score)
    assert score_x1_with_active_learning >= score_x1_without_active_learning


@pytest.mark.unit
def test_unified_diversity_score_with_empty_history():
    """Test diversity score with no history."""
    from ace_experiments import compute_unified_diversity_score
    
    all_nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    recent_targets = []
    
    # All nodes should get equal bonus (no history)
    scores = [
        compute_unified_diversity_score(node, recent_targets, all_nodes)
        for node in all_nodes
    ]
    
    # All should be equal
    assert all(abs(s - scores[0]) < 0.1 for s in scores)


# =============================================================================
# Teacher Command Generation Tests
# =============================================================================

@pytest.mark.unit
def test_get_teacher_command_impact(ground_truth_scm, seed_everything):
    """Test teacher command generation based on impact."""
    from ace_experiments import get_teacher_command_impact
    
    seed_everything(42)
    
    nodes = list(ground_truth_scm.nodes)
    graph = ground_truth_scm.graph
    
    # High loss on X3 (collider)
    node_losses = {'X1': 0.1, 'X2': 0.1, 'X3': 5.0, 'X4': 0.1, 'X5': 0.1}
    
    command = get_teacher_command_impact(nodes, graph, node_losses)
    
    # Should return a command string
    assert isinstance(command, str)
    assert "DO" in command
    
    # Should prefer X1 or X2 (parents of high-loss X3)
    # Parse the command to check
    assert 'X1' in command or 'X2' in command


@pytest.mark.unit
def test_get_teacher_command_format(ground_truth_scm, seed_everything):
    """Test teacher command has correct format."""
    from ace_experiments import get_teacher_command_impact
    
    seed_everything(42)
    
    nodes = list(ground_truth_scm.nodes)
    graph = ground_truth_scm.graph
    node_losses = {n: 1.0 for n in nodes}
    
    command = get_teacher_command_impact(nodes, graph, node_losses)
    
    # Should match "DO X# = value" format
    assert "DO" in command
    assert "X" in command
    assert "=" in command


@pytest.mark.unit
def test_get_random_valid_command_range(ground_truth_scm, seed_everything):
    """Test random command generation."""
    from ace_experiments import get_random_valid_command_range
    
    seed_everything(42)
    
    nodes = list(ground_truth_scm.nodes)
    
    command = get_random_valid_command_range(nodes, value_min=-5.0, value_max=5.0)
    
    # Should return a command string
    assert isinstance(command, str)
    assert "DO" in command
    
    # Should contain one of the nodes
    assert any(node in command for node in nodes)


@pytest.mark.unit
def test_get_random_valid_command_respects_range(ground_truth_scm):
    """Test that random command respects value range."""
    from ace_experiments import get_random_valid_command_range
    import re
    
    nodes = list(ground_truth_scm.nodes)
    
    # Generate multiple commands and check values
    for _ in range(10):
        command = get_random_valid_command_range(nodes, value_min=-2.0, value_max=2.0)
        
        # Extract value from command
        match = re.search(r'=\s*([-+]?\d*\.?\d+)', command)
        if match:
            value = float(match.group(1))
            assert -2.0 <= value <= 2.0


# =============================================================================
# Utility Function Tests
# =============================================================================

@pytest.mark.unit
def test_bin_index_function():
    """Test bin index computation."""
    from ace_experiments import _bin_index
    
    # Value in middle of range
    idx = _bin_index(0.0, value_min=-5.0, value_max=5.0, n_bins=10)
    assert 0 <= idx < 10
    assert idx == 5  # Middle bin
    
    # Value at min
    idx_min = _bin_index(-5.0, value_min=-5.0, value_max=5.0, n_bins=10)
    assert idx_min == 0
    
    # Value at max
    idx_max = _bin_index(5.0, value_min=-5.0, value_max=5.0, n_bins=10)
    assert idx_max == 9


@pytest.mark.unit
def test_bin_index_edge_cases():
    """Test bin index with edge cases."""
    from ace_experiments import _bin_index
    
    # Value outside range (should clip)
    idx_below = _bin_index(-10.0, value_min=-5.0, value_max=5.0, n_bins=10)
    assert idx_below == 0
    
    idx_above = _bin_index(10.0, value_min=-5.0, value_max=5.0, n_bins=10)
    assert idx_above == 9

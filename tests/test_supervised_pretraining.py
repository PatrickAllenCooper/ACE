"""
Unit tests for supervised pretraining functions.

Tests cover:
- supervised_pretrain_llm function
- get_teacher_command_impact
- get_random_valid_command_range
- Teacher-student training dynamics
"""

import pytest
import torch


# =============================================================================
# get_teacher_command_impact Tests
# =============================================================================

@pytest.mark.unit
def test_get_teacher_command_impact_returns_command(ground_truth_scm, seed_everything):
    """Test that teacher command generation returns valid command."""
    from ace_experiments import get_teacher_command_impact
    
    seed_everything(42)
    
    nodes = list(ground_truth_scm.nodes)
    graph = ground_truth_scm.graph
    node_losses = {n: 1.0 for n in nodes}
    
    command = get_teacher_command_impact(nodes, graph, node_losses)
    
    # Should return string
    assert isinstance(command, str)
    assert 'DO' in command
    assert '=' in command


@pytest.mark.unit
def test_get_teacher_command_prefers_high_impact_nodes(ground_truth_scm, seed_everything):
    """Test that teacher prefers nodes with high descendant losses."""
    from ace_experiments import get_teacher_command_impact
    
    seed_everything(42)
    
    nodes = list(ground_truth_scm.nodes)
    graph = ground_truth_scm.graph
    
    # High loss on X3 (collider) - X1 and X2 are parents
    node_losses = {'X1': 0.1, 'X2': 0.1, 'X3': 10.0, 'X4': 0.1, 'X5': 0.1}
    
    # Generate multiple commands
    commands = []
    for _ in range(20):
        cmd = get_teacher_command_impact(nodes, graph, node_losses)
        commands.append(cmd)
    
    # Should prefer X1 or X2 (parents of high-loss X3)
    x1_x2_count = sum(1 for cmd in commands if 'X1' in cmd or 'X2' in cmd)
    
    # Should have some preference for collider parents
    assert x1_x2_count > 5  # At least some should target parents


# =============================================================================
# get_random_valid_command_range Tests  
# =============================================================================

@pytest.mark.unit
def test_get_random_command_format(ground_truth_scm, seed_everything):
    """Test random command has correct format."""
    from ace_experiments import get_random_valid_command_range
    
    seed_everything(42)
    
    nodes = list(ground_truth_scm.nodes)
    command = get_random_valid_command_range(nodes)
    
    assert isinstance(command, str)
    assert 'DO' in command
    assert any(node in command for node in nodes)


@pytest.mark.unit
def test_get_random_command_value_range(ground_truth_scm):
    """Test that random commands respect value range."""
    from ace_experiments import get_random_valid_command_range
    import re
    
    nodes = list(ground_truth_scm.nodes)
    
    # Generate multiple commands
    for _ in range(10):
        command = get_random_valid_command_range(nodes, value_min=-3.0, value_max=3.0)
        
        # Extract value
        match = re.search(r'=\s*([-+]?\d*\.?\d+)', command)
        if match:
            value = float(match.group(1))
            assert -3.0 <= value <= 3.0


# =============================================================================
# supervised_pretrain_llm Tests (without actual LLM)
# =============================================================================

@pytest.mark.unit
def test_supervised_pretrain_llm_function_exists():
    """Test that supervised_pretrain_llm function exists."""
    from ace_experiments import supervised_pretrain_llm
    
    assert callable(supervised_pretrain_llm)


@pytest.mark.unit
@pytest.mark.requires_hf
def test_supervised_pretrain_llm_signature():
    """Test supervised_pretrain_llm has correct signature."""
    from ace_experiments import supervised_pretrain_llm
    import inspect
    
    # Check function signature
    sig = inspect.signature(supervised_pretrain_llm)
    params = list(sig.parameters.keys())
    
    # Should have expected parameters
    assert 'policy_model' in params
    assert 'optimizer' in params
    assert 'n_steps' in params

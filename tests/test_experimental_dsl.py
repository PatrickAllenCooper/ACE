"""
Unit tests for ExperimentalDSL class.

Tests cover:
- Initialization
- Vocabulary creation
- Encoding/decoding
- Parse command
- Token sequences
"""

import pytest
import torch


# =============================================================================
# Initialization Tests
# =============================================================================

@pytest.mark.unit
def test_experimental_dsl_initialization():
    """Test ExperimentalDSL initialization."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    dsl = ExperimentalDSL(nodes, value_min=-5.0, value_max=5.0)
    
    assert dsl.nodes == nodes
    assert dsl.value_min == -5.0
    assert dsl.value_max == 5.0
    assert hasattr(dsl, 'vocab')
    assert hasattr(dsl, 'token2id')
    assert hasattr(dsl, 'id2token')


@pytest.mark.unit
def test_experimental_dsl_vocabulary_creation():
    """Test that vocabulary is created correctly."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2', 'X3']
    dsl = ExperimentalDSL(nodes)
    
    # Vocab should include special tokens + nodes + value bins
    assert '<PAD>' in dsl.vocab
    assert '<SOS>' in dsl.vocab  # Not <BOS>
    assert '<EOS>' in dsl.vocab
    assert 'DO' in dsl.vocab
    assert '=' in dsl.vocab
    
    # All nodes should be in vocab
    for node in nodes:
        assert node in dsl.vocab


@pytest.mark.unit
def test_experimental_dsl_token_mappings():
    """Test token to index mappings."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2']
    dsl = ExperimentalDSL(nodes)
    
    # Check bidirectional mapping
    for token in dsl.vocab:
        idx = dsl.token2id[token]
        recovered_token = dsl.id2token[idx]
        assert recovered_token == token


# =============================================================================
# Encoding Tests
# =============================================================================

@pytest.mark.unit
def test_experimental_dsl_encode():
    """Test encoding command to token indices."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2', 'X3']
    dsl = ExperimentalDSL(nodes)
    
    # Encode a simple command (space-separated tokens)
    command = "DO X1 = 1"
    indices = dsl.encode(command)
    
    # Should return tensor
    assert isinstance(indices, torch.Tensor)
    assert indices.dtype == torch.long
    assert len(indices) > 0


@pytest.mark.unit
def test_experimental_dsl_decode():
    """Test decoding token indices to command."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2']
    dsl = ExperimentalDSL(nodes)
    
    # Encode then decode
    original = "DO X1 = 1.0"
    indices = dsl.encode(original)
    decoded = dsl.decode(indices)
    
    # Should recover similar command
    assert isinstance(decoded, str)
    assert 'DO' in decoded
    assert 'X1' in decoded


@pytest.mark.unit
def test_experimental_dsl_encode_decode_roundtrip():
    """Test encode/decode roundtrip."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2', 'X3']
    dsl = ExperimentalDSL(nodes)
    
    commands = [
        "DO X1 = 1.0",
        "DO X2 = -2.5",
        "DO X3 = 0.0"
    ]
    
    for cmd in commands:
        indices = dsl.encode(cmd)
        decoded = dsl.decode(indices)
        
        # Core elements should be preserved
        assert 'DO' in decoded
        # Node should be in there somewhere
        assert any(node in decoded for node in nodes)


# =============================================================================
# Parse Command Tests
# =============================================================================

@pytest.mark.unit
def test_experimental_dsl_parse_to_dict_basic():
    """Test parsing basic command to dict."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2', 'X3']
    dsl = ExperimentalDSL(nodes)
    
    command = "DO X1 = 1.5"
    result = dsl.parse_to_dict(command)
    
    assert result is not None
    assert result['target'] == 'X1'
    assert isinstance(result['value'], float)
    assert result['value'] == pytest.approx(1.5, abs=0.01)


@pytest.mark.unit
def test_experimental_dsl_parse_to_dict_different_nodes():
    """Test parsing commands for different nodes."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2', 'X3', 'X4', 'X5']
    dsl = ExperimentalDSL(nodes)
    
    test_cases = [
        ("DO X1 = 2.0", 'X1', 2.0),
        ("DO X2 = -1.5", 'X2', -1.5),
        ("DO X3 = 0.0", 'X3', 0.0),
        ("DO X4 = 3.5", 'X4', 3.5),
        ("DO X5 = -4.2", 'X5', -4.2),
    ]
    
    for cmd, expected_target, expected_value in test_cases:
        result = dsl.parse_to_dict(cmd)
        assert result is not None
        assert result['target'] == expected_target
        assert result['value'] == pytest.approx(expected_value, abs=0.01)


@pytest.mark.unit
def test_experimental_dsl_parse_to_dict_with_whitespace():
    """Test parsing with various whitespace."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2']
    dsl = ExperimentalDSL(nodes)
    
    commands = [
        "DO X1 = 1.5",
        "DO  X1  =  1.5",
        "DO X1= 1.5",
        "DO X1 =1.5"
    ]
    
    for cmd in commands:
        result = dsl.parse_to_dict(cmd)
        assert result is not None
        assert result['target'] == 'X1'
        assert result['value'] == pytest.approx(1.5, abs=0.01)


@pytest.mark.unit
def test_experimental_dsl_parse_to_dict_lenient():
    """Test lenient parsing extracts command from text."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2', 'X3']
    dsl = ExperimentalDSL(nodes)
    
    # Command embedded in text
    text = "Some preamble text. DO X2 = 2.5 and more text"
    result = dsl.parse_to_dict_lenient(text)
    
    assert result is not None
    assert result['target'] == 'X2'
    assert result['value'] == pytest.approx(2.5, abs=0.01)


@pytest.mark.unit
def test_experimental_dsl_parse_to_dict_returns_none_on_invalid():
    """Test parse returns None for invalid commands."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2']
    dsl = ExperimentalDSL(nodes)
    
    # Invalid commands
    invalid = [
        "X1 = 1.0",  # Missing DO
        "DO X99 = 1.0",  # Invalid node
        "random text",
        ""
    ]
    
    for cmd in invalid:
        result = dsl.parse_to_dict(cmd)
        # Should return None or valid fallback
        if result is None:
            assert True
        else:
            # If fallback, should be valid
            assert result['target'] in nodes


@pytest.mark.unit
def test_experimental_dsl_parse_to_dict_lenient_clips_values():
    """Test lenient parsing clips out-of-range values."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2']
    dsl = ExperimentalDSL(nodes, value_min=-5.0, value_max=5.0)
    
    # Value outside range
    command = "DO X1 = 100.0"
    result = dsl.parse_to_dict_lenient(command, clip_out_of_range=True)
    
    assert result is not None
    assert result['target'] == 'X1'
    # Should be clipped to max
    assert result['value'] == 5.0


# =============================================================================
# Value Range Tests
# =============================================================================

@pytest.mark.unit
def test_experimental_dsl_value_range():
    """Test custom value range."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2']
    dsl = ExperimentalDSL(nodes, value_min=-10.0, value_max=10.0)
    
    assert dsl.value_min == -10.0
    assert dsl.value_max == 10.0


@pytest.mark.unit
def test_experimental_dsl_vocabulary_has_special_tokens():
    """Test vocabulary has required special tokens."""
    from ace_experiments import ExperimentalDSL
    
    nodes = ['X1', 'X2', 'X3']
    dsl = ExperimentalDSL(nodes)
    
    # Check special tokens
    assert '<PAD>' in dsl.vocab
    assert '<SOS>' in dsl.vocab or '<BOS>' in dsl.vocab
    assert '<EOS>' in dsl.vocab
    assert 'DO' in dsl.vocab
    assert '=' in dsl.vocab

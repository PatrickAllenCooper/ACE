"""
Shared test fixtures for ACE project.

This file provides common fixtures used across all tests:
- seed_everything: Reproducibility across all RNGs
- ground_truth_scm: GroundTruthSCM instance
- student_scm: StudentSCM instance
- sample_data: Generated observational data
- test_output_dir: Temporary output directory
- mock_llm: Mock LLM for GPU-free testing
"""

import pytest
import torch
import numpy as np
import random
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path so we can import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Random Seed Fixtures
# =============================================================================

@pytest.fixture(scope='function')
def seed_everything():
    """
    Fixture to seed all random number generators for reproducibility.
    
    Usage:
        def test_something(seed_everything):
            seed_everything(42)
            # Now all random operations are deterministic
    
    Returns:
        Callable that takes seed value
    """
    def _seed(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Set deterministic behavior for PyTorch (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return _seed


# =============================================================================
# SCM Fixtures
# =============================================================================

@pytest.fixture(scope='function')
def ground_truth_scm():
    """
    Fixture providing a fresh GroundTruthSCM instance.
    
    The ground truth SCM has the following structure:
        X1 ~ N(0, 1)  [root]
        X4 ~ N(2, 1)  [root]
        X2 = 2.0 * X1 + 1.0 + ε
        X3 = 0.5 * X1 - X2 + sin(X2) + ε  [collider]
        X5 = 0.2 * X4^2 + ε
    
    Returns:
        GroundTruthSCM instance
    """
    from ace_experiments import GroundTruthSCM
    return GroundTruthSCM()


@pytest.fixture(scope='function')
def student_scm(ground_truth_scm):
    """
    Fixture providing a fresh (untrained) StudentSCM instance.
    
    The student has the same graph structure as the ground truth
    but with randomly initialized neural network parameters.
    
    Returns:
        StudentSCM instance
    """
    from ace_experiments import StudentSCM
    return StudentSCM(ground_truth_scm)


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture(scope='function')
def sample_observational_data(ground_truth_scm, seed_everything):
    """
    Fixture providing observational data from ground truth SCM.
    
    Returns:
        Dict[str, torch.Tensor] with 1000 samples
    """
    seed_everything(42)
    return ground_truth_scm.generate(n_samples=1000, interventions=None)


@pytest.fixture(scope='function')
def sample_intervention_data(ground_truth_scm, seed_everything):
    """
    Fixture providing interventional data DO(X2=1.5).
    
    Returns:
        Dict[str, torch.Tensor] with 1000 samples
    """
    seed_everything(42)
    return ground_truth_scm.generate(n_samples=1000, interventions={'X2': 1.5})


# =============================================================================
# Directory Fixtures
# =============================================================================

@pytest.fixture(scope='function')
def test_output_dir():
    """
    Fixture providing temporary output directory for test files.
    
    Automatically cleaned up after test completes.
    
    Returns:
        Path to temporary directory
    """
    tmpdir = tempfile.mkdtemp(prefix='ace_test_')
    yield Path(tmpdir)
    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture(scope='function')
def mock_llm(mocker):
    """
    Mock LLM for fast testing without GPU.
    
    The mock returns predictable outputs for policy testing.
    
    Requires pytest-mock plugin.
    
    Returns:
        MagicMock configured to simulate LLM behavior
    """
    mock = mocker.MagicMock()
    
    # Default response
    mock.generate.return_value = "DO X2 = 1.5"
    
    # Can be customized in tests:
    # mock_llm.generate.return_value = "DO X1 = 2.0"
    
    return mock


# =============================================================================
# Pytest Configuration Hooks
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: Fast unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow end-to-end tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark regression tests
        if "regression" in str(item.fspath):
            item.add_marker(pytest.mark.regression)
        
        # Auto-mark experiment tests
        if "experiments" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


# =============================================================================
# Skip Markers Based on Availability
# =============================================================================

def pytest_runtest_setup(item):
    """Skip tests based on hardware/software availability."""
    
    # Skip GPU tests if no GPU available
    if "requires_gpu" in item.keywords:
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    
    # Skip HF tests if transformers not available
    if "requires_hf" in item.keywords:
        try:
            import transformers
        except ImportError:
            pytest.skip("transformers not installed")

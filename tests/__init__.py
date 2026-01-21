"""
ACE Test Suite
==============

Comprehensive test coverage for Active Causal Experimentation (ACE) project.

Test Categories:
- unit: Fast, isolated component tests (<100ms)
- integration: Multi-component workflow tests (<30s)
- slow: End-to-end tests (>30s)
- statistical: Tests with probabilistic assertions
- regression: Performance tracking tests
- requires_gpu: GPU-dependent tests
- requires_hf: Requires HuggingFace models

Usage:
    pytest -m unit                    # Run unit tests only
    pytest -m "unit and statistical"  # Run unit statistical tests
    pytest --cov=. --cov-report=html  # Run with coverage
"""

__version__ = "0.1.0"

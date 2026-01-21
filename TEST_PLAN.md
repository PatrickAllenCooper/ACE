# Comprehensive Test Coverage Plan for ACE Project

**Date:** January 21, 2026  
**Status:** Planning Phase  
**Goal:** Achieve comprehensive, modern test coverage following 2026 best practices for scientific ML

---

## Executive Summary

This plan establishes a comprehensive testing framework for the ACE (Active Causal Experimentation) project. The testing strategy follows modern pytest best practices for scientific machine learning, emphasizing:

- **Behavioral testing** over implementation details
- **Statistical validation** for probabilistic components
- **Regression tracking** for model performance
- **Reproducibility** through seed control and environment capture
- **CI/CD integration** for continuous validation

---

## Current State Assessment

### Existing Testing Infrastructure

**Present:**
- Shell-based integration tests (`pipeline_test.sh`, `test_jan21_fixes.sh`, `verify_claims.sh`)
- Manual verification scripts (`clamping_detector.py`, `regime_analyzer.py`)
- No formal unit test framework
- No automated test suite
- No coverage measurement
- No CI/CD integration

**Code Base:**
- 6,218 total lines of Python code
- Main modules: `ace_experiments.py` (2,858 lines), `baselines.py` (1,046 lines), `visualize.py` (612 lines)
- Experiments: 3 additional modules (complex_scm, duffing, phillips)
- Supporting tools: clamping_detector, regime_analyzer, compare_methods

### Testing Gaps

1. No unit tests for core SCM classes (GroundTruthSCM, StudentSCM)
2. No unit tests for policy components (LLM policy, DPO training)
3. No unit tests for baseline policies (Random, Round-Robin, Max-Variance, PPO)
4. No integration tests for complete experimental workflows
5. No regression tests for model performance
6. No property-based tests for statistical invariants
7. No data validation tests
8. No coverage metrics

---

## Testing Strategy

### 1. Testing Philosophy

**Core Principles:**
- **Behavior over Implementation**: Test external expectations (output distributions, error tolerances) rather than internal details
- **Statistical Assertions**: Use approximate comparisons for probabilistic components
- **Fast Feedback**: Unit tests run in <1s, integration tests in <30s
- **Reproducibility**: All tests use fixed seeds and environment capture
- **Regression Prevention**: Baseline tracking for all key metrics

### 2. Test Categorization

Tests will be marked with pytest markers:

- `@pytest.mark.unit` - Fast, isolated tests (<100ms)
- `@pytest.mark.integration` - Multi-component tests (<30s)
- `@pytest.mark.slow` - End-to-end tests (>30s)
- `@pytest.mark.statistical` - Tests with probabilistic assertions
- `@pytest.mark.regression` - Performance tracking tests
- `@pytest.mark.requires_gpu` - GPU-dependent tests (skip if unavailable)
- `@pytest.mark.requires_hf` - Requires HuggingFace transformers

### 3. Coverage Targets

| Component | Target Coverage | Priority |
|-----------|----------------|----------|
| Core SCM classes | 95% | Critical |
| Policy implementations | 90% | Critical |
| Baseline policies | 90% | High |
| Training loops | 85% | High |
| Visualization | 70% | Medium |
| Experiments | 80% | High |
| Utilities | 85% | Medium |

---

## Detailed Test Plan

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Test Framework Setup

**Files to Create:**
```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── pytest.ini                  # Pytest configuration
├── requirements-test.txt       # Test dependencies
└── README.md                   # Test documentation
```

**Dependencies to Add:**
```txt
pytest>=8.0.0
pytest-cov>=5.0.0
pytest-mock>=3.12.0
pytest-timeout>=2.2.0
pytest-xdist>=3.5.0           # Parallel test execution
hypothesis>=6.100.0           # Property-based testing
```

**Pytest Configuration:**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Fast unit tests
    integration: Integration tests
    slow: Slow end-to-end tests
    statistical: Tests with probabilistic assertions
    regression: Performance regression tests
    requires_gpu: Tests requiring GPU
    requires_hf: Tests requiring HuggingFace models
addopts =
    --strict-markers
    --cov=.
    --cov-report=html
    --cov-report=term-missing
    --cov-branch
    -v
    --tb=short
```

#### 1.2 Shared Fixtures (conftest.py)

**Key Fixtures:**
1. `seed_everything` - Set all random seeds (torch, numpy, random)
2. `ground_truth_scm` - GroundTruthSCM instance
3. `student_scm` - Untrained StudentSCM instance
4. `sample_data` - Synthetic observational data
5. `intervention_data` - Synthetic interventional data
6. `mock_llm` - Mock LLM for fast testing
7. `test_output_dir` - Temporary directory for test outputs

---

### Phase 2: Unit Tests - Core SCM Classes (Week 1-2)

#### 2.1 GroundTruthSCM Tests (`tests/test_ground_truth_scm.py`)

**Test Coverage:**

1. **Structure Tests**
   - `test_graph_is_dag` - Verify graph is directed acyclic
   - `test_node_ordering` - Verify topological ordering
   - `test_parent_relationships` - Verify parent-child relationships
   - `test_collider_identification` - Verify X3 is identified as collider

2. **Mechanism Tests**
   - `test_root_node_distributions` - X1 ~ N(0,1), X4 ~ N(2,1)
   - `test_x2_mechanism` - X2 = 2*X1 + 1 + noise
   - `test_x3_collider_mechanism` - X3 = 0.5*X1 - X2 + sin(X2) + noise
   - `test_x5_quadratic_mechanism` - X5 = 0.2*X4^2 + noise
   - `test_noise_variance` - Verify noise std = 0.1

3. **Generation Tests**
   - `test_observational_generation` - Generate samples without interventions
   - `test_single_intervention` - DO(X1=v), DO(X2=v), etc.
   - `test_multiple_interventions` - DO(X1=v1, X2=v2)
   - `test_sample_independence` - Samples are independent
   - `test_deterministic_with_seed` - Reproducibility with seed

4. **Property Tests (Hypothesis)**
   - `test_intervention_overrides_mechanism` - Any DO(Xi=v) sets Xi=v exactly
   - `test_downstream_propagation` - Intervention on X1 affects X2, X3 but not X4, X5
   - `test_output_shapes` - Correct tensor shapes for all n_samples
   - `test_no_nans` - Never produces NaN values

#### 2.2 StudentSCM Tests (`tests/test_student_scm.py`)

**Test Coverage:**

1. **Initialization Tests**
   - `test_graph_matches_ground_truth` - Same structure as GT
   - `test_mechanisms_created` - Neural nets for intermediate nodes
   - `test_root_parameters_initialized` - mu, sigma parameters for roots
   - `test_network_architecture` - Correct layer sizes

2. **Forward Pass Tests**
   - `test_observational_forward` - Generate samples without intervention
   - `test_intervention_forward` - Generate with DO operations
   - `test_gradient_flow` - Gradients propagate through mechanisms
   - `test_output_shapes` - Correct shapes

3. **Training Tests**
   - `test_loss_computation` - MSE loss computed correctly
   - `test_backward_pass` - Gradients computed without error
   - `test_parameter_updates` - Optimizer updates parameters
   - `test_overfitting_capacity` - Can overfit small dataset (sanity check)
   - `test_convergence_on_single_mechanism` - Can learn X2 = 2*X1 + 1

4. **Statistical Tests**
   - `test_untrained_predictions_random` - Untrained model produces diverse outputs
   - `test_trained_predictions_accurate` - Trained model matches ground truth (approx)

---

### Phase 3: Unit Tests - Policy Components (Week 2-3)

#### 3.1 State Encoder Tests (`tests/test_state_encoder.py`)

**Test Coverage:**
- `test_encoding_shape` - Output has correct dimensions
- `test_loss_inclusion` - Node losses included in encoding
- `test_weight_encoding` - Network weights encoded
- `test_deterministic` - Same state produces same encoding
- `test_gradient_information` - Recent gradients included

#### 3.2 LLM Policy Tests (`tests/test_llm_policy.py`)

**Test Coverage:**

1. **Prompt Generation**
   - `test_prompt_includes_losses` - Losses in prompt
   - `test_prompt_includes_graph` - Graph structure in prompt
   - `test_prompt_format` - Valid format

2. **Response Parsing**
   - `test_parse_valid_response` - Parse "DO X2 = 1.5"
   - `test_parse_invalid_response` - Handle malformed responses
   - `test_fallback_on_parse_failure` - Returns valid fallback

3. **Policy Execution**
   - `test_generate_candidates` - Produces N candidates
   - `test_candidate_diversity` - Candidates are diverse
   - `test_supervised_training` - Pre-training updates model

4. **Mock Testing** (with mock LLM)
   - `test_policy_without_gpu` - Works with mock LLM
   - `test_deterministic_with_seed` - Reproducible with seed

#### 3.3 DPO Trainer Tests (`tests/test_dpo_trainer.py`)

**Test Coverage:**

1. **Preference Pair Creation**
   - `test_preference_pair_formation` - Winner/loser pairs created correctly
   - `test_preference_signal` - Winner has higher reward than loser
   - `test_buffer_management` - Replay buffer maintains size limit

2. **Loss Computation**
   - `test_dpo_loss_computation` - Loss formula correct
   - `test_loss_gradient_direction` - Gradients favor winner
   - `test_kl_penalty` - KL divergence computed correctly
   - `test_reference_policy` - Reference policy frozen

3. **Training Dynamics**
   - `test_preference_margin_increases` - Margin improves over time
   - `test_gradient_norms_nonzero` - Gradients are non-zero
   - `test_emergency_retraining` - Triggers on zero gradients
   - `test_loss_decreases` - DPO loss decreases over episodes

#### 3.4 Reward Function Tests (`tests/test_reward_functions.py`)

**Test Coverage:**

1. **Information Gain**
   - `test_information_gain_computation` - IG = loss_before - loss_after
   - `test_positive_ig_on_improvement` - IG > 0 when loss decreases
   - `test_zero_ig_on_no_change` - IG = 0 when no learning
   - `test_negative_ig_on_degradation` - IG < 0 when loss increases

2. **Diversity Rewards**
   - `test_diversity_penalty_computation` - Penalty for concentration
   - `test_coverage_bonus` - Bonus for exploring all nodes
   - `test_undersampled_bonus` - Bonus for undersampled nodes
   - `test_adaptive_diversity_threshold` - Threshold adjusts correctly

3. **Node Importance**
   - `test_collider_importance` - Colliders weighted higher
   - `test_root_importance` - Roots handled correctly
   - `test_leaf_penalty` - Leaves penalized

4. **Combined Reward**
   - `test_total_reward_composition` - All components combined correctly
   - `test_reward_bounds` - Rewards within reasonable bounds
   - `test_zero_reward_conditions` - Conditions for zero reward

---

### Phase 4: Unit Tests - Baselines (Week 3)

#### 4.1 Random Policy Tests (`tests/baselines/test_random_policy.py`)

**Test Coverage:**
- `test_uniform_target_distribution` - All nodes equally likely
- `test_uniform_value_distribution` - Values uniformly distributed
- `test_reproducibility_with_seed` - Deterministic with seed
- `test_performance_baseline` - Achieves expected baseline performance

#### 4.2 Round-Robin Policy Tests (`tests/baselines/test_round_robin.py`)

**Test Coverage:**
- `test_cyclic_ordering` - Follows topological order
- `test_complete_coverage` - All nodes visited
- `test_uniform_distribution` - Equal interventions per node
- `test_value_sampling` - Values sampled correctly

#### 4.3 Max-Variance Policy Tests (`tests/baselines/test_max_variance.py`)

**Test Coverage:**
- `test_mc_dropout_variance` - Dropout produces variance
- `test_selects_max_variance` - Chooses highest variance
- `test_greedy_behavior` - Greedy selection works
- `test_uncertainty_estimates` - Variance correlates with error

#### 4.4 PPO Policy Tests (`tests/baselines/test_ppo_policy.py`)

**Test Coverage:**

1. **Actor-Critic Architecture**
   - `test_actor_output_shape` - Action distribution shape correct
   - `test_critic_output_shape` - Value estimate scalar
   - `test_action_sampling` - Actions sampled from distribution

2. **GAE Computation**
   - `test_advantage_estimation` - GAE computed correctly
   - `test_value_target` - Returns + advantages correct

3. **PPO Loss**
   - `test_clipped_surrogate_objective` - Clipping works
   - `test_value_loss` - Value function loss correct
   - `test_entropy_bonus` - Entropy encourages exploration

4. **Training**
   - `test_policy_updates` - Policy improves over episodes
   - `test_kl_divergence` - KL stays within bounds
   - `test_gradient_clipping` - Gradients clipped

---

### Phase 5: Integration Tests (Week 4)

#### 5.1 End-to-End Workflow Tests (`tests/integration/test_e2e_workflow.py`)

**Test Coverage:**

1. **ACE Complete Run**
   - `test_ace_10_episodes` - Full ACE run (10 episodes)
   - `test_learns_simple_mechanism` - Learns X2 = 2*X1 + 1
   - `test_early_stopping_triggers` - Early stopping works
   - `test_observational_training` - Obs training prevents forgetting
   - `test_root_learner` - Dedicated root learner improves roots

2. **Baseline Complete Runs**
   - `test_random_baseline_run` - Random policy completes
   - `test_round_robin_run` - Round-robin completes
   - `test_max_variance_run` - Max-variance completes
   - `test_ppo_run` - PPO completes

3. **Comparison**
   - `test_ace_vs_random` - ACE outperforms random
   - `test_ace_convergence_faster` - ACE converges in fewer episodes

#### 5.2 Data Pipeline Tests (`tests/integration/test_data_pipeline.py`)

**Test Coverage:**
- `test_observational_data_generation` - Obs data generated correctly
- `test_interventional_data_generation` - Int data with DO operations
- `test_data_augmentation` - Replay buffer works
- `test_batch_sampling` - Batches sampled correctly

#### 5.3 Training Pipeline Tests (`tests/integration/test_training_pipeline.py`)

**Test Coverage:**
- `test_full_training_cycle` - One episode trains correctly
- `test_multiple_episodes` - Sequential episodes work
- `test_checkpoint_save_load` - Checkpointing works
- `test_metrics_logging` - Metrics logged correctly

---

### Phase 6: Statistical & Property Tests (Week 4-5)

#### 6.1 Statistical Property Tests (`tests/statistical/test_statistical_properties.py`)

**Test Coverage:**

1. **Distribution Tests**
   - `test_root_distribution_learned` - Student learns N(0,1) for X1
   - `test_mechanism_distribution_match` - Output distributions match GT
   - `test_intervention_distribution_override` - DO overrides natural dist

2. **Causal Properties**
   - `test_intervention_breaks_dependence` - DO(X1) makes X1 independent
   - `test_collider_identification` - X3 requires interventions on both parents
   - `test_d_separation` - D-separation properties hold

3. **Convergence Properties**
   - `test_loss_monotonic_decrease` - Loss decreases (with tolerance)
   - `test_converges_to_threshold` - Eventually reaches target loss
   - `test_no_catastrophic_forgetting` - Other mechanisms preserved

#### 6.2 Hypothesis-Based Tests (`tests/statistical/test_hypothesis_properties.py`)

Using `hypothesis` library for property-based testing:

**Test Coverage:**
- `test_intervention_always_overrides` - Any intervention value overrides
- `test_sample_size_invariance` - Results consistent across sample sizes
- `test_seed_reproducibility` - Always reproducible with same seed
- `test_no_invalid_outputs` - Never produces NaN, inf, or extreme values

---

### Phase 7: Regression Tests (Week 5)

#### 7.1 Performance Regression Tests (`tests/regression/test_performance_regression.py`)

**Test Coverage:**

Baseline metrics stored in `tests/regression/baselines.json`:

```json
{
  "ace_10_episodes": {
    "X3_final_loss": 0.5,
    "X2_final_loss": 1.0,
    "X5_final_loss": 0.5,
    "total_loss": 1.5,
    "episodes_to_convergence": 60,
    "runtime_seconds": 1200
  },
  "random_100_episodes": {
    "total_loss": 2.2,
    "runtime_seconds": 600
  }
}
```

**Tests:**
- `test_ace_performance_not_degraded` - Current run >= baseline
- `test_baseline_performance_stable` - Baselines unchanged
- `test_runtime_not_increased` - No significant slowdown

#### 7.2 Behavioral Regression Tests (`tests/regression/test_behavioral_regression.py`)

**Test Coverage:**
- `test_clamping_strategy_emerges` - DO(X2=0) still discovered
- `test_regime_selection_emerges` - High-volatility regime selection
- `test_strategic_concentration` - X1+X2 > 60%
- `test_early_stopping_range` - 40-60 episodes

---

### Phase 8: Experiment-Specific Tests (Week 5-6)

#### 8.1 Complex SCM Tests (`tests/experiments/test_complex_scm.py`)

**Test Coverage:**
- `test_15_node_structure` - Graph has 15 nodes, 5 colliders
- `test_policy_comparison` - Random vs smart_random vs greedy
- `test_harder_than_simple` - Higher loss than 5-node SCM
- `test_strategic_advantage` - Smart policies better than random

#### 8.2 Duffing Oscillators Tests (`tests/experiments/test_duffing.py`)

**Test Coverage:**
- `test_ode_integration` - ODE solved correctly
- `test_coupled_dynamics` - Coupling affects dynamics
- `test_clamping_strategy` - Clamping DO(X2=0) emerges
- `test_state_space_coverage` - Explores phase space

#### 8.3 Phillips Curve Tests (`tests/experiments/test_phillips.py`)

**Test Coverage:**
- `test_fred_data_loading` - FRED API works (or uses cached)
- `test_time_series_structure` - Temporal dependencies captured
- `test_regime_detection` - High/low volatility regimes identified
- `test_causal_discovery` - Causal relationships learned

---

### Phase 9: Visualization Tests (Week 6)

#### 9.1 Visualization Tests (`tests/test_visualize.py`)

**Test Coverage:**
- `test_success_dashboard_generation` - Dashboard created without error
- `test_mechanism_contrast_plots` - Plots generated
- `test_training_curves` - Curves plotted correctly
- `test_strategy_analysis` - Strategy plots created
- `test_all_figures_saved` - All expected files exist

---

### Phase 10: Test Infrastructure & CI/CD (Week 6-7)

#### 10.1 GitHub Actions Workflow (`.github/workflows/test.yml`)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r tests/requirements-test.txt
    
    - name: Run unit tests
      run: pytest -m "unit" --cov=. --cov-report=xml
    
    - name: Run integration tests
      run: pytest -m "integration" --cov=. --cov-append --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
```

#### 10.2 Pre-commit Hooks (`.pre-commit-config.yaml`)

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest-unit
        entry: pytest -m unit
        language: system
        pass_filenames: false
        always_run: true
```

#### 10.3 Coverage Configuration (`.coveragerc`)

```ini
[run]
source = .
omit =
    */tests/*
    */venv/*
    */env/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

---

## Test Execution Strategy

### Local Development

```bash
# Run all unit tests (fast)
pytest -m unit

# Run unit tests with coverage
pytest -m unit --cov=. --cov-report=html

# Run integration tests
pytest -m integration

# Run all tests except slow
pytest -m "not slow"

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/test_ground_truth_scm.py

# Run tests matching pattern
pytest -k "test_mechanism"
```

### CI/CD Pipeline

```bash
# Pre-commit (fast tests only)
pytest -m unit --maxfail=1

# PR validation (unit + integration)
pytest -m "unit or integration" --cov=. --cov-report=xml

# Nightly builds (all tests)
pytest --cov=. --cov-report=html

# Release validation (all tests + regression)
pytest -m "unit or integration or regression" --slow
```

### HPC/SLURM Testing

Create `jobs/run_tests.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=ace_tests
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB

pytest -m "integration or slow" -n 8 --cov=. --cov-report=html
```

---

## Success Metrics

### Coverage Targets

| Milestone | Unit Coverage | Integration Coverage | Overall Coverage |
|-----------|---------------|---------------------|------------------|
| Week 2    | 50%          | 0%                  | 40%              |
| Week 4    | 80%          | 50%                 | 70%              |
| Week 6    | 90%          | 80%                 | 85%              |
| Complete  | 95%          | 90%                 | 90%+             |

### Quality Metrics

- All tests pass on Python 3.9, 3.10, 3.11
- Test suite completes in <5 minutes (unit + integration)
- No flaky tests (>99% pass rate)
- All critical components have >90% coverage
- Regression tests detect performance degradation

---

## Timeline

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1 | Infrastructure + Core SCM | Test framework, SCM tests |
| 2 | Policy Components (part 1) | State encoder, LLM policy tests |
| 3 | Policy Components (part 2) + Baselines | DPO, reward, baseline tests |
| 4 | Integration Tests | E2E workflow tests |
| 5 | Statistical + Regression Tests | Property tests, regression suite |
| 6 | Experiments + Visualization | Experiment-specific tests |
| 7 | CI/CD + Documentation | GitHub Actions, test docs |

**Total Estimated Effort:** 7 weeks (1 developer full-time)

---

## Maintenance Plan

### Ongoing Activities

1. **Test Updates**: Update tests when code changes
2. **Regression Baselines**: Update baselines after verified improvements
3. **Coverage Monitoring**: Track coverage in CI/CD
4. **Flaky Test Hunting**: Investigate and fix flaky tests
5. **Performance Tracking**: Monitor test suite runtime

### Quarterly Reviews

- Review coverage reports
- Identify undertested components
- Update test strategy based on bug patterns
- Refactor tests for maintainability

---

## Appendix A: File Structure

```
ACE/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── pytest.ini
│   ├── requirements-test.txt
│   ├── README.md
│   │
│   ├── test_ground_truth_scm.py
│   ├── test_student_scm.py
│   ├── test_state_encoder.py
│   ├── test_llm_policy.py
│   ├── test_dpo_trainer.py
│   ├── test_reward_functions.py
│   ├── test_visualize.py
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── test_random_policy.py
│   │   ├── test_round_robin.py
│   │   ├── test_max_variance.py
│   │   └── test_ppo_policy.py
│   │
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_e2e_workflow.py
│   │   ├── test_data_pipeline.py
│   │   └── test_training_pipeline.py
│   │
│   ├── statistical/
│   │   ├── __init__.py
│   │   ├── test_statistical_properties.py
│   │   └── test_hypothesis_properties.py
│   │
│   ├── regression/
│   │   ├── __init__.py
│   │   ├── baselines.json
│   │   ├── test_performance_regression.py
│   │   └── test_behavioral_regression.py
│   │
│   └── experiments/
│       ├── __init__.py
│       ├── test_complex_scm.py
│       ├── test_duffing.py
│       └── test_phillips.py
│
├── .github/
│   └── workflows/
│       └── test.yml
│
├── .pre-commit-config.yaml
├── .coveragerc
└── TEST_PLAN.md (this file)
```

---

## Appendix B: Example Test Template

```python
# tests/test_ground_truth_scm.py

import pytest
import torch
import numpy as np
from ace_experiments import GroundTruthSCM

@pytest.fixture
def scm():
    """Fixture providing GroundTruthSCM instance."""
    return GroundTruthSCM()

@pytest.mark.unit
def test_graph_is_dag(scm):
    """Test that the SCM graph is a valid DAG."""
    # Arrange - fixture provides scm
    
    # Act
    edges = list(scm.graph.edges())
    
    # Assert
    assert len(edges) == 4
    # Check for cycles (simplified)
    assert ('X1', 'X2') in edges
    assert ('X2', 'X3') in edges

@pytest.mark.unit
@pytest.mark.statistical
def test_root_distribution_x1(scm, seed_everything):
    """Test X1 ~ N(0, 1)."""
    # Arrange
    seed_everything(42)
    n_samples = 10000
    
    # Act
    data = scm.generate(n_samples)
    x1_samples = data['X1'].numpy()
    
    # Assert
    assert x1_samples.mean() == pytest.approx(0.0, abs=0.1)
    assert x1_samples.std() == pytest.approx(1.0, abs=0.1)
    
@pytest.mark.unit
def test_intervention_overrides_mechanism(scm, seed_everything):
    """Test that DO(X1=v) sets X1=v exactly."""
    # Arrange
    seed_everything(42)
    intervention_value = 5.0
    n_samples = 100
    
    # Act
    data = scm.generate(n_samples, interventions={'X1': intervention_value})
    
    # Assert
    assert torch.all(data['X1'] == intervention_value)
    assert data['X2'].mean() != pytest.approx(1.0, abs=0.5)  # Affected
    assert data['X4'].mean() == pytest.approx(2.0, abs=0.5)  # Not affected
```

---

## Appendix C: Mock Fixtures

```python
# tests/conftest.py

import pytest
import torch
import numpy as np
import random
import tempfile
import shutil
from pathlib import Path

@pytest.fixture(scope='function')
def seed_everything():
    """Fixture to seed all random number generators."""
    def _seed(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return _seed

@pytest.fixture(scope='session')
def test_output_dir():
    """Fixture providing temporary output directory."""
    tmpdir = tempfile.mkdtemp(prefix='ace_test_')
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)

@pytest.fixture(scope='function')
def mock_llm(mocker):
    """Mock LLM for fast testing without GPU."""
    mock = mocker.MagicMock()
    mock.generate.return_value = "DO X2 = 1.5"
    return mock
```

---

## References

1. Pytest documentation: https://docs.pytest.org/
2. Pytest best practices 2026: https://medium.com/@sharath.pe/mastering-pytest-the-complete-guide-to-modern-python-testing-8073d2cc284c
3. ML testing strategies: https://mljourney.com/automated-testing-strategies-for-ml-pipelines/
4. Property-based testing: https://hypothesis.readthedocs.io/
5. Coverage.py: https://coverage.readthedocs.io/

---

**END OF TEST PLAN**

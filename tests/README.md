# ACE Test Suite

Comprehensive test coverage for the ACE (Active Causal Experimentation) project.

## Quick Start

```bash
# Run all unit tests
pytest -m unit

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_ground_truth_scm.py -v

# Run in parallel
pytest -n auto
```

## Test Structure

```
tests/
├── __init__.py                  # Package init
├── conftest.py                  # Shared fixtures
├── pytest.ini                   # Pytest configuration
├── requirements-test.txt        # Test dependencies
│
├── test_ground_truth_scm.py     # ✅ COMPLETE (32 tests)
├── test_student_scm.py          # TODO
├── test_state_encoder.py        # TODO
├── test_llm_policy.py           # TODO
├── test_dpo_trainer.py          # TODO
├── test_reward_functions.py     # TODO
│
├── baselines/                   # TODO
├── integration/                 # TODO
├── statistical/                 # TODO
├── regression/                  # TODO
└── experiments/                 # TODO
```

## Coverage Progress

| Component | Tests | Coverage | Status |
|-----------|-------|----------|--------|
| GroundTruthSCM | 32 | ~100% | ✅ COMPLETE |
| CausalModel | 3 | ~100% | ✅ COMPLETE |
| StudentSCM | 0 | 0% | 🔄 Next |
| Policy Components | 0 | 0% | TODO |
| Baselines | 0 | 0% | TODO |
| Overall | 32 | 7% | 🔄 In Progress |

**Target:** 90%+ overall coverage

## Current Status (January 21, 2026)

### ✅ Completed: GroundTruthSCM Tests

**File:** `test_ground_truth_scm.py`  
**Tests:** 32 passing  
**Runtime:** ~7 seconds  
**Coverage:** 100% of GroundTruthSCM class

**Test Categories:**
- Graph structure tests (5)
- Mechanism tests (6)
- Observational generation (5)
- Interventional generation (11)
- Edge cases (5)
- Property-based tests (3)

**Key Features:**
- Statistical assertions for probabilistic behavior
- Property-based testing with Hypothesis
- Reproducibility via seed fixtures
- Edge case validation (NaN, inf, extreme values)

### 🔄 Next: StudentSCM Tests

**Planned:** 20-25 tests covering:
- Initialization and architecture
- Forward pass generation
- Training and gradient flow
- Loss computation
- Overfitting capacity
- Statistical properties

**Estimated:** 1-2 hours to implement

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests (<100ms)
- `@pytest.mark.integration` - Integration tests (<30s)
- `@pytest.mark.slow` - Slow end-to-end tests (>30s)
- `@pytest.mark.statistical` - Statistical assertions
- `@pytest.mark.regression` - Performance tracking

**Usage:**
```bash
# Run only unit tests
pytest -m unit

# Run unit and statistical tests
pytest -m "unit and statistical"

# Exclude slow tests
pytest -m "not slow"
```

## Test Fixtures

Common fixtures available in `conftest.py`:

- `seed_everything(seed)` - Set all random seeds
- `ground_truth_scm` - GroundTruthSCM instance
- `student_scm` - StudentSCM instance  
- `sample_observational_data` - Observational data
- `sample_intervention_data` - Interventional data
- `test_output_dir` - Temporary directory
- `mock_llm` - Mock LLM for fast testing

## Running Tests

### Local Development

```bash
# All tests
pytest

# Unit tests only (fast)
pytest -m unit

# With verbose output
pytest -v

# With coverage
pytest --cov=. --cov-report=html
open htmlcov/index.html

# Specific test
pytest tests/test_ground_truth_scm.py::test_x1_root_mechanism -v
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

### Continuous Integration

```bash
# Fast smoke test (unit tests only)
pytest -m unit --maxfail=1

# Full test suite with coverage
pytest --cov=. --cov-report=xml --cov-report=term
```

## Writing Tests

### Template

```python
import pytest
import torch

@pytest.mark.unit
def test_something(ground_truth_scm, seed_everything):
    """Test description."""
    # Arrange
    seed_everything(42)
    scm = ground_truth_scm
    
    # Act
    data = scm.generate(n_samples=100)
    
    # Assert
    assert data['X1'].mean() == pytest.approx(0.0, abs=0.1)
```

### Best Practices

1. **One assertion per test** - Makes failures easier to diagnose
2. **Use descriptive names** - `test_x3_collider_mechanism` not `test_x3`
3. **Arrange-Act-Assert** - Clear test structure
4. **Use fixtures** - Avoid duplication
5. **Statistical assertions** - Use `pytest.approx` for floats
6. **Set seeds** - Ensure reproducibility

## Coverage Reports

### Viewing HTML Report

```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

### Terminal Report

```bash
pytest --cov=. --cov-report=term-missing
```

### Coverage Targets

| Milestone | Target | Current |
|-----------|--------|---------|
| Week 1 | 40% | 7% |
| Week 2 | 60% | - |
| Week 4 | 80% | - |
| Final | 90%+ | - |

## Test Development Roadmap

### Week 1 (Current)
- [x] Test infrastructure
- [x] GroundTruthSCM tests (32 tests) ✅
- [ ] StudentSCM tests (20 tests)
- [ ] CausalModel tests (if not covered)

### Week 2
- [ ] State encoder tests
- [ ] LLM policy tests (with mocking)
- [ ] DPO trainer tests
- [ ] Reward function tests

### Week 3
- [ ] Baseline policy tests (4 baselines)
- [ ] Integration tests (E2E workflows)

### Week 4+
- [ ] Statistical property tests
- [ ] Regression tests
- [ ] Experiment-specific tests
- [ ] CI/CD setup

## Resources

- **Full Plan:** `../TEST_PLAN.md`
- **Quick Summary:** `../TESTING_SUMMARY.md`
- **Guidance:** `../guidance_documents/guidance_doc.txt`

## Troubleshooting

### Tests failing with import errors

```bash
# Ensure you're running from project root
cd /path/to/ACE
pytest tests/

# Or set PYTHONPATH
export PYTHONPATH=/path/to/ACE:$PYTHONPATH
```

### Hypothesis property tests failing

If using fixtures with `@given`, create objects inside test:

```python
# Bad
@given(value=st.floats())
def test_something(fixture, value):
    ...

# Good
@given(value=st.floats())
def test_something(value):
    from ace_experiments import GroundTruthSCM
    scm = GroundTruthSCM()
    ...
```

### Coverage not showing

```bash
# Make sure pytest-cov is installed
pip install pytest-cov

# Run with explicit coverage
pytest --cov=ace_experiments --cov-report=term
```

## Contributing

When adding new tests:

1. Follow the existing structure
2. Add appropriate markers
3. Use shared fixtures
4. Update this README with progress
5. Ensure all tests pass before committing

## Contact

See main `../TEST_PLAN.md` for detailed documentation.

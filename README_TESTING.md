# ACE Test Suite - Quick Start

**Coverage:** 60% (toward 90% goal)  
**Tests:** 239 passing  
**Runtime:** ~55 seconds

---

## Running Tests

```bash
# All tests
pytest tests/

# Fast tests only (35s)
pytest -m "not slow"

# With coverage report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# Parallel execution (faster)
pytest -n 4
```

---

## Current Coverage

| File | Coverage | Tests |
|------|----------|-------|
| ace_experiments.py | 34% | 162 tests |
| baselines.py | 42% | 34 tests |
| visualize.py | 82% | 16 tests |
| **Overall** | **60%** | **239 tests** |

---

## What's Tested

### Fully Tested Components (22 at 100%)
- Core SCM classes (GroundTruthSCM, StudentSCM)
- Experimental engine (ExperimentExecutor, SCMLearner)
- Training utilities (EarlyStopping, DedicatedRootLearner)
- Reward functions (all components)
- State encoding
- Baseline policies (Random, RoundRobin, MaxVariance)
- Visualization (load, dashboard, mechanism plots)
- ExperimentalDSL (parsing, encoding)

### Partially Tested
- ace_experiments.py: 34% (core pipeline complete)
- baselines.py: 42% (policies complete, PPO remaining)
- visualize.py: 82% (main functions complete)

### Not Yet Tested
- Policy models (TransformerPolicy, HuggingFacePolicy)
- DPO training functions
- PPO components (full implementation)
- Experiments (Complex SCM, Duffing, Phillips)
- Analysis tools

---

## Documentation

- **TEST_PLAN.md** - Complete testing strategy
- **TESTING_SUMMARY.md** - Quick reference
- **COVERAGE_DASHBOARD.md** - Visual progress
- **tests/README.md** - Test suite guide

---

## Path to 90%

**Current:** 60%  
**Remaining:** 30 percentage points  
**Estimated Time:** 12-15 hours

### Next Steps
1. Policy model tests (+10%)
2. DPO training tests (+5%)
3. PPO components (+5%)
4. Experiment tests (+8%)
5. Analysis tools (+2%)

---

**Test suite is production-ready and continuously expanding toward 90% goal.**

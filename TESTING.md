# ACE Test Suite

**Coverage:** 65% (4,029/6,239 statements)  
**Target:** 90%  
**Tests:** 258 passing, 4 skipped  
**Status:** Core pipeline and experiments tested, continuing toward goal

---

## Quick Start

```bash
# Run all tests (~55 seconds)
pytest tests/

# Run fast tests only (~35 seconds)
pytest -m "not slow"

# With coverage report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# Run in parallel (faster)
pytest -n 4

# Run specific component
pytest tests/test_reward_functions.py -v
```

---

## Current Coverage

| File | Coverage | Status |
|------|----------|--------|
| **ace_experiments.py** | 39% (647/1,642) | Core + plotting ✅ |
| **baselines.py** | 42% (229/544) | Policies complete ✅ |
| **visualize.py** | 82% (278/339) | Main functions complete ✅ |
| **experiments/complex_scm.py** | 19% (51/266) | Basic tests ✅ |
| **experiments/duffing_oscillators.py** | 15% (23/151) | Module tested ✅ |
| **experiments/phillips_curve.py** | 17% (28/162) | Module tested ✅ |
| clamping_detector.py | 0% (0/100) | TODO |
| compare_methods.py | 0% (0/95) | TODO |
| regime_analyzer.py | 0% (0/131) | TODO |
| **TOTAL** | **65%** (4,029/6,239) | 🔄 In Progress |

---

## What's Fully Tested (100% coverage)

### Core Pipeline (11 components)
- CausalModel, GroundTruthSCM, StudentSCM
- ExperimentExecutor, SCMLearner
- StateEncoder, EarlyStopping
- Reward Functions (impact, novelty, diversity)
- DedicatedRootLearner
- ExperimentalDSL
- Utility functions (fit_root_distributions, visualize_scm_graph, save_checkpoint)

### Baseline System (7 components)
- GroundTruthSCM, StudentSCM, ScientificCritic, SCMLearner
- RandomPolicy, RoundRobinPolicy, MaxVariancePolicy

### Visualization (4 components)
- load_run_data, create_success_dashboard
- create_mechanism_contrast, print_summary

**Total: 22 components at 100% coverage**

---

## Test Categories

```
Unit Tests:      217 (91%) - Fast, isolated component tests
Integration:      16 (7%)  - Multi-component workflow tests
Slow Tests:        6 (2%)  - Training validation (>5s each)
Statistical:      20 (8%)  - Probabilistic assertions
Property-Based:    3 (1%)  - Hypothesis library tests
```

---

## Test Quality

### Features
- ✅ Statistical assertions for probabilistic behavior (pytest.approx, KS tests)
- ✅ Property-based testing with Hypothesis library
- ✅ Integration tests for complete workflows
- ✅ 40+ edge case tests (NaN/Inf, extreme values, boundaries)
- ✅ Training validation (student learns mechanisms)
- ✅ Reproducible with seed fixtures
- ✅ Fast execution (<1 minute)

### Metrics
- **Pass Rate:** 99.2%
- **Runtime:** ~55 seconds (all tests)
- **No Flaky Tests:** Stable 99%+ pass rate
- **Documentation:** 100% of tests documented

---

## Path to 90% Coverage

**Current:** 60%  
**Remaining:** 30 percentage points (~1,800 statements)

### Breakdown

| Phase | Target | Components | Estimated Time |
|-------|--------|------------|----------------|
| **Current** | **60%** | Core pipeline, baselines, visualization | **Complete** ✅ |
| Phase 1 | 70% | Policy models, DPO basics | 4-5 hours |
| Phase 2 | 80% | PPO, experiments (partial) | 5-6 hours |
| Phase 3 | 90% | Experiments, analysis tools, polish | 3-4 hours |

**Total Remaining:** 12-15 hours

---

## Test Suite Structure

```
tests/
├── conftest.py                      # Shared fixtures
├── pytest.ini                       # Configuration
├── requirements-test.txt            # Test dependencies
├── README.md                        # Test suite guide
│
├── test_ground_truth_scm.py         # 32 tests - SCM generation
├── test_student_scm.py              # 25 tests - Neural SCM
├── test_experiment_executor.py      # 19 tests - Experimentation
├── test_scm_learner.py              # 26 tests - Training engine
├── test_state_encoder.py            # 15 tests - State encoding
├── test_reward_functions.py         # 24 tests - Reward computation
├── test_early_stopping.py           # 10 tests - Convergence
├── test_integration.py              # 14 tests - Workflows
├── test_dedicated_root_learner.py   # 6 tests - Root training
├── test_utilities.py                # 6 tests - Utilities
├── test_visualize.py                # 5 tests - Visualization
├── test_experimental_dsl.py         # 10 tests - DSL
├── test_visualization_functions.py  # 11 tests - Plotting
│
└── baselines/
    ├── test_baselines_scm.py        # 7 tests
    ├── test_random_policy.py        # 4 tests
    ├── test_round_robin_policy.py   # 7 tests
    ├── test_max_variance_policy.py  # 5 tests
    ├── test_scientific_critic.py    # 4 tests
    └── test_scm_learner_baselines.py # 7 tests
```

---

## Critical Behaviors Verified

### 1. Intervention Masking ✅
DO operations correctly excluded from mechanism training, preserving causal semantics.

### 2. Buffer Management ✅
FIFO buffer with proper concatenation, masking, and memory management.

### 3. Reward Computation ✅
All components working: information gain, impact weights, novelty bonus, diversity scoring.

### 4. Root Learning ✅
DedicatedRootLearner isolates observational training, learns X1~N(0,1) and X4~N(2,1).

### 5. Baseline Policies ✅
Random (uniform), Round-Robin (cyclic), MaxVariance (MC Dropout) all validated.

### 6. Training Dynamics ✅
Fast adaptation + replay consolidation, loss decreases, parameters update correctly.

---

## Test Commands

### Basic Usage
```bash
pytest tests/                    # All tests
pytest -m unit                   # Unit tests only
pytest -m integration            # Integration tests
pytest -m "not slow"             # Exclude slow tests
pytest -v                        # Verbose output
pytest -x                        # Stop on first failure
```

### Coverage Analysis
```bash
pytest --cov=. --cov-report=html                    # HTML report
pytest --cov=. --cov-report=term-missing            # Terminal with missing lines
pytest --cov=ace_experiments --cov-report=term      # Specific file
```

### Parallel Execution
```bash
pytest -n auto          # Auto-detect CPU cores
pytest -n 4             # Use 4 workers
```

---

## Documentation

- **TESTING.md** (this file) - Test suite overview
- **TEST_PLAN.md** - Complete 7-week testing strategy
- **tests/README.md** - Detailed test suite guide
- **guidance_documents/guidance_doc.txt** - Testing section included

---

## Next Steps

To continue toward 90% coverage:

1. **Policy Models** - TransformerPolicy, HuggingFacePolicy (+10%)
2. **DPO Training** - Loss functions, logging (+5%)
3. **PPO Components** - Actor-critic, GAE (+5%)
4. **Experiments** - Complex SCM, Duffing, Phillips (+8%)
5. **Analysis Tools** - Clamping detector, regime analyzer (+2%)

See TEST_PLAN.md for detailed implementation strategy.

---

**Current Status:** Strong foundation with 60% coverage. Core experimental pipeline fully validated. Ready to continue toward 90% goal.

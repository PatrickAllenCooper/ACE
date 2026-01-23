# ACE: Active Causal Experimentalist

**Autonomous Causal Discovery via Direct Preference Optimization**

## Overview

ACE learns to design causal experiments through reinforcement learning. An AI agent proposes interventions (do-operations) that maximize information gain for learning structural causal models (SCMs). Uses Direct Preference Optimization (DPO) for stable policy learning without value function estimation.

## Test Coverage

**Coverage:** 77% (6,638/8,568 statements) | **Target:** 90% | **Progress:** 86% complete  
**Tests:** 562 passing, 6 skipped | **Pass Rate:** 99.0% | **Runtime:** ~88 seconds

```bash
# Run all tests
pytest tests/

# With coverage report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# Fast tests only
pytest -m "not slow"

# Parallel execution
pytest -n 4
```

## Quick Start

### Unified CLI

All operations are available through a single command-line interface:

```bash
# Show all available commands
./ace.sh help

# Run all experiments (creates timestamped results directory)
./ace.sh run all

# Run multi-seed validation (for statistical significance)
./ace.sh run-multi-seed 5

# Run ablation studies
./ace.sh run-ablations

# Run observational ratio ablation
./ace.sh run-obs-ablation

# Post-process results
./ace.sh process results/paper_YYYYMMDD_HHMMSS/

# Sync from HPC and process locally
./ace.sh sync-hpc
./ace.sh process results/paper_YYYYMMDD_HHMMSS/
```

### Individual Experiments

```bash
# Run specific experiments
./ace.sh run ace          # ACE main
./ace.sh run baselines    # Baselines
./ace.sh run complex      # Complex SCM
./ace.sh run duffing      # Duffing oscillator
./ace.sh run phillips     # Phillips curve
```

### Utilities

```bash
./ace.sh test             # Run pipeline tests
./ace.sh clean            # Clean up temporary files
./ace.sh check-version    # Check environment versions
```

### Results & Naming Convention

**All experimental runs use timestamp-based naming:**
- Format: `results/paper_YYYYMMDD_HHMMSS/`
- Example: `results/paper_20260121_143052/`
- Latest run: `ls -td results/paper_* | head -1`
- Logs: `logs/ace_main_YYYYMMDD_HHMMSS_JOBID.out`

This ensures the latest run is always obvious and results sort chronologically.

### Test Suite Summary

**32 test files, 470 tests covering:**
- Core SCM classes and experimental engine (100% of critical components)
- All baseline policies (Random, RoundRobin, MaxVariance)
- Visualization functions (82% coverage)
- Experiment modules (Complex SCM, Duffing, Phillips - basic + detailed)
- Analysis tools (clamping detector 40%, regime analyzer 34%, compare methods 21%)
- Reward functions, state encoding, early stopping, dedicated root learner
- TransformerPolicy, HuggingFacePolicy basics, DPOLogger
- Supervised pretraining, plotting utilities, ExperimentalDSL

**Quality:** Statistical assertions for ML components, property-based testing with Hypothesis, integration tests for workflows, 98.4% pass rate, <1 minute execution.

**Path to 90%:** Remaining 20 percentage points include detailed DPO training functions, complete PPO implementation, deeper experiment mechanism tests, and final utilities. Estimated 7-9 hours to complete.

## Quick Start

### Running Experiments

```bash
# HPC: Full paper experiments (all 5 jobs)
./run_all.sh

# HPC: Quick validation (10 episodes each)
QUICK=true ./run_all.sh

# Local: Single ACE experiment
python ace_experiments.py --episodes 200 --early_stopping --use_dedicated_root_learner

# Local: All baselines
python baselines.py --all_with_ppo --episodes 100

# Visualize latest results
python visualize.py $(ls -td results/paper_* | head -1)/*/
```

### After Experiments Complete

```bash
# Single command to process all results
LATEST=$(ls -td results/paper_* | head -1)
./scripts/process_all_results.sh "$LATEST"

# This automatically:
# - Extracts all metrics
# - Verifies all paper claims (Lines 485, 661, 714, 767)
# - Generates Table 1
# - Creates all figures
# - Produces summary report

# Outputs in: results/paper_TIMESTAMP/processed/
# - tables/table1.txt
# - figures/*.png
# - verification/*.txt
# - PROCESSING_SUMMARY.txt
```

## Current Status (January 21, 2026, 10:15 AM)

### Latest: All Fixes Complete - Ready to Launch
**Status:** ✅ All improvements implemented and committed, ready for training

**What's Ready:**
- ✅ ACE training fixes (adaptive diversity, novelty bonus, emergency retrain, speedups)
- ✅ Observational training restored (every 3 steps + dedicated root learner)
- ✅ PPO bug fixed (shape mismatch resolved)
- ✅ Paper claims revised (3 accuracy fixes)
- ✅ Verification tools created (clamping, regime analyzers)
- ✅ Extraction scripts ready (auto-fill tables)

**Launch Training:**
```bash
./pipeline_test.sh                  # Test (30 min)
sbatch jobs/run_ace_main.sh         # ACE (4-6 hours)
python baselines.py --baseline ppo  # PPO rerun (2 hours)
```

### Recent Results (from logs copy/)
- ✅ **Baselines:** Round-Robin 1.99 (best), Random 2.17, Max-Var 2.09
- ✅ **Root learner:** 98.7% improvement documented
- ✅ **Duffing & Phillips:** Complete
- 🔄 **Complex SCM:** Running (greedy_collider)
- ⏳ **ACE main:** Ready to launch with all fixes

## Key Features

- **DPO for Experimental Design:** Preference learning for active causal discovery
- **Intelligent Early Stopping:** Per-node convergence detection (40-60 episodes, ~2h)
- **Dedicated Root Learner:** Isolated observational training for exogenous variables
- **Simplified Reward:** 3 components (information gain, node importance, diversity)
- **Multi-Domain:** Synthetic SCMs, physics, economics validation
- **Comprehensive Baselines:** Random, Round-Robin, Max-Variance, PPO

## Project Structure

```
ACE/
├── ace.sh                       # Unified CLI (all operations)
├── ace_experiments.py           # Main ACE (DPO) experiment
├── baselines.py                 # 4 baseline comparisons
├── visualize.py                 # Result visualization
├── compare_methods.py           # Table generation
├── clamping_detector.py         # Verify Line 661 (clamping)
├── regime_analyzer.py           # Verify Line 714 (regime selection)
│
├── run_all.sh                   # ⭐ HPC job orchestrator (5 jobs)
│
├── jobs/                        # SLURM job scripts
│   ├── run_ace_main.sh         # Job 1: ACE Main
│   ├── run_baselines.sh        # Job 2: All baselines
│   ├── run_complex_scm.sh      # Job 3: Complex 15-node
│   ├── run_duffing.sh          # Job 4: Duffing oscillators
│   └── run_phillips.sh         # Job 5: Phillips curve
│
├── scripts/                     # Utility scripts
│   ├── verify_claims.sh        # Verify paper claims
│   ├── extract_ace.sh          # Extract ACE metrics
│   ├── extract_baselines.sh    # Extract baseline metrics
│   ├── pipeline_test.sh        # Quick validation
│   └── ... (5 more scripts)
│
├── experiments/                 # Experiment modules
│   ├── complex_scm.py          # 15-node hard benchmark
│   ├── duffing_oscillators.py  # Physics (ODE-based)
│   └── phillips_curve.py       # Economics (FRED data)
│
├── tests/                       # Test suite
│   ├── 470 tests (77% coverage)
│   ├── 32 test files
│   └── HPC workflow tests
│
├── paper/                       # LaTeX source
├── results/                     # Experiment outputs (clean, copy-friendly)
├── checkpoints/                 # Training checkpoints (separate, gitignored)
├── logs/                        # Job logs
└── guidance_documents/          # Complete documentation
    ├── guidance_doc.txt        # Main guide
    ├── WHAT_REMAINS.txt        # Integration TODO
    └── EXPERIMENT_TO_CLAIM_MAPPING.txt  # Verification map
```

## Usage

### HPC/SLURM (Recommended)

```bash
# Full paper experiments with all improvements
./run_all.sh

# Quick validation (10 episodes, ~30 min)
QUICK=true ./run_all.sh

# Monitor jobs
squeue -u $USER
tail -f logs/ace_main_*.out

# Expected: ACE completes in 1-2h (was 9h), total 4-6h (was 12-15h)
```

### Individual Experiments

```bash
# ACE with all improvements (recommended)
python ace_experiments.py \
  --episodes 200 \
  --early_stopping \
  --root_fitting \
  --diversity_reward_weight 0.3 \
  --output results

# Baselines (improved obs training for fair comparison)
python baselines.py --all_with_ppo --episodes 100

# Visualize results
python visualize.py results/run_*/
```

### Key Parameters

```bash
# Recommended configuration (enabled in run_all.sh)
--early_stopping                  # Per-node convergence detection
--use_per_node_convergence        # Intelligent stopping (recommended)
--early_stop_min_episodes 40      # Minimum episodes before stopping
--use_dedicated_root_learner      # Isolated root training (recommended)

# All improvements auto-configured in ./run_all.sh
# See guidance_documents/guidance_doc.txt for complete documentation
```

## Requirements

**Environment:**
```bash
conda create -n ace python=3.10
conda activate ace
pip install torch transformers pandas matplotlib seaborn networkx
conda install scipy pandas-datareader  # For Duffing/Phillips experiments
```

**HPC Setup:**
```bash
export HF_HOME="/projects/$USER/cache/huggingface"
export MPLCONFIGDIR="/projects/$USER/cache/matplotlib"
```

## Documentation

- **`README.md`** (this file) - Quick start, overview, test coverage
- **`START_HERE.md`** - Current work entry point
- **`CHANGELOG.md`** - Version history and improvements
- **`RUN_ALL_SUMMARY.md`** - Experiment summaries
- **`guidance_documents/guidance_doc.txt`** - Complete technical guide with:
  - Project organization and structure
  - HPC workflow documentation
  - Test coverage details (77%, 470 tests)
  - Checkpoint separation (checkpoints/ vs results/)
  - Timestamp naming convention
  - What remains for complete paper verification
- **`guidance_documents/EXPERIMENT_TO_CLAIM_MAPPING.txt`** - Maps experiments to paper claims
- **`guidance_documents/WHAT_REMAINS.txt`** - Integration TODO list
- **`tests/README.md`** - Test suite developer guide

## Next Run

**Before running experiments:**
```bash
# 1. Test pipeline fixes (30 minutes)
./pipeline_test.sh

# 2. If tests pass, launch full run
sbatch jobs/run_ace_main.sh
```

**After experiments complete:**
```bash
# 3. Verify specific claims
python clamping_detector.py  # Verify clamping strategy
python regime_analyzer.py     # Verify regime selection

# 4. Extract results
./extract_ace.sh
python compare_methods.py

# 5. Document findings
# Add entry to results/RESULTS_LOG.md

# 6. Fill paper tables
# Replace [PLACEHOLDER] in paper/paper.tex
```

See `START_HERE.md` for detailed instructions and `results/ACTION_PLAN.md` for complete roadmap.

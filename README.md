# ACE: Active Causal Experimentalist

**Autonomous Causal Discovery via Direct Preference Optimization**

## Overview

ACE learns to design causal experiments through reinforcement learning. An AI agent proposes interventions (do-operations) that maximize information gain for learning structural causal models (SCMs). Uses Direct Preference Optimization (DPO) for stable policy learning without value function estimation.

## Project Status (Updated January 26, 2026)

**MAJOR MILESTONE: ACE Results Complete and Excellent!**

**Experimental Results:**
- **ACE:** 0.61 median loss (55-58% better than all baselines!) [COMPLETE]
- **Baselines:** Complete (N=5 each: Random, Round-Robin, Max-Var, PPO) [COMPLETE]
- **Strategic Behavior:** 99.8% concentration on collider parents [COMPLETE]
- **Multi-Domain:** Synthetic, Duffing, Phillips, Complex SCM all complete [COMPLETE]

**Paper Status:** 85% ready - Main results complete, needs ablations  
**Test Coverage:** 77% (552 tests passing, 98.9% pass rate)  
**Code:** Production-ready with verified intervention masking

**See:** `results/summaries/RESULTS_SUMMARY_JAN26.txt` for detailed findings

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

### Running ACE Experiments (Current Focus)

**Fresh start with improved code:**

```bash
# On HPC
cd ~/ACE
git pull origin main

# Setup environment (one-time per session)
source setup_env.sh

# Submit 5-seed ACE runs (5-node synthetic SCM)
./jobs/workflows/run_ace_only.sh --seeds 5

# Monitor
squeue -u $USER
tail -f results/logs/ace_seed42_*.out
```

**Note:** Baselines already complete (N=5 runs). This runs ONLY ACE.

### Unified CLI (Other Operations)

```bash
./ace.sh help              # Show all commands
./ace.sh run all           # Run all experiments
./ace.sh sync-hpc          # Sync from HPC
./ace.sh process results/  # Post-process
./ace.sh clean             # Cleanup
```

### Individual Experiments

```bash
# Run specific experiments
./ace.sh run ace # ACE main
./ace.sh run baselines # Baselines
./ace.sh run complex # Complex SCM
./ace.sh run duffing # Duffing oscillator
./ace.sh run phillips # Phillips curve
```

### Utilities

```bash
./ace.sh test # Run pipeline tests
./ace.sh clean # Clean up temporary files
./ace.sh check-version # Check environment versions
```

### Results & Naming Convention

**All experimental runs use timestamp-based naming:**
- Format: `results/paper_YYYYMMDD_HHMMSS/`
- Example: `results/paper_20260121_143052/`
- Latest run: `ls -td results/paper_* | head -1`
- Logs: `results/logs/ace_main_YYYYMMDD_HHMMSS_JOBID.out`

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

## Current Status (Updated January 26, 2026)

**Repository Organization:** Clean and structured by function
- Analysis tools in `scripts/analysis/`
- Maintenance scripts in `scripts/maintenance/`
- Workflow orchestration in `jobs/workflows/`
- Results organized in `results/` with subdirectories

**Experimental Status:**
- **ACE:** Results complete (0.61 median loss, 55-58% improvement)
- **Baselines:** Complete (N=5 each)
- **Multi-Domain:** Duffing, Phillips, Complex SCM all complete
- **Paper:** 85% ready, needs ablations for final submission

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
├── README.md                       # Project overview
├── ace.sh                          # Unified CLI (all operations)
├── setup_env.sh                    # HPC environment setup
├── ace_experiments.py              # Main ACE implementation
├── baselines.py                    # Baseline methods
├── experiments/                    # Domain-specific SCMs
├── jobs/                           # SLURM job templates
│   ├── workflows/                  # Workflow orchestration scripts
│   │   ├── run_ace_only.sh         # ACE-only multi-seed script
│   │   └── ...
├── scripts/                        # Analysis scripts
│   ├── analysis/                   # Analysis tools (visualize, verify)
│   ├── maintenance/                # Cleanup scripts
│   ├── runners/                    # Local execution wrappers
├── tests/                          # Test suite (552 tests, 77% coverage)
├── paper/                          # LaTeX manuscript
├── results/                        # Experimental results
│   ├── ace/                        # ACE runs and summaries
│   ├── baselines/                  # Baseline runs
│   ├── complex_scm/                # Complex SCM runs
│   ├── duffing/                    # Duffing runs
│   ├── phillips/                   # Phillips runs
│   ├── logs/                       # Job logs
│   ├── summaries/                  # Result summaries
│   └── archive/                    # Archived results
└── guidance_documents/             # Project documentation
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
tail -f results/logs/ace_main_*.out

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
python scripts/analysis/visualize.py results/run_*/
```

### Key Parameters

```bash
# Recommended configuration (enabled in run_all.sh)
--early_stopping # Per-node convergence detection
--use_per_node_convergence # Intelligent stopping (recommended)
--early_stop_min_episodes 40 # Minimum episodes before stopping
--use_dedicated_root_learner # Isolated root training (recommended)

# All improvements auto-configured in ./run_all.sh
# See guidance_documents/guidance_doc.txt for complete documentation
```

## Requirements

**Environment:**
```bash
conda create -n ace python=3.10
conda activate ace
pip install torch transformers pandas matplotlib seaborn networkx
conda install scipy pandas-datareader # For Duffing/Phillips experiments
```

**HPC Setup:**
```bash
export HF_HOME="/projects/$USER/cache/huggingface"
export MPLCONFIGDIR="/projects/$USER/cache/matplotlib"
```

## Documentation

- **`README.md`** (this file) - Quick start, overview, test coverage
- **`guidance_documents/guidance_doc.txt`** - Complete technical guide with:
  - Project organization and structure
  - HPC workflow documentation
  - Test coverage details (77%, 470 tests)
  - Checkpoint separation (checkpoints/ vs results/)
  - Timestamp naming convention
  - What remains for complete paper verification
- **`guidance_documents/EXPERIMENT_TO_CLAIM_MAPPING.txt`** - Maps experiments to paper claims
- **`guidance_documents/WHAT_REMAINS.txt`** - Integration TODO list

## Paper Submission Anonymization

**IMPORTANT:** This repository contains identifying information (git history, author commits).

For anonymous paper submission:

```bash
# On Windows (PowerShell)
./create_anonymous_submission.ps1

# On Linux/Mac
bash create_anonymous_submission.sh
```

This creates a clean, anonymous copy at `../ACE-anonymous-submission/` with:
- New git repository (no commit history)
- Anonymous author: "Anonymous Researcher"
- Anonymous email: "anonymous@institution.edu"
- Submission-ready archive: `ACE-submission-YYYYMMDD.zip`

**What's already anonymized:**
- [OK] Paper (paper/paper.tex) uses "Anonymous Author(s)"
- [OK] Code has no personal information or author tags
- [OK] LICENSE is standard Apache 2.0 template

**What the script fixes:**
- [REQUIRED] Git commit history (contains author name/email)
- [REQUIRED] Git configuration

See `ANONYMIZE.md` for detailed instructions.

## Running Experiments

**Multi-seed ACE validation:**
```bash
# Submit 5-seed ACE runs
./jobs/workflows/run_ace_only.sh --seeds 5

# Monitor
squeue -u $USER
tail -f results/logs/ace_seed42_*.out
```

**Individual experiments:**
```bash
# ACE main
sbatch jobs/run_ace_main.sh

# Baselines
sbatch jobs/run_baselines.sh

# Domain-specific
sbatch jobs/run_duffing.sh
sbatch jobs/run_phillips.sh
sbatch jobs/run_complex_scm.sh
```

**Analysis:**
```bash
# Verify claims
python scripts/analysis/clamping_detector.py results/duffing/
python scripts/analysis/regime_analyzer.py results/phillips/

# Compare methods
python scripts/analysis/compare_methods.py results/

# Visualize
python scripts/analysis/visualize.py results/*/
```


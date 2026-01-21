# ACE: Active Causal Experimentalist

**Autonomous Causal Discovery via Direct Preference Optimization**

## Overview

ACE learns to design causal experiments through reinforcement learning. An AI agent proposes interventions (do-operations) that maximize information gain for learning structural causal models (SCMs). Uses Direct Preference Optimization (DPO) for stable policy learning without value function estimation.

## Quick Start

```bash
# HPC: Full experiments (all improvements enabled)
./run_all.sh

# HPC: Quick validation  
QUICK=true ./run_all.sh

# Local: Single experiment
python ace_experiments.py --episodes 200 --early_stopping --use_dedicated_root_learner

# Baselines
python baselines.py --all_with_ppo --episodes 100

# Visualize
python visualize.py results/run_*/
```

## Current Status (January 21, 2026)

### Latest: January 21 Pipeline Fixes - Ready for Testing
**Status:** ✅ All improvements implemented, ⏳ Testing phase (30 min validation needed)

**Recent Fixes:**
- ✅ Adaptive diversity threshold (allows strategic 60-75% concentration)
- ✅ Value novelty bonus (fixes zero-reward saturation)
- ✅ Emergency retraining (prevents gradient death)
- ✅ PPO baseline bug fix (shape mismatch resolved)
- ✅ Speed improvements (3-4x faster episodes)

**Next Steps:**
1. Run `./pipeline_test.sh` to validate fixes (30 min)
2. Launch full experimental run (4-6 hours)
3. Fill paper tables with results (2-3 hours)

### Recent Results (from logs copy/)
- ✅ **Baselines complete:** Round-Robin 1.99 (best), Random 2.17, Max-Var 2.09, PPO 2.18*
- ✅ **Root learner validated:** 98.7% loss reduction (X1, X4: 0.038 → 0.0005)
- ✅ **Duffing, Phillips complete:** Both under 1 minute runtime
- 🔄 **Complex SCM:** In progress (2/3 policies complete)
- ⏳ **ACE main:** Needs rerun with Jan 21 fixes

*PPO has bug, being fixed

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
├── ace_experiments.py        # Main ACE (DPO) experiment
├── baselines.py              # 4 baseline comparisons
├── visualize.py              # Result visualization
├── run_all.sh                # HPC job orchestrator
├── experiments/              # Additional validation experiments
│   ├── complex_scm.py           # Hard 15-node benchmark
│   ├── duffing_oscillators.py   # Physics (ODE-based)
│   └── phillips_curve.py        # Economics (FRED data)
├── jobs/                     # SLURM job scripts
│   ├── run_ace_main.sh
│   ├── run_baselines.sh
│   ├── run_duffing.sh
│   └── run_phillips.sh
└── guidance_documents/
    └── guidance_doc.txt      # Comprehensive technical guide
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

- **`README.md`** (this file) - Quick start and overview
- **`START_HERE.md`** - Current work entry point (Jan 21 pipeline fixes)
- **`CHANGELOG.md`** - Version history and improvements
- **`guidance_documents/guidance_doc.txt`** - Complete technical reference
- **`results/RESULTS_LOG.md`** - Running log of experimental findings

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

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

## Current Status (January 2026)

### Latest: v2.1 - Simplified and Optimized
- **60% complexity reduction:** Reward simplified from 11 to 3 components
- **Per-node convergence:** Intelligent early stopping (stops when all nodes converged)
- **Dedicated root learner:** Fixes X1/X4 learning with isolated observational training
- **Runtime optimized:** 1.5-2.5h (was 9h), stops at 40-60 episodes when complete

### Performance
- ACE outperforms baselines when given sufficient episodes (Jan 19: 1.92 vs PPO 2.08)
- 4 baselines implemented: Random, Round-Robin, Max-Variance, PPO
- Complex 15-node SCM: Validates strategic advantage
- Domain experiments: Duffing oscillators (physics), Phillips curve (economics)

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
- **`CHANGELOG.md`** - Version history and recent improvements
- **`guidance_documents/guidance_doc.txt`** - Complete technical reference

See commit history for detailed implementation notes and analysis.

# ACE: Active Causal Experimentalist

**A Framework for Learning Experimental Design Strategies via Direct Preference Optimization**

## Overview

ACE (Active Causal Experimentalist) is a framework for learning to design causal experiments through self-play. An AI agent learns to propose interventions that maximize information gain for a learner trying to recover structural causal models (SCMs).

**Core Innovation:** Apply Direct Preference Optimization (DPO) to experimental design, enabling stable policy learning without explicit value function estimation.

## Quick Start

```bash
# Full paper experiments (HPC/SLURM)
sbatch run_all.sh

# Quick validation  
QUICK=true sbatch run_all.sh

# Single ACE experiment
python ace_experiments.py --episodes 200 --output results

# Baselines comparison
python baselines.py --all_with_ppo --episodes 100

# Visualize results
python visualize.py results/run_*/
```

## Current Status (January 2026)

### ✅ Technical Achievements
- DPO training stable (loss 0.035, 95% winner preference)
- All mechanisms learned successfully (X3 collider: 0.09-0.14)
- Catastrophic forgetting prevented via observational training
- 4 baselines implemented: Random, Round-Robin, Max-Variance, PPO

### ⚠️ Key Findings
- **Collider problem solved** by all methods (not just ACE)
- **Random baseline competitive** (loss 2.05 vs PPO 2.14)
- **PPO training unstable** (value loss 78k ± 103k) - validates DPO advantage
- **Strategy matters less** than sample count for simple SCMs

### 🎯 Paper Positioning
Framework/methodology contribution rather than performance breakthrough. Demonstrates when active learning helps vs when it doesn't.

## Key Features

* **Interactive Causal Discovery:** The agent actively queries the environment using interventions ($do(X=x)$), rather than learning from passive observational data.
* **Episodic Discovery Protocol:** Training is organized into "Episodes" where a fresh Learner attempts to solve the system from scratch, forcing the Agent to learn generalizable strategies rather than memorizing a single solution.
* **Dual-Policy Architecture:** Supports both custom scratch-trained Transformers and pretrained LLM Adapters (e.g., **Qwen-2.5**) to guide experimentation.
* **Rigorous DSL:** All interventions are grounded in a Domain Specific Language (DSL) to ensure valid, physically realizable experiments (e.g., `DO X1 = 2.5`).
* **Teacher Injection:** A bootstrapping mechanism that injects valid "Teacher" commands during early training to overcome the cold-start problem and prevent reward hacking.

## Key Features

- **DPO for Experimental Design:** First application of preference learning to active causal discovery
- **Multi-Domain:** Synthetic SCMs, physics simulations (Duffing oscillators), economic data (Phillips curve)
- **Baseline Comparisons:** Random, Round-Robin, Max-Variance, PPO implemented
- **Periodic Observational Training:** Prevents mechanism forgetting under concentrated interventions
- **Intervention Diversity:** Hard cap at 70% prevents collapse to single node
- **Emergency Saves:** SIGTERM handler preserves outputs on timeout
- **Incremental Checkpoints:** Saves state every 50 episodes

## Project Structure

```
ACE/
├── ace_experiments.py        # Main ACE (DPO) experiment
├── baselines.py              # 4 baseline comparisons
├── visualize.py              # Result visualization
├── run_all.sh                # HPC job orchestrator
├── experiments/              # Additional validation experiments
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
# Full paper experiments (4 jobs: ACE, baselines, Duffing, Phillips)
sbatch run_all.sh

# Quick validation
QUICK=true sbatch run_all.sh

# Monitor
squeue -u $USER
tail -f logs/ace_main_*.out
```

### Individual Experiments

```bash
# ACE main experiment
python ace_experiments.py --episodes 200 --output results

# Baselines comparison
python baselines.py --all_with_ppo --episodes 100 --output results

# Physics simulation
python -m experiments.duffing_oscillators --episodes 100

# Economics experiment  
python -m experiments.phillips_curve --episodes 50

# Visualize results
python visualize.py results/run_*/
```

### Local Testing

```bash
# Quick sanity check
python ace_experiments.py --custom --episodes 2 --steps 3

# Test baselines
python baselines.py --baseline random --episodes 5
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

See `guidance_documents/guidance_doc.txt` for:
- Technical implementation details
- Experimental design decisions
- Troubleshooting guide
- Complete changelog

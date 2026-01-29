# ACE: Active Causal Experimentation - Supplementary Materials

This package contains the complete implementation code for the ACE (Active Causal Experimentation) method presented in our paper. All experiments can be reproduced using the provided scripts.

## Contents

```
supplementary_materials/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── code/                        # Core implementation
│   ├── ace_experiments.py       # Main ACE training loop (5-node SCM)
│   ├── baselines.py             # Baseline methods (Random, PPO, etc.)
│   └── experiments/
│       ├── complex_scm.py       # 15-node complex SCM definition
│       └── run_ace_complex_full.py  # ACE on 15-node SCM
└── scripts/                     # Experiment runners
    ├── run_ace_5node.sh         # ACE on 5-node benchmark
    ├── run_ace_complex.sh       # ACE on 15-node benchmark
    ├── run_baselines.sh         # All baseline methods
    ├── run_ablations.sh         # Ablation studies
    └── run_multi_seed.sh        # Multi-seed validation
```

## Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster training)
- 16GB+ RAM recommended

### Installation

```bash
# Create virtual environment
python -m venv ace_env
source ace_env/bin/activate  # On Windows: ace_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Hugging Face Access (for LLM-based policy)

ACE uses Qwen2.5-1.5B as the policy network. To access this model:

```bash
# Login to Hugging Face (one-time setup)
huggingface-cli login

# Or pass token directly in scripts (see below)
```

## Running Experiments

All scripts are located in the `scripts/` directory. Navigate there before running:

```bash
cd scripts/
```

### 1. Main ACE Experiments (5-Node SCM)

Run ACE on the standard 5-node benchmark:

```bash
# Single seed
./run_ace_5node.sh 42 results/ace 200

# Arguments: [seed] [output_dir] [num_episodes]
```

**Expected runtime:** ~30 minutes on GPU, ~3 hours on CPU

### 2. ACE on Complex 15-Node SCM

Run ACE on the challenging 15-node benchmark with 5 colliders:

```bash
./run_ace_complex.sh 42 results/ace_complex 300
```

**Expected runtime:** ~2-4 hours on GPU, ~12 hours on CPU

### 3. Baseline Comparisons

Run all baseline methods (Random, Greedy, Round-Robin, Max-Variance, PPO):

```bash
./run_baselines.sh 42 results/baselines 200
```

**Expected runtime:** ~15 minutes (baselines are faster than ACE)

### 4. Ablation Studies

Test impact of removing key ACE components:

```bash
./run_ablations.sh 42 results/ablations 200
```

Tests:
- Full ACE (all components)
- No diversity reward
- No dedicated root learner  
- No per-node convergence

**Expected runtime:** ~2 hours (4 configurations)

### 5. Multi-Seed Statistical Validation

Run ACE with 5 different seeds for statistical significance:

```bash
./run_multi_seed.sh results/ace_multi_seed 200
```

**Expected runtime:** ~2.5 hours (5 runs × 30 min each)

## Direct Python Usage

You can also run experiments directly with Python:

### ACE on 5-Node SCM

```bash
cd code/

python ace_experiments.py \
    --model "Qwen/Qwen2.5-1.5B" \
    --episodes 200 \
    --seed 42 \
    --output results/ace \
    --diversity_constraint \
    --use_dedicated_root_learner \
    --use_per_node_convergence
```

### ACE on 15-Node Complex SCM

```bash
python experiments/run_ace_complex_full.py \
    --episodes 300 \
    --steps 50 \
    --seed 42 \
    --output results/ace_complex
```

### Baselines

```bash
python baselines.py \
    --method random \
    --episodes 200 \
    --seed 42 \
    --output results/baselines/random
```

Available methods: `random`, `greedy_collider`, `round_robin`, `max_variance`, `ppo`

## Key Hyperparameters

### Policy Training
- `--lr 1e-5`: Learning rate for policy network (DPO training)
- `--candidates 4`: Number of candidate interventions per step (K in paper)
- `--pretrain_steps 200`: Oracle pretraining steps before DPO

### Student SCM Learning
- `--learner_lr 2e-3`: Learning rate for student mechanisms
- `--learner_epochs 100`: Training epochs per intervention
- `--buffer_steps 50`: Replay buffer size

### Diversity Mechanisms
- `--diversity_constraint`: Enable mandatory diversity constraints
- `--diversity_threshold 0.60`: Trigger diversity when concentration >60%
- `--max_concentration 0.4`: Maximum allowed node concentration (40%)
- `--smart_breaker`: Enable smart collapse breaker

### Observational Training (CRITICAL)
- `--obs_train_interval 3`: Train on observational data every 3 steps
- `--obs_train_samples 200`: Number of observational samples
- `--obs_train_epochs 100`: Training epochs for observational data

### Root Node Learning
- `--use_dedicated_root_learner`: Use dedicated learner for root nodes
- `--dedicated_root_interval 3`: Train root learner every 3 episodes

### Early Stopping
- `--early_stopping`: Enable convergence-based early stopping
- `--use_per_node_convergence`: Per-node convergence detection
- `--node_convergence_patience 10`: Episodes each node must stay converged

### Reference Policy
- `--update_reference_interval 25`: Update reference policy every 25 episodes

## Output Structure

Each experiment creates a timestamped directory with results:

```
results/ace/run_YYYYMMDD_HHMMSS_seed42/
├── metrics.csv                  # Episode-by-episode metrics
├── dpo_training.csv             # DPO loss progression
├── node_losses.csv              # Per-node mechanism losses
├── experiment.log               # Detailed training log
├── mechanism_contrast.png       # Oracle vs Student comparison
├── scm_graph.png                # SCM structure with losses
├── training_curves.png          # Learning curves
├── strategy_analysis.png        # Intervention distribution
└── value_diversity.csv          # Value exploration tracking
```

### Key Metrics in `metrics.csv`

- `dpo_loss`: DPO training loss
- `reward`: Information gain from intervention
- `cov_bonus`: Coverage bonus (node importance)
- `score`: Total score (reward + bonuses)
- `target`: Intervention target node
- `value`: Intervention value
- `episode`: Episode number
- `step`: Step within episode

## Reproducing Paper Results

### Table 1: 5-Node SCM Results

```bash
cd scripts/

# ACE (N=5 seeds)
./run_multi_seed.sh results/table1_ace 200

# Baselines (N=5 seeds)
for seed in 42 123 456 789 1011; do
    ./run_baselines.sh $seed results/table1_baselines/$seed 200
done
```

### Table 2: 15-Node Complex SCM Results

```bash
# ACE on complex SCM (N=5 seeds)
for seed in 42 123 456 789 1011; do
    ./run_ace_complex.sh $seed results/table2_ace/$seed 300
done

# Baselines on complex SCM
cd ../code/
for seed in 42 123 456 789 1011; do
    python experiments/complex_scm.py \
        --policy random \
        --episodes 200 \
        --seed $seed \
        --output ../scripts/results/table2_baselines/random/$seed
    
    python experiments/complex_scm.py \
        --policy greedy_collider \
        --episodes 200 \
        --seed $seed \
        --output ../scripts/results/table2_baselines/greedy/$seed
done
```

### Table 3: Ablation Study

```bash
cd scripts/

# Run ablations with 3 seeds
for seed in 42 123 456; do
    ./run_ablations.sh $seed results/table3_ablations/$seed 200
done
```

## Computing Statistics

After running multi-seed experiments, compute mean ± std:

```python
import pandas as pd
import glob
import numpy as np

# Collect results from all seeds
results = []
for path in glob.glob("results/ace_multi_seed/seed_*/run_*/metrics.csv"):
    df = pd.read_csv(path)
    # Get final episode metrics
    final = df[df['episode'] == df['episode'].max()]
    results.append(final['reward'].mean())  # Or other metric

mean_reward = np.mean(results)
std_reward = np.std(results)
print(f"Final Reward: {mean_reward:.3f} ± {std_reward:.3f}")
```

## Troubleshooting

### CUDA Out of Memory

If you encounter GPU memory issues:

```bash
# Reduce batch size or use CPU
python ace_experiments.py --device cpu ...
```

### Slow Training

For faster iteration during development:

```bash
# Reduce episodes and steps
python ace_experiments.py --episodes 50 --steps 10 ...
```

### Hugging Face Authentication

If model download fails:

```bash
# Set token environment variable
export HF_TOKEN="your_token_here"

# Or pass directly
python ace_experiments.py --token "your_token_here" ...
```

## Architecture Overview

### ACE Components

1. **Policy Network**: Qwen2.5-1.5B (LLM-based)
2. **DPO Training**: Preference learning from best/worst intervention pairs
3. **Lookahead Evaluation**: K=4 candidates evaluated on cloned learners
4. **Observational Training**: Periodic injection to prevent forgetting
5. **Diversity Mechanisms**: Collapse detection, smart breakers, forced diversity
6. **Epistemic Curiosity**: Strategic loser selection (novel > collapsed)
7. **Per-Node Convergence**: Individual mechanism convergence tracking

### Student SCM Learning

- **Architecture**: Multi-layer perceptrons (MLPs) for each mechanism
- **Training**: Gradient descent with replay buffer
- **Masking**: Interventions excluded from loss computation

## Hardware Requirements

### Minimum (CPU only)
- 4 cores
- 16GB RAM
- ~12 hours for full 5-node experiment

### Recommended (GPU)
- NVIDIA GPU with 8GB+ VRAM
- 8 CPU cores
- 32GB RAM
- ~30 minutes for full 5-node experiment

### Optimal (Multi-GPU)
- Multiple GPUs for parallel seed runs
- 64GB+ RAM
- Can run 5 seeds in parallel

## Citation

If you use this code, please cite our paper:

```bibtex
@article{ace2026,
  title={ACE: Active Causal Experimentation with Large Language Models},
  author={[Authors]},
  journal={[Journal]},
  year={2026}
}
```

## License

This code is released under the MIT License. See LICENSE file for details.

## Support

For questions or issues:
1. Check this README first
2. Review the paper for method details
3. Examine the code comments in `ace_experiments.py`
4. Open an issue on the repository (if public)

## Acknowledgments

This implementation uses:
- PyTorch for deep learning
- Transformers library for LLM integration
- NetworkX for graph operations

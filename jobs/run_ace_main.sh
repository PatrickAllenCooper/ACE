#!/bin/bash

#SBATCH --job-name=ace_main
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ACE Main Experiment Job Script
# The core DPO-based causal discovery experiment

# --- Environment Setup ---
if command -v module &> /dev/null; then
    module purge || true
    module load cuda || echo "Warning: Could not load cuda module."
fi

export HF_HOME="${HF_HOME:-/projects/$USER/cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/projects/$USER/cache/matplotlib}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" 2>/dev/null || true

if [ "$CONDA_DEFAULT_ENV" != "ace" ]; then
    if [ -f "/projects/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
        source /projects/$USER/miniconda3/etc/profile.d/conda.sh
        conda activate ace
    fi
fi

# --- Job Info ---
echo "========================================"
echo "ACE Main Experiment"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Output: $OUTPUT_DIR"
echo "Episodes: $EPISODES"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi
echo "========================================"
echo ""

# --- Run ACE Experiment ---
# Updated Jan 20, 2026 with critical efficiency improvements:
# - Early stopping (saves 80% compute time)
# - Enhanced root node learning (3x observational training)
# - Multi-objective diversity rewards (prevents policy collapse)
# - Reference policy stability (periodic updates)

python ace_experiments.py \
    --episodes ${EPISODES:-200} \
    --output "${OUTPUT_DIR:-results/ace}" \
    \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    \
    --root_fitting \
    --root_fit_interval 5 \
    --root_fit_samples 500 \
    --root_fit_epochs 100 \
    --use_dedicated_root_learner \
    --dedicated_root_interval 3 \
    \
    --undersampled_bonus 200.0 \
    --diversity_reward_weight 0.3 \
    --max_concentration 0.7 \
    --concentration_penalty 150.0 \
    \
    --update_reference_interval 25 \
    \
    --pretrain_steps 200 \
    --pretrain_interval 25 \
    --smart_breaker

# UPDATED Jan 21, 2026:
# - Increased max_concentration 0.5 → 0.7 (allow strategic concentration for collider learning)
# - Reduced concentration_penalty 200 → 150 (less harsh penalty)
# - Added adaptive threshold logic in compute_unified_diversity_score()
# - Added value novelty bonus to combat zero-reward saturation
# - Added emergency retraining when gradients near zero
# - Reduced candidates after episode 20 for speed

# --- Summary ---
echo ""
echo "========================================"
echo "ACE Main Experiment Complete"
echo "========================================"
echo "Finished: $(date)"
echo "Results: ${OUTPUT_DIR:-results/ace}"

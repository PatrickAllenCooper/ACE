#!/bin/bash

#SBATCH --job-name=ace_baselines
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

# Baselines Job Script
# Run all baseline methods: Random, Round-Robin, Max-Variance, PPO

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
echo "ACE Baselines Experiment"
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

# --- Run Baselines ---
# Updated Jan 20, 2026: Use improved observational training defaults for fair comparison
python baselines.py \
    --all_with_ppo \
    --episodes ${EPISODES:-100} \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --output "${OUTPUT_DIR:-results/baselines}"

# --- Summary ---
echo ""
echo "========================================"
echo "Baselines Complete"
echo "========================================"
echo "Finished: $(date)"
echo "Results: ${OUTPUT_DIR:-results/baselines}"

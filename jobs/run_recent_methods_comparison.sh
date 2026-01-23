#!/bin/bash

#SBATCH --job-name=recent_methods
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# Recent Methods Comparison Job
# Compares ACE to CORE-inspired and GACBO-inspired methods

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
echo "Recent Methods Comparison"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Output: $OUTPUT_DIR"
echo "Seed: $RANDOM_SEED"
echo ""

# --- Run Comparison ---
python -m experiments.compare_recent_methods \
    --episodes ${EPISODES:-50} \
    --steps 25 \
    --seed ${RANDOM_SEED:-42} \
    --output "${OUTPUT_DIR:-results/recent_methods_comparison}"

# --- Summary ---
echo ""
echo "========================================"
echo "Comparison Complete"
echo "========================================"
echo "Finished: $(date)"
echo "Results: ${OUTPUT_DIR:-results/recent_methods_comparison}"

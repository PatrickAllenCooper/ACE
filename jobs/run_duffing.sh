#!/bin/bash

#SBATCH --job-name=ace_duffing
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Duffing Oscillators Job Script
# Physics domain experiment with coupled oscillators

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
echo "Duffing Oscillators Experiment"
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

# --- Run Duffing Experiment ---
python -m experiments.duffing_oscillators \
    --episodes ${EPISODES:-100} \
    --output "${OUTPUT_DIR:-results/duffing}"

# --- Summary ---
echo ""
echo "========================================"
echo "Duffing Oscillators Complete"
echo "========================================"
echo "Finished: $(date)"
echo "Results: ${OUTPUT_DIR:-results/duffing}"

#!/bin/bash

#SBATCH --job-name=ace_phillips
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# Phillips Curve Job Script
# Economics domain experiment with FRED data

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
echo "Phillips Curve Experiment"
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

# --- Run Phillips Curve Experiment ---
python -m experiments.phillips_curve \
    --episodes ${EPISODES:-100} \
    --output "${OUTPUT_DIR:-results/phillips}"

# --- Summary ---
echo ""
echo "========================================"
echo "Phillips Curve Complete"
echo "========================================"
echo "Finished: $(date)"
echo "Results: ${OUTPUT_DIR:-results/phillips}"

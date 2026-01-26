#!/bin/bash

#SBATCH --job-name=complex_scm
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Complex 15-Node SCM Experiment
# Tests whether strategic intervention matters for harder problems

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
echo "Complex SCM Experiment"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Policy: ${POLICY:-random}"
echo "Episodes: ${EPISODES:-200}"
echo "========================================"
echo ""

# --- Run All Three Policies for Comparison ---
for POLICY in random smart_random greedy_collider; do
    echo ""
    echo "Running policy: $POLICY"
    echo "---"
    
    python -m experiments.complex_scm \
        --policy $POLICY \
        --episodes ${EPISODES:-200} \
        --steps 30 \
        --output "${OUTPUT_DIR:-results/complex_scm}"
    
    echo "[COMPLETE] $POLICY complete"
done

# --- Summary ---
echo ""
echo "========================================"
echo "Complex SCM Experiments Complete"
echo "========================================"
echo "Finished: $(date)"
echo "Results: ${OUTPUT_DIR:-results/complex_scm}"

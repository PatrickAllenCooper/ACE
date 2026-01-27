#!/bin/bash
#SBATCH --job-name=complex_scm
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/complex_scm_%j.out
#SBATCH --error=logs/complex_scm_%j.err

# Run ONLY Complex 15-Node SCM Experiments
# Extended baselines and lookahead ablation already complete
# This completes the final 1/3 of critical experiments

echo "=============================================="
echo "Complex 15-Node SCM Experiments"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=============================================="

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

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Working directory: $(pwd)"
echo ""

# Run complex SCM only (extended baselines and lookahead already done)
python -u scripts/runners/run_critical_experiments.py \
    --complex-scm \
    --seeds 42 123 456 789 1011 \
    --output-dir results/critical_experiments_20260127_075735

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="

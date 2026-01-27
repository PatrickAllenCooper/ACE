#!/bin/bash
#SBATCH --job-name=ace_abl_fast
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ablation_fast_%j.out
#SBATCH --error=logs/ablation_fast_%j.err

# Fast Ablation Studies - Reduced episodes with early stopping
# Expected runtime: 1-2 hours per ablation, 3-5 hours for all
#
# Usage:
#   sbatch jobs/run_ablations_fast.sh                      # Run all ablations sequentially
#   ABLATION=no_dpo sbatch jobs/run_ablations_fast.sh      # Run specific ablation (3 seeds)
#   bash jobs/workflows/submit_ablations_fast.sh           # Submit all 4 in parallel

echo "=============================================="
echo "Fast Ablation Studies for ICML Response"
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

# Parse ablation type
ABLATION_TYPE="${ABLATION:-all}"

if [ "$ABLATION_TYPE" == "all" ]; then
    echo "Running ALL ablations (4 types × 3 seeds = 12 runs)"
    python scripts/runners/run_ablations_fast.py --all --seeds 42 123 456 --max-episodes 100
else
    echo "Running specific ablation: $ABLATION_TYPE"
    python scripts/runners/run_ablations_fast.py --ablation "$ABLATION_TYPE" --seeds 42 123 456 --max-episodes 100
fi

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="

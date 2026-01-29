#!/bin/bash
#SBATCH --job-name=ace_15node_s42
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/ace_complex_s42_%j.out
#SBATCH --error=logs/ace_complex_s42_%j.err

# Single seed run of ACE on Complex 15-Node SCM
# With all optimizations for overnight run

echo "=============================================="
echo "ACE Complex SCM - Single Seed"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=============================================="

# Environment setup
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
echo ""

# Run seed 42 with all optimizations
SEED=42
OUTPUT_DIR="results/ace_complex_scm_optimized"
mkdir -p "$OUTPUT_DIR"

echo "Running ACE Complex SCM - Seed $SEED"
echo "Output: $OUTPUT_DIR"

python -u experiments/run_ace_complex_full.py \
    --episodes 300 \
    --seed $SEED \
    --output "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "Results: $OUTPUT_DIR/seed_${SEED}"
echo "=============================================="

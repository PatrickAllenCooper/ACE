#!/bin/bash
#SBATCH --job-name=ace_complex
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ace_complex_%j.out
#SBATCH --error=logs/ace_complex_%j.err

# Run standard ACE on Complex 15-Node SCM
# No ablations, no changes - just ACE as designed
# 5 seeds for statistical validation

echo "=============================================="
echo "ACE on Complex 15-Node SCM"
echo "Job ID: $SLURM_JOB_ID"
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
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT="results/ace_complex_scm_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT"

# Run ACE for each seed
for SEED in 42 123 456 789 1011; do
    echo ""
    echo "=========================================="
    echo "ACE Complex SCM - Seed $SEED"
    echo "=========================================="
    
    OUTPUT_DIR="${BASE_OUTPUT}/seed_${SEED}"
    
    python -u experiments/run_ace_complex_full.py \
        --episodes 200 \
        --seed $SEED \
        --output "$OUTPUT_DIR"
    
    echo "  ✓ Seed $SEED complete"
    
    # Extract final loss for summary
    if [ -f "$OUTPUT_DIR/run_*/results.csv" ]; then
        FINAL_LOSS=$(tail -1 "$OUTPUT_DIR"/run_*/results.csv | cut -d',' -f5)
        echo "  Final loss: $FINAL_LOSS"
    fi
    
    echo ""
done

echo "=============================================="
echo "ACE Complex SCM complete (all 5 seeds)"
echo "Completed: $(date)"
echo "Results: $BASE_OUTPUT"
echo "=============================================="

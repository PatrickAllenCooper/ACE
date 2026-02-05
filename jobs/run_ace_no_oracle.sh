#!/bin/bash
#SBATCH --job-name=ace_no_oracle
#SBATCH --partition=aa100
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/ace_no_oracle_%j.out
#SBATCH --error=logs/ace_no_oracle_%j.err

# ACE without Oracle Pretraining (N=5 seeds)
# Eliminates privileged information from main results
# Expected: 40-80 episodes per seed with early stopping

echo "=============================================="
echo "ACE without Oracle Pretraining (N=5 seeds)"
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

BASE_OUTPUT="results/ace_no_oracle"
mkdir -p "$BASE_OUTPUT"

# Run all 5 seeds sequentially
for SEED in 42 123 456 789 1011; do
    echo ""
    echo "=========================================="
    echo "ACE No Oracle - Seed $SEED"
    echo "=========================================="
    
    OUTPUT_DIR="${BASE_OUTPUT}/seed_${SEED}"
    
    python -u ace_experiments.py \
        --model "Qwen/Qwen2.5-1.5B" \
        --episodes 200 \
        --steps 25 \
        --seed $SEED \
        --output "$OUTPUT_DIR" \
        --obs_train_interval 3 \
        --obs_train_samples 200 \
        --obs_train_epochs 100 \
        --root_fitting \
        --use_dedicated_root_learner \
        --dedicated_root_interval 3 \
        --undersampled_bonus 200.0 \
        --diversity_reward_weight 0.3 \
        --max_concentration 0.7 \
        --pretrain_steps 0 \
        --smart_breaker
    
    echo "  ✓ Seed $SEED complete"
    echo ""
done

echo "=============================================="
echo "ACE No Oracle complete (all 5 seeds)"
echo "Completed: $(date)"
echo "Results: $BASE_OUTPUT"
echo ""
echo "Extract results:"
echo "  python scripts/compute_statistics.py $BASE_OUTPUT"
echo "=============================================="

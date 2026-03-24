#!/bin/bash

#SBATCH --job-name=ace_reviewer
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# ============================================================================
# ICML 2026 Reviewer Response Experiments
# ============================================================================
# Runs all experiments needed to address reviewer concerns:
#   1. N=10 ACE seeds (5 additional)
#   2. Bayesian OED baseline
#   3. Graph misspecification ablation
#   4. Hyperparameter sensitivity grid
#   5. K ablation (preference pair efficiency)
#   6. Duffing/Phillips baselines
#
# Usage:
#   sbatch jobs/run_reviewer_response.sh
# ============================================================================

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

echo "========================================"
echo "ICML 2026 Reviewer Response Experiments"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/reviewer_response_${TIMESTAMP}"

# --- Run Additional ACE Seeds (5 new seeds for N=10 total) ---
echo ""
echo ">>> Running additional ACE seeds (314, 271, 577, 618, 141) <<<"
for SEED in 314 271 577 618 141; do
    echo "  ACE seed $SEED..."
    python -u ace_experiments.py \
        --episodes 200 \
        --seed $SEED \
        --use_dedicated_root_learner \
        --obs_train_interval 3 \
        --obs_train_samples 200 \
        --obs_train_epochs 100 \
        --root_fitting \
        --root_fit_interval 5 \
        --root_fit_samples 500 \
        --root_fit_epochs 100 \
        --undersampled_bonus 200.0 \
        --diversity_reward_weight 0.3 \
        --max_concentration 0.7 \
        --concentration_penalty 150.0 \
        --update_reference_interval 25 \
        --pretrain_steps 200 \
        --pretrain_interval 25 \
        --smart_breaker \
        --output "${OUTPUT_DIR}/ace_seed_${SEED}" \
        2>&1 | tee "${OUTPUT_DIR}/ace_seed_${SEED}.log"
    echo "  ACE seed $SEED complete."
done

# --- Run Additional Baseline Seeds ---
echo ""
echo ">>> Running additional baseline seeds (314, 271, 577, 618, 141) at 171 episodes <<<"
for SEED in 314 271 577 618 141; do
    echo "  Baselines seed $SEED..."
    python -u baselines.py \
        --all_with_ppo \
        --episodes 171 \
        --seed $SEED \
        --obs_train_interval 3 \
        --obs_train_samples 200 \
        --output "${OUTPUT_DIR}/baselines_seed_${SEED}" \
        2>&1 | tee "${OUTPUT_DIR}/baselines_seed_${SEED}.log"
    echo "  Baselines seed $SEED complete."
done

# --- Run Reviewer Experiment Suite ---
echo ""
echo ">>> Running reviewer experiment suite <<<"
python -u scripts/runners/run_reviewer_experiments.py \
    --all \
    --seeds 42 123 456 789 1011 314 271 577 618 141 \
    --episodes 171 \
    --output "${OUTPUT_DIR}/experiments" \
    2>&1 | tee "${OUTPUT_DIR}/reviewer_experiments.log"

echo ""
echo "========================================"
echo "Reviewer Response Experiments Complete"
echo "========================================"
echo "Finished: $(date)"
echo "Results: ${OUTPUT_DIR}"

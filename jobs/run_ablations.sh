#!/bin/bash

#SBATCH --job-name=ace_ablations
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ACE Ablation Studies Job Script
# Tests: no-DPO, no-convergence, no-root-learner, no-diversity
#
# Usage:
#   sbatch jobs/run_ablations.sh
#   ABLATION=no_dpo SEED=42 sbatch jobs/run_ablations.sh

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

# --- Parameters ---
ABLATION="${ABLATION:-no_dpo}"
SEED="${SEED:-42}"
EPISODES="${EPISODES:-200}"
STEPS="${STEPS:-25}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-results/ablations_${TIMESTAMP}/${ABLATION}/seed_${SEED}}"

# --- Job Info ---
echo "========================================"
echo "ACE Ablation Study: $ABLATION"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Ablation: $ABLATION"
echo "Seed: $SEED"
echo "Episodes: $EPISODES"
echo "Output: $OUTPUT_DIR"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi
echo "========================================"
echo ""

# --- Set ablation-specific flags ---
case $ABLATION in
    no_dpo)
        echo "Testing: ACE without DPO (custom transformer policy)"
        ABLATION_FLAGS="--custom"
        ;;
    no_convergence)
        echo "Testing: ACE without per-node convergence"
        ABLATION_FLAGS="--no_per_node_convergence"
        ;;
    no_root_learner)
        echo "Testing: ACE without dedicated root learner"
        ABLATION_FLAGS="--no_dedicated_root_learner"
        ;;
    no_diversity)
        echo "Testing: ACE without diversity reward"
        ABLATION_FLAGS="--no_diversity_reward"
        ;;
    *)
        echo "ERROR: Unknown ablation: $ABLATION"
        echo "Valid options: no_dpo, no_convergence, no_root_learner, no_diversity"
        exit 1
        ;;
esac

echo ""

# --- Run Ablation ---
python ace_experiments.py \
    --episodes $EPISODES \
    --steps $STEPS \
    --seed $SEED \
    --output "$OUTPUT_DIR" \
    \
    --early_stopping \
    --early_stop_patience 20 \
    --early_stop_min_episodes 40 \
    --use_per_node_convergence \
    \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    \
    --root_fitting \
    --use_dedicated_root_learner \
    --dedicated_root_interval 3 \
    \
    --undersampled_bonus 200.0 \
    --diversity_reward_weight 0.3 \
    --max_concentration 0.7 \
    \
    --pretrain_steps 200 \
    --pretrain_interval 25 \
    --smart_breaker \
    \
    $ABLATION_FLAGS

# --- Summary ---
echo ""
echo "========================================"
echo "Ablation Complete: $ABLATION (seed $SEED)"
echo "========================================"
echo "Finished: $(date)"
echo "Results: $OUTPUT_DIR"
echo ""
echo "To analyze all ablations:"
echo "  python scripts/analyze_ablations.py results/ablations_${TIMESTAMP}/ --latex"

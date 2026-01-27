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
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err

# Single Ablation Job - No Python wrapper, direct execution
# Usage: ABLATION=no_dpo sbatch jobs/run_single_ablation.sh

echo "=============================================="
echo "Fast Ablation: $ABLATION"
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

# --- Set ablation type and flags ---
ABLATION="${ABLATION:-no_dpo}"

case $ABLATION in
    no_dpo)
        ABLATION_FLAGS="--custom"
        ;;
    no_convergence)
        ABLATION_FLAGS="--no_per_node_convergence"
        ;;
    no_root_learner)
        ABLATION_FLAGS="--no_dedicated_root_learner"
        ;;
    no_diversity)
        ABLATION_FLAGS="--no_diversity_reward"
        ;;
    *)
        echo "ERROR: Unknown ablation: $ABLATION"
        exit 1
        ;;
esac

echo "Ablation type: $ABLATION"
echo "Flags: $ABLATION_FLAGS"
echo ""

# --- Run all 3 seeds sequentially ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT="results/ablations_fast_${TIMESTAMP}"

for SEED in 42 123 456; do
    echo "========================================"
    echo "Running seed $SEED"
    echo "========================================"
    
    OUTPUT_DIR="${BASE_OUTPUT}/${ABLATION}/seed_${SEED}"
    
    python ace_experiments.py \
        --episodes 100 \
        --steps 25 \
        --seed $SEED \
        --output "$OUTPUT_DIR" \
        --early_stopping \
        --early_stop_patience 15 \
        --early_stop_min_episodes 30 \
        --use_per_node_convergence \
        --obs_train_interval 3 \
        --obs_train_samples 200 \
        --obs_train_epochs 100 \
        --root_fitting \
        --use_dedicated_root_learner \
        --dedicated_root_interval 3 \
        --undersampled_bonus 200.0 \
        --diversity_reward_weight 0.3 \
        --max_concentration 0.7 \
        --pretrain_steps 200 \
        --pretrain_interval 25 \
        --smart_breaker \
        $ABLATION_FLAGS
    
    echo "Completed seed $SEED"
    echo ""
done

echo "=============================================="
echo "Completed: $(date)"
echo "Results: $BASE_OUTPUT"
echo "=============================================="

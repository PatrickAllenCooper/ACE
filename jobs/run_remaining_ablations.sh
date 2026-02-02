#!/bin/bash
#SBATCH --job-name=ablations_final
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablations_final_%j.out
#SBATCH --error=logs/ablations_final_%j.err

# Complete remaining ablations: no_convergence, no_root_learner, no_diversity
# 3 ablations × 3 seeds = 9 runs total
# Saves results after each seed

echo "=============================================="
echo "Remaining Ablations (3 types × 3 seeds)"
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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT="results/ablations_complete"

# Run each ablation for all 3 seeds
for ABLATION in no_convergence no_root_learner no_diversity; do
    
    echo ""
    echo "=========================================="
    echo "ABLATION: $ABLATION"
    echo "=========================================="
    
    case $ABLATION in
        no_convergence)
            FLAGS="--custom --no_per_node_convergence"
            ;;
        no_root_learner)
            FLAGS="--custom --no_dedicated_root_learner"
            ;;
        no_diversity)
            FLAGS="--custom --no_diversity_reward"
            ;;
    esac
    
    for SEED in 42 123 456; do
        echo ""
        echo "Running $ABLATION seed $SEED..."
        
        OUTPUT_DIR="${BASE_OUTPUT}/${ABLATION}/seed_${SEED}"
        
        python -u ace_experiments.py \
            --episodes 100 \
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
            --pretrain_steps 200 \
            --pretrain_interval 25 \
            --smart_breaker \
            $FLAGS
        
        echo "  ✓ Completed $ABLATION seed $SEED"
    done
    
    echo "  ✓ $ABLATION complete (all 3 seeds)"
done

echo ""
echo "=============================================="
echo "All ablations complete"
echo "Completed: $(date)"
echo "Results: $BASE_OUTPUT"
echo "=============================================="

#!/bin/bash
#SBATCH --job-name=ace_abl_verified
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablations_verified_%j.out
#SBATCH --error=logs/ablations_verified_%j.err

# Verified Ablations - Fixed early stopping issue
# Usage: ABLATION=no_diversity sbatch jobs/run_ablations_verified.sh

echo "=============================================="
echo "Verified Ablation: $ABLATION"
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
export MPLBACKEND=Agg
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

# --- Set ablation type ---
ABLATION="${ABLATION:-no_diversity}"

case $ABLATION in
    no_convergence)
        ABLATION_FLAGS="--custom --no_per_node_convergence"
        ;;
    no_root_learner)
        ABLATION_FLAGS="--custom --no_dedicated_root_learner"
        ;;
    no_diversity)
        ABLATION_FLAGS="--custom --no_diversity_reward"
        ;;
    *)
        echo "ERROR: Unknown ablation: $ABLATION (use: no_convergence, no_root_learner, no_diversity)"
        exit 1
        ;;
esac

echo "Ablation type: $ABLATION"
echo "Flags: $ABLATION_FLAGS"
echo ""

# --- Run all 3 seeds sequentially ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT="results/ablations_verified_${TIMESTAMP}"

for SEED in 42 123 456; do
    echo "========================================"
    echo "Running $ABLATION seed $SEED"
    echo "========================================"
    
    OUTPUT_DIR="${BASE_OUTPUT}/${ABLATION}/seed_${SEED}"
    
    # CRITICAL: NO --early_stopping, NO --use_per_node_convergence
    # Let it run FULL 100 episodes to get true degradation
    python -u ace_experiments.py \
        --episodes 100 \
        --steps 25 \
        --seed $SEED \
        --output "$OUTPUT_DIR" \
        --obs_train_interval 3 \
        --obs_train_samples 200 \
        --obs_train_epochs 100 \
        --undersampled_bonus 200.0 \
        --max_concentration 0.7 \
        --pretrain_steps 200 \
        --pretrain_interval 25 \
        --smart_breaker \
        $ABLATION_FLAGS
    
    echo "  ✓ Completed $ABLATION seed $SEED"
    
    # CRITICAL: Save results immediately after each seed
    if [ -f "$OUTPUT_DIR/run_*/node_losses.csv" 2>/dev/null ]; then
        FINAL_LOSS=$(tail -1 $OUTPUT_DIR/run_*/node_losses.csv | cut -d',' -f3)
        echo "    Final loss: $FINAL_LOSS"
    fi
    echo ""
done

echo "=============================================="
echo "All seeds complete for $ABLATION"
echo "Completed: $(date)"
echo "Results: $BASE_OUTPUT"
echo "=============================================="

#!/bin/bash
#SBATCH --job-name=ace_abl_verified
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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
            ABLATION_FLAGS="--no_per_node_convergence"
            ;;
        no_root_learner)
            ABLATION_FLAGS="--no_dedicated_root_learner"
            ;;
        no_diversity)
            ABLATION_FLAGS="--no_diversity_reward"
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
SCRATCH_BASE="${SLURM_SCRATCH:-/scratch/local/$SLURM_JOB_ID}/ablations_${ABLATION}_${TIMESTAMP}"
FINAL_BASE="results/ablations_verified_${TIMESTAMP}"

mkdir -p "$SCRATCH_BASE"
mkdir -p "$FINAL_BASE"

echo "Scratch dir: $SCRATCH_BASE"
echo "Final output: $FINAL_BASE"
echo ""

for SEED in 42 123 456; do
    echo "========================================"
    echo "Running $ABLATION seed $SEED"
    echo "========================================"
    
    OUTPUT_DIR="${SCRATCH_BASE}/${ABLATION}/seed_${SEED}"
    
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
        --model "Qwen/Qwen2.5-1.5B" \
        --pretrain_interval 25 \
        --smart_breaker \
        $ABLATION_FLAGS
    
    echo "  ✓ Completed $ABLATION seed $SEED"
    
    # CRITICAL: Copy results from scratch to projects immediately
    FINAL_DIR="${FINAL_BASE}/${ABLATION}/seed_${SEED}"
    mkdir -p "$FINAL_DIR"
    cp -r "$OUTPUT_DIR/run_"* "$FINAL_DIR/" 2>/dev/null || echo "    Warning: Some files may not have copied"
    
    # Verify copy and report final loss
    if [ -f "$FINAL_DIR/run_"*/node_losses.csv 2>/dev/null ]; then
        FINAL_LOSS=$(tail -1 "$FINAL_DIR"/run_*/node_losses.csv 2>/dev/null | cut -d',' -f3)
        echo "    Final loss: $FINAL_LOSS"
        echo "    Saved to: $FINAL_DIR"
    fi
    echo ""
done

echo "=============================================="
echo "Copying all results from scratch to projects..."
cp -r "$SCRATCH_BASE"/* "$FINAL_BASE/" 2>/dev/null

# Clean scratch
echo "Cleaning scratch directory..."
rm -rf "$SCRATCH_BASE"

echo "=============================================="
echo "All seeds complete for $ABLATION"
echo "Completed: $(date)"
echo "Results: $FINAL_BASE"
echo "=============================================="

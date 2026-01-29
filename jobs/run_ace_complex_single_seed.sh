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

# Run seed 42 with all optimizations (FULL ACE architecture)
SEED=42
OUTPUT_DIR="results/ace_complex_scm_optimized"
mkdir -p "$OUTPUT_DIR"

echo "Running FULL ACE on Complex 15-Node SCM"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "All ACE components enabled: DPO, Lookahead, Diversity, Smart Breakers, Obs Training"

python -u experiments/run_ace_complex_full.py \
    --model "Qwen/Qwen2.5-1.5B" \
    --episodes 300 \
    --steps 50 \
    --candidates 4 \
    --seed $SEED \
    --output "$OUTPUT_DIR" \
    --lr 1e-5 \
    --learner_lr 2e-3 \
    --learner_epochs 100 \
    --buffer_steps 50 \
    --pretrain_steps 500 \
    --cov_bonus 60.0 \
    --diversity_reward_weight 0.3 \
    --max_concentration 0.4 \
    --diversity_constraint \
    --diversity_threshold 0.60 \
    --smart_breaker \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    --update_reference_interval 25 \
    --early_stopping \
    --use_per_node_convergence \
    --early_stop_min_episodes 40 \
    --node_convergence_patience 10 \
    --value_min -5.0 \
    --value_max 5.0

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "Results: $OUTPUT_DIR/seed_${SEED}"
echo "=============================================="

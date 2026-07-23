#!/bin/bash
# ACE budget-fairness worker (30-node) -- LOOKAHEAD_MODE, SEED, OUT via env
# See curc_budget_fairness_ace_seed.sh (5-node) for the mode semantics.

if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    module load cuda 2>/dev/null || true
fi
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true

export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /projects/paco0228/ACE
echo "30-node budget-fairness ACE lookahead_mode=$LOOKAHEAD_MODE seed=$SEED started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

EXTRA_FLAGS=""
if [ "$LOOKAHEAD_MODE" = "student" ]; then
    EXTRA_FLAGS="--lookahead_on_student"
fi

JOB_TAG="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="$OUT/ace_${LOOKAHEAD_MODE}/seed_${SEED}/job_${JOB_TAG}"

python -u ace_experiments.py \
    --large_scale 30 \
    --episodes 300 \
    --seed "$SEED" \
    --use_dedicated_root_learner \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    $EXTRA_FLAGS \
    --output "$OUTPUT_DIR"

echo "30-node budget-fairness ACE lookahead_mode=$LOOKAHEAD_MODE seed=$SEED finished at $(date)"

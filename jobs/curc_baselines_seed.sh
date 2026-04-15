#!/bin/bash
# Baselines seed worker -- SEED and OUT passed via env

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true
export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

cd /projects/paco0228/ACE
echo "Baselines seed $SEED started at $(date)"

python -u baselines.py \
    --all_with_ppo \
    --episodes 171 \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --output "$OUT/baselines/seed_${SEED}"

echo "Baselines seed $SEED finished at $(date)"

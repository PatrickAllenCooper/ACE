#!/bin/bash
# Hyperparameter grid cell worker -- COV, GAMMA, SEED, LABEL, OUT passed via env

if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    module load cuda 2>/dev/null || true
fi
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true

export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

cd /projects/paco0228/ACE
echo "Hyperparam $LABEL (cov=$COV, gamma=$GAMMA) seed $SEED started at $(date)"

python -u ace_experiments.py \
    --episodes 100 \
    --seed "$SEED" \
    --cov_bonus "$COV" \
    --diversity_reward_weight "$GAMMA" \
    --use_dedicated_root_learner \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    --output "$OUT/hyperparam/${LABEL}"

echo "Hyperparam $LABEL seed $SEED finished at $(date)"

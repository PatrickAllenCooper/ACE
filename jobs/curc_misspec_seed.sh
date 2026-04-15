#!/bin/bash
# Graph misspecification ACE worker -- MISSPEC, SEED, OUT passed via env

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
echo "Graph misspec $MISSPEC seed $SEED started at $(date)"

MISSPEC_ARG=""
if [ "$MISSPEC" != "none" ]; then
    MISSPEC_ARG="--graph_misspec $MISSPEC"
fi

python -u ace_experiments.py \
    --episodes 200 \
    --seed "$SEED" \
    --use_dedicated_root_learner \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    $MISSPEC_ARG \
    --output "$OUT/misspec/${MISSPEC}/seed_${SEED}"

echo "Graph misspec $MISSPEC seed $SEED finished at $(date)"

#!/bin/bash
# ACE seed worker -- called by curc_submit_all.sh with SEED and OUT env vars
# Uses exact config from run_ace_only.sh (the config that produced 0.61 median)

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
echo "ACE seed $SEED started on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u ace_experiments.py \
    --episodes 200 \
    --seed "$SEED" \
    --use_dedicated_root_learner \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    --output "$OUT/ace/seed_${SEED}"

echo "ACE seed $SEED finished at $(date)"

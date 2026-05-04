#!/bin/bash
# 30-node baseline worker -- METHOD, SEED, OUT passed via sbatch --export
#
# Runs the MLP-based student learner for one (method, seed) pair on the
# 30-node LargeScaleSCM, producing results comparable to ACE on the same system.
#
# SLURM vars expected:
#   METHOD  : random | round_robin | max_variance
#   SEED    : integer seed
#   OUT     : absolute path to results root
#
# Output lands in: $OUT/$METHOD/seed_$SEED/

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true

export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

cd /projects/paco0228/ACE
echo "30-node baseline method=$METHOD seed=$SEED started at $(date)"

python -u scripts/runners/run_30node_baseline_seed.py \
    --method   "$METHOD" \
    --seed     "$SEED"   \
    --episodes 150       \
    --steps    25        \
    --obs_train_interval 3 \
    --obs_train_samples  200 \
    --output   "$OUT"

echo "30-node baseline method=$METHOD seed=$SEED finished at $(date)"

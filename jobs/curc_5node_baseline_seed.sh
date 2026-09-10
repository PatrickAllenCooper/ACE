#!/bin/bash
# 5-node baseline worker -- METHOD, SEED, OUT (and optionally EPISODES) via env
#
# Runs scripts/runners/run_5node_baseline_seed.py for one (method, seed) at
# the Table 1 protocol (171 episodes x 25 steps, observational refresh every
# 3 steps with 200 samples). Since the Sept 2026 metric audit the runner logs
# BOTH evaluators per step (observational/unweighted 'total_loss' and ACE's
# broad-range/root-weighted 'ace_total_loss').
#
# Output: $OUT/$METHOD/seed_$SEED/{node_losses.csv,per_episode.csv,summary.csv}

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

cd /projects/paco0228/ACE
echo "5-node baseline method=$METHOD seed=$SEED episodes=${EPISODES:-171} started at $(date)"

python -u scripts/runners/run_5node_baseline_seed.py \
    --method   "$METHOD" \
    --seed     "$SEED"   \
    --episodes "${EPISODES:-171}" \
    --steps    25 \
    --obs_train_interval 3 \
    --obs_train_samples  200 \
    --output   "$OUT"

echo "5-node baseline method=$METHOD seed=$SEED finished at $(date)"

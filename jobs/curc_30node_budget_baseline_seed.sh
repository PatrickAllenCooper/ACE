#!/bin/bash
# 30-node baseline budget-fairness worker -- METHOD, QUERY_BUDGET, SEED, OUT via env
#
# Runs scripts/runners/run_30node_baseline_seed.py at a total-environment-
# query budget matched to Phase 1's ACE (env-lookahead) measurement. Includes
# bayesian_oed (F4Cb's omitted baseline) alongside the passive heuristics.

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true

export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

cd /projects/paco0228/ACE
echo "30-node budget-fairness baseline method=$METHOD query_budget=$QUERY_BUDGET seed=$SEED started at $(date)"

python -u scripts/runners/run_30node_baseline_seed.py \
    --method "$METHOD" \
    --seed "$SEED" \
    --episodes 5000 \
    --steps 25 \
    --query_budget "$QUERY_BUDGET" \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --output "$OUT"

echo "30-node budget-fairness baseline method=$METHOD query_budget=$QUERY_BUDGET seed=$SEED finished at $(date)"

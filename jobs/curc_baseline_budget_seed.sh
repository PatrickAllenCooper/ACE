#!/bin/bash
# 5-node baseline budget-fairness worker -- QUERY_BUDGET, SEED, OUT via env
#
# Runs all baselines.py methods (including PPO) at a total-environment-
# query budget matched to ACE's env-based lookahead cost, instead of a fixed
# episode count. Phase 2 of curc_submit_5node_budget_fairness.sh.

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
echo "Budget-fairness baselines query_budget=$QUERY_BUDGET seed=$SEED started at $(date)"

python -u baselines.py \
    --all_with_ppo \
    --episodes 2000 \
    --query_budget "$QUERY_BUDGET" \
    --seed "$SEED" \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --output "$OUT/seed_${SEED}"

echo "Budget-fairness baselines query_budget=$QUERY_BUDGET seed=$SEED finished at $(date)"

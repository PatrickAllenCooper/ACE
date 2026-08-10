#!/bin/bash
# 5-node baseline budget-fairness worker -- METHOD, QUERY_BUDGET, SEED, OUT via env
#
# Runs scripts/runners/run_5node_baseline_seed.py at a total-environment-
# query budget matched to Phase 1's ACE (env-lookahead) measurement. One
# method per job (mirrors the 30-node worker), writing to
# $OUT/$METHOD/seed_$SEED/{node_losses.csv,summary.csv,query_budget.json}
# so scripts/analysis/aggregate_budget_fairness.py can read it directly.
#
# Bayesian OED is not one of METHOD's choices here -- the 5-node Bayesian
# OED baseline lives in run_reviewer_experiments.py and needs its own
# query-budget wiring; see curc_submit_5node_budget_fairness_baselines.sh.

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true

export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

cd /projects/paco0228/ACE
echo "5-node budget-fairness baseline method=$METHOD query_budget=$QUERY_BUDGET seed=$SEED started at $(date)"

python -u scripts/runners/run_5node_baseline_seed.py \
    --method "$METHOD" \
    --seed "$SEED" \
    --episodes 2000 \
    --steps 25 \
    --query_budget "$QUERY_BUDGET" \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --output "$OUT"

echo "5-node budget-fairness baseline method=$METHOD query_budget=$QUERY_BUDGET seed=$SEED finished at $(date)"

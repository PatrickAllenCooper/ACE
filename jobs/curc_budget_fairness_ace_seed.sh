#!/bin/bash
# ACE budget-fairness worker (5-node) -- LOOKAHEAD_MODE, SEED, OUT via env
#
# LOOKAHEAD_MODE:
#   env     -- standard ACE lookahead (queries the ground-truth environment
#              for all K candidates per step; only the executed winner is
#              charged against the reported intervention budget)
#   student -- --lookahead_on_student: scores candidates on the learner's own
#              current beliefs, so lookahead makes zero oracle queries and
#              the executed-interventions budget is honest on its own
#
# Phase 1 of the budget-fairness suite (see curc_submit_5node_budget_fairness.sh).
# Every run writes query_budget.json; average the "env" runs' total sample
# count across seeds to get the QUERY_BUDGET argument for Phase 2
# (curc_submit_5node_budget_fairness_baselines.sh).

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
echo "Budget-fairness ACE lookahead_mode=$LOOKAHEAD_MODE seed=$SEED started at $(date)"

EXTRA_FLAGS=""
if [ "$LOOKAHEAD_MODE" = "student" ]; then
    EXTRA_FLAGS="--lookahead_on_student"
fi

python -u ace_experiments.py \
    --episodes 200 \
    --seed "$SEED" \
    --use_dedicated_root_learner \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    $EXTRA_FLAGS \
    --output "$OUT/ace_${LOOKAHEAD_MODE}/seed_${SEED}"

echo "Budget-fairness ACE lookahead_mode=$LOOKAHEAD_MODE seed=$SEED finished at $(date)"

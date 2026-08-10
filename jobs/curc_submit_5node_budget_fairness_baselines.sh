#!/bin/bash
# =============================================================================
# Budget-fairness suite, 5-node benchmark, Phase 2 of 2.
#
# Random, Round-Robin, Max-Variance, and PPO at a total-environment-query
# budget matched to Phase 1's ACE (env-lookahead) measurement, instead of a
# fixed episode count. Bayesian OED is separate (its 5-node baseline lives in
# run_reviewer_experiments.py, not baselines.py, and needs its own
# query-budget wiring -- not yet done; track as a follow-up).
#
# One method per job (mirrors curc_submit_30node_budget_fairness_baselines.sh
# and its run_30node_baseline_seed.py), via the new
# scripts/runners/run_5node_baseline_seed.py, so output lands as
# baselines/{method}/seed_{seed}/{node_losses.csv,summary.csv,
# query_budget.json} -- the layout scripts/analysis/aggregate_budget_fairness.py
# expects. (A prior version of this script called `baselines.py
# --all_with_ppo` directly into one shared timestamped directory per seed,
# which does not match that layout and would have silently produced zero
# baseline rows at aggregation time; fixed before Phase 1 even finished so
# it is caught before burning compute on a broken Phase 2.)
#
# Usage (from /projects/paco0228/ACE), AFTER Phase 1 has completed:
#   cd /projects/paco0228/ACE
#   git pull   # ensure run_5node_baseline_seed.py is present
#   bash jobs/curc_submit_5node_budget_fairness_baselines.sh <QUERY_BUDGET>
#
# <QUERY_BUDGET> = mean total sample count from Phase 1's
#   results/curc_5node_budget_fairness/ace_env/seed_*/*/query_budget.json
#   ("total" -> "samples"), computed as described in
#   curc_submit_5node_budget_fairness.sh's header comment.
#
# Set SKIP_COMPLETED=1 to skip any (method, seed) cell that already has a
# summary.csv, so a resubmission after a partial failure does not requeue
# cells that already finished (this is only a completed-output check --
# it does not detect a duplicate that is still PENDING/RUNNING in the
# queue; check `squeue -u $USER` yourself before resubmitting).
#
# 4 methods x 5 seeds = 20 jobs.
#
# Output: results/curc_5node_budget_fairness/baselines/{method}/seed_{seed}/
# =============================================================================

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <QUERY_BUDGET>"
    echo "  QUERY_BUDGET = mean total sample count from Phase 1's ace_env runs"
    echo "  (see curc_submit_5node_budget_fairness.sh's header comment for how to compute it)"
    exit 1
fi
QUERY_BUDGET="$1"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine/paco0228/ACE/results/curc_5node_budget_fairness/baselines"
mkdir -p "$OUT/logs"

cell_done() {
    local method=$1 seed=$2
    [[ -f "$OUT/${method}/seed_${seed}/summary.csv" ]]
}

echo "================================================================"
echo " 5-node Budget-Fairness Suite, Phase 2 (baselines) -- up to 20 jobs"
echo "================================================================"
echo " Query budget : $QUERY_BUDGET total environment samples"
echo " Output       : $OUT"
echo " Started      : $(date)"
echo "================================================================"

METHODS="random round_robin max_variance ppo"
SEEDS="42 123 456 789 1011"

for METHOD in $METHODS; do
    for SEED in $SEEDS; do
        if [ "$SKIP_COMPLETED" = "1" ] && cell_done "$METHOD" "$SEED"; then
            echo "  SKIP (done): bf5bl_${METHOD} seed=$SEED"
            continue
        fi
        JOB=$(sbatch --parsable \
            --job-name="bf5bl_${METHOD:0:3}_s${SEED}" \
            --partition=acpu --qos=cpu-normal \
            --nodes=1 --ntasks=1 \
            --cpus-per-task=4 --mem=8G \
            --time=08:00:00 \
            --output="$OUT/logs/${METHOD}_seed${SEED}_%j.out" \
            --error="$OUT/logs/${METHOD}_seed${SEED}_%j.err" \
            --export=ALL,METHOD=$METHOD,QUERY_BUDGET=$QUERY_BUDGET,SEED=$SEED,OUT=$OUT \
            jobs/curc_5node_budget_baseline_seed.sh)
        echo "  Submitted: method=$METHOD seed=$SEED (budget=$QUERY_BUDGET) -> Job $JOB"
    done
done

echo ""
echo "Jobs submitted (up to 20; fewer if SKIP_COMPLETED=1 skipped some)."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally with:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"
echo ""
echo "Decision gate: python scripts/analysis/aggregate_budget_fairness.py \\"
echo "  --root /scratch/alpine/paco0228/ACE/results/curc_5node_budget_fairness"

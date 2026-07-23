#!/bin/bash
# =============================================================================
# Budget-fairness suite, 5-node benchmark, Phase 2 of 2.
#
# Runs all baselines.py methods (Random, Round-Robin, Max-Variance, PPO --
# Bayesian OED is separate, via baselines.py's own MaxVariancePolicy-style
# candidate probing does not apply to it directly at 5 nodes; see
# run_reviewer_experiments.py for the 5-node Bayesian OED baseline) at a
# total-environment-query budget matched to Phase 1's ACE (env-lookahead)
# measurement, instead of a fixed episode count.
#
# Usage (from /projects/paco0228/ACE), AFTER Phase 1 has completed:
#   cd /projects/paco0228/ACE
#   git pull   # ensure latest --query_budget / --seed flags in baselines.py
#   bash jobs/curc_submit_5node_budget_fairness_baselines.sh <QUERY_BUDGET>
#
# <QUERY_BUDGET> = mean total sample count from Phase 1's
#   results/curc_5node_budget_fairness/ace_env/seed_*/*/query_budget.json
#   ("total" -> "samples"), computed as described in
#   curc_submit_5node_budget_fairness.sh's header comment.
#
# 5 seeds (each seed runs all 4 baseline methods internally).
#
# Output: results/curc_5node_budget_fairness/baselines/seed_{seed}/
# =============================================================================

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <QUERY_BUDGET>"
    echo "  QUERY_BUDGET = mean total sample count from Phase 1's ace_env runs"
    echo "  (see this script's header comment for how to compute it)"
    exit 1
fi
QUERY_BUDGET="$1"

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine1/paco0228/ACE/results/curc_5node_budget_fairness/baselines"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " 5-node Budget-Fairness Suite, Phase 2 (baselines) -- 5 jobs"
echo "================================================================"
echo " Query budget : $QUERY_BUDGET total environment samples"
echo " Output       : $OUT"
echo " Started      : $(date)"
echo "================================================================"

SEEDS="42 123 456 789 1011"

for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
        --job-name="bf5bl_s${SEED}" \
        --partition=amilan --qos=normal \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=8G \
        --time=08:00:00 \
        --output="$OUT/logs/baselines_seed${SEED}_%j.out" \
        --error="$OUT/logs/baselines_seed${SEED}_%j.err" \
        --export=ALL,QUERY_BUDGET=$QUERY_BUDGET,SEED=$SEED,OUT=$OUT \
        jobs/curc_baseline_budget_seed.sh)
    echo "  Submitted: baselines seed=$SEED (budget=$QUERY_BUDGET) -> Job $JOB"
done

echo ""
echo "5 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally with:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"
echo ""
echo "Decision gate: compare final/best MSE against Phase 1's ace_env and"
echo "ace_student results at this matched total-query budget."

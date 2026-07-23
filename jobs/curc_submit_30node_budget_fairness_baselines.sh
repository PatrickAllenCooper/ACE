#!/bin/bash
# =============================================================================
# Budget-fairness suite, 30-node benchmark, Phase 2 of 2.
#
# Random, Round-Robin, Max-Variance, and Bayesian OED at a total-environment-
# query budget matched to Phase 1's ACE (env-lookahead) measurement.
#
# Usage (from /projects/paco0228/ACE), AFTER Phase 1 has completed:
#   cd /projects/paco0228/ACE
#   git pull   # ensure latest --query_budget flag in run_30node_baseline_seed.py
#   bash jobs/curc_submit_30node_budget_fairness_baselines.sh <QUERY_BUDGET>
#
# 4 methods x 5 seeds = 20 jobs.
#
# Output: results/curc_30node_budget_fairness/baselines/{method}/seed_{seed}/
# =============================================================================

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <QUERY_BUDGET>"
    exit 1
fi
QUERY_BUDGET="$1"

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine1/paco0228/ACE/results/curc_30node_budget_fairness/baselines"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " 30-node Budget-Fairness Suite, Phase 2 (baselines) -- 20 jobs"
echo "================================================================"
echo " Query budget : $QUERY_BUDGET total environment samples"
echo " Output       : $OUT"
echo " Started      : $(date)"
echo "================================================================"

METHODS="random round_robin max_variance bayesian_oed"
SEEDS="42 123 456 789 1011"

for METHOD in $METHODS; do
    for SEED in $SEEDS; do
        JOB=$(sbatch --parsable \
            --job-name="bf30bl_${METHOD:0:3}_s${SEED}" \
            --partition=amilan --qos=normal \
            --nodes=1 --ntasks=1 \
            --cpus-per-task=4 --mem=8G \
            --time=10:00:00 \
            --output="$OUT/logs/${METHOD}_seed${SEED}_%j.out" \
            --error="$OUT/logs/${METHOD}_seed${SEED}_%j.err" \
            --export=ALL,METHOD=$METHOD,QUERY_BUDGET=$QUERY_BUDGET,SEED=$SEED,OUT=$OUT \
            jobs/curc_30node_budget_baseline_seed.sh)
        echo "  Submitted: method=$METHOD seed=$SEED (budget=$QUERY_BUDGET) -> Job $JOB"
    done
done

echo ""
echo "20 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally with:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"

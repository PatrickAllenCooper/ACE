#!/bin/bash
# =============================================================================
# Budget-fairness suite, 5-node benchmark, Phase 1 of 2.
#
# The decisive experiment for the ICLR resubmission: 4 of 5 NeurIPS 2026
# reviewers (wZrW, F4Cb, d6tT, TnpG) independently flagged that ACE's
# lookahead queries the ground-truth environment once per candidate (K=4)
# but only the executed winner counts against the reported intervention
# budget, while baselines get exactly 1 query per step. This phase runs ACE
# under BOTH accountings so Phase 2 can match baselines fairly to each:
#
#   ace_env     : standard ACE (env-based lookahead, K queries/step -- what
#                 the NeurIPS submission reported)
#   ace_student : --lookahead_on_student (zero-query lookahead; the
#                 executed-interventions budget is honest on its own)
#
# 2 conditions x 5 seeds = 10 jobs.
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   git pull   # ensure latest --lookahead_on_student / query_budget.json
#   bash jobs/curc_submit_5node_budget_fairness.sh
#
# After both conditions finish, compute the mean total query count for
# ace_env from its query_budget.json files, then run Phase 2:
#   python -c "
#   import json, glob, numpy as np
#   totals = [json.load(open(f))['total']['samples']
#             for f in glob.glob('results/curc_5node_budget_fairness/ace_env/seed_*/*/query_budget.json')]
#   print(int(np.mean(totals)))
#   "
#   bash jobs/curc_submit_5node_budget_fairness_baselines.sh <QUERY_BUDGET>
#
# Decision gate (per the ICLR plan): if ACE still beats the query-budget-
# matched baselines, keep the strong headline claim with dual accounting
# reported explicitly; if not, lead with ace_student as the paper's primary
# configuration since its reported budget is honest without any matching.
#
# Output: results/curc_5node_budget_fairness/ace_{env,student}/seed_{seed}/
#
# SLURM resources per job: same as the main 5-node ACE runs (aa100, 8h).
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine1/paco0228/ACE/results/curc_5node_budget_fairness"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " 5-node Budget-Fairness Suite, Phase 1 (ACE variants) -- 10 jobs"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

MODES="env student"
SEEDS="42 123 456 789 1011"

for MODE in $MODES; do
    for SEED in $SEEDS; do
        JOB=$(sbatch --parsable \
            --job-name="bf5_${MODE:0:3}_s${SEED}" \
            --partition=aa100 --qos=normal \
            --nodes=1 --ntasks=1 --gres=gpu:1 \
            --cpus-per-task=8 --mem=32G \
            --time=08:00:00 \
            --output="$OUT/logs/ace_${MODE}_seed${SEED}_%j.out" \
            --error="$OUT/logs/ace_${MODE}_seed${SEED}_%j.err" \
            --export=ALL,LOOKAHEAD_MODE=$MODE,SEED=$SEED,OUT=$OUT \
            jobs/curc_budget_fairness_ace_seed.sh)
        echo "  Submitted: ACE lookahead_mode=$MODE seed=$SEED -> Job $JOB"
    done
done

echo ""
echo "10 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally and proceed to Phase 2 (see header)."

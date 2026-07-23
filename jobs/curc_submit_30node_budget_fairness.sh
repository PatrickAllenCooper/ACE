#!/bin/bash
# =============================================================================
# Budget-fairness suite, 30-node benchmark, Phase 1 of 2.
#
# Same rationale as curc_submit_5node_budget_fairness.sh, at the scale that
# is the paper's actual scalability showcase. 2 conditions x 5 seeds = 10 jobs
# (also serves as part of the N=3 -> N=5 seed expansion for ace_env, since
# ace_env with the default lookahead is the same configuration as the
# submission's headline 30-node ACE result).
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   git pull
#   bash jobs/curc_submit_30node_budget_fairness.sh
#
# After both conditions finish, compute the mean total query count for
# ace_env (see curc_submit_5node_budget_fairness.sh for the one-liner, with
# the glob pattern updated to this OUT path), then run Phase 2:
#   bash jobs/curc_submit_30node_budget_fairness_baselines.sh <QUERY_BUDGET>
#
# Output: results/curc_30node_budget_fairness/ace_{env,student}/seed_{seed}/job_{jobid}/
#
# SLURM resources per job: same as curc_large_scale_seed.sh (aa100, 8h, 64G).
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine1/paco0228/ACE/results/curc_30node_budget_fairness"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " 30-node Budget-Fairness Suite, Phase 1 (ACE variants) -- 10 jobs"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

MODES="env student"
SEEDS="42 123 456 789 1011"

for MODE in $MODES; do
    for SEED in $SEEDS; do
        JOB=$(sbatch --parsable \
            --job-name="bf30_${MODE:0:3}_s${SEED}" \
            --partition=aa100 --qos=normal \
            --nodes=1 --ntasks=1 --gres=gpu:1 \
            --cpus-per-task=8 --mem=64G \
            --time=08:00:00 \
            --output="$OUT/logs/ace_${MODE}_seed${SEED}_%j.out" \
            --error="$OUT/logs/ace_${MODE}_seed${SEED}_%j.err" \
            --export=ALL,LOOKAHEAD_MODE=$MODE,SEED=$SEED,OUT=$OUT \
            jobs/curc_30node_budget_ace_seed.sh)
        echo "  Submitted: ACE 30-node lookahead_mode=$MODE seed=$SEED -> Job $JOB"
    done
done

echo ""
echo "10 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally and proceed to Phase 2 (see header)."

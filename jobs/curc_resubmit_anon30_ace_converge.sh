#!/bin/bash
# =============================================================================
# Phase 1 (rebuttal-critical): rerun anon30 ACE to its best-MSE plateau so the
# paper can state the convergence story plainly instead of relying on a
# wall-time-truncated ~19-episode run.
#
# Why: the May 2026 anon30 ACE jobs hit the 8h wall at ~ep 19 (vs canonical-30's
# 120). Best-MSE had already plateaued by ~ep 20, but the rebuttal needs that
# shown, not asserted. This resubmit uses STABLE_DIR=1 so each window RESUMES
# from the prior checkpoint, and caps at 40 episodes (plateau + generous
# margin) with a long wall-time. Re-run this script for additional windows
# until `n_episodes` >= ~40 in the aggregate.
#
# Lands in the SAME tree the aggregator/figure already read:
#   results/curc_30node_followup/anon30/ace/seed_<seed>/run_*/node_losses.csv
#
# Usage (from /projects/paco0228/ACE):
#   git pull
#   bash jobs/curc_resubmit_anon30_ace_converge.sh
#   # ...later, after it runs out of wall time, run it again to continue:
#   bash jobs/curc_resubmit_anon30_ace_converge.sh
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/projects/paco0228/ACE/results/curc_30node_followup"
mkdir -p "$OUT/logs"

SEEDS="${SEEDS:-42 123 456 789 1011}"

echo "================================================================"
echo " anon30 ACE convergence rerun (STABLE_DIR resume, cap 40 ep)"
echo " Output : $OUT   Seeds: $SEEDS   Started: $(date)"
echo "================================================================"

for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
        --job-name="a30conv_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=64G \
        --time=24:00:00 \
        --output="$OUT/logs/a30conv_seed${SEED}_%j.out" \
        --error="$OUT/logs/a30conv_seed${SEED}_%j.err" \
        --export=ALL,CONDITION=anon30,METHOD=ace,SEED=$SEED,OUT=$OUT,STABLE_DIR=1,EPISODES_ACE=40 \
        jobs/curc_30node_followup_seed.sh)
    echo "  Submitted: anon30 ACE converge seed=$SEED -> Job $JOB"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "After completion, refresh metrics:"
echo "  conda activate ace"
echo "  python scripts/analysis/aggregate_followup_results.py"
echo "  # check best_episode column -- it documents the plateau for the prose caveat"

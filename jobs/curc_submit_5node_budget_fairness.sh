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
# SLURM resources per job: aa100, 8h, 128G host RAM. Bumped from 64G after
# every one of the 10 jobs (both 32G and 64G attempts, Aug 3 and Aug 8) was
# OOM-killed at 200 episodes. Notably the 30-node Phase 1 (same 64G, same
# ace_experiments.py path) completed cleanly at its capped 40 episodes --
# the 5-node run's 200-episode budget is the more likely driver than graph
# size, which points at per-episode host-RAM growth (e.g. accumulating
# diagnostics/candidate-probe buffers across episodes) rather than a fixed
# per-run cost. Flagged as a follow-up profiling item; 128G is a practical
# unblock, not a fix for the underlying growth if it exists.
# =============================================================================

set -euo pipefail

# GPU targeting. Default to RTX Pro 6000 (Aug 2026 Alpine expansion): aa100's
# a100-40gb GRES hard-caps a single-GPU request at 80640 MiB (~78.75GB) host
# RAM regardless of --mem, which the 200-episode 5-node run already exceeds
# at 64G -- there is no room to raise it further on that GRES type. RTX Pro
# 6000 nodes proved they can grant 90-140G per single-GPU job cleanly in the
# model-scale sweep, so route there instead. Override if needed, e.g. to go
# back to aa100 (only viable up to ~78G) or to H200:
#   GPU_PARTITION=<partition> GPU_QOS=<qos> GPU_GRES=<gres> bash jobs/curc_submit_5node_budget_fairness.sh
GPU_PARTITION="${GPU_PARTITION:-artxpro6000}"
GPU_QOS="${GPU_QOS:-gpu-normal}"
GPU_GRES="${GPU_GRES:-gpu:rtx_pro_6000:1}"

# Set SKIP_COMPLETED=1 to skip any (mode, seed) cell that already has a
# query_budget.json (i.e. finished a full run, not just a wall-time-killed
# partial one), so a resubmission after an OOM fix does not requeue cells
# that already succeeded. This is only a completed-output check -- it does
# not detect a duplicate that is still PENDING/RUNNING in the queue; check
# `squeue -u $USER` yourself before resubmitting (the Aug 2026 quota
# exhaustion incident was exactly this: resubmitting while old copies were
# still queued, not re-running already-completed cells).
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine/paco0228/ACE/results/curc_5node_budget_fairness"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " 5-node Budget-Fairness Suite, Phase 1 (ACE variants) -- 10 jobs"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

MODES="env student"
SEEDS="42 123 456 789 1011"

cell_done() {
    local mode=$1 seed=$2
    local seed_dir="$OUT/ace_${mode}/seed_${seed}"
    [[ -d "$seed_dir" ]] || return 1
    find "$seed_dir" -name query_budget.json 2>/dev/null | grep -q .
}

for MODE in $MODES; do
    for SEED in $SEEDS; do
        if [ "$SKIP_COMPLETED" = "1" ] && cell_done "$MODE" "$SEED"; then
            echo "  SKIP (done): bf5_${MODE} seed=$SEED"
            continue
        fi
        JOB=$(sbatch --parsable \
            --job-name="bf5_${MODE:0:3}_s${SEED}" \
            --partition=$GPU_PARTITION --qos=$GPU_QOS \
            --nodes=1 --ntasks=1 --gres=$GPU_GRES \
            --cpus-per-task=8 --mem=128G \
            --time=08:00:00 \
            --output="$OUT/logs/ace_${MODE}_seed${SEED}_%j.out" \
            --error="$OUT/logs/ace_${MODE}_seed${SEED}_%j.err" \
            --export=ALL,LOOKAHEAD_MODE=$MODE,SEED=$SEED,OUT=$OUT \
            jobs/curc_budget_fairness_ace_seed.sh)
        echo "  Submitted: ACE lookahead_mode=$MODE seed=$SEED -> Job $JOB"
    done
done

echo ""
echo "Jobs submitted (up to 10; fewer if SKIP_COMPLETED=1 skipped some)."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally and proceed to Phase 2 (see header)."

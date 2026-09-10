#!/bin/bash
# =============================================================================
# Controls for the 30-node seed expansion (seeds 2022-2026), demanded by every
# reviewer in the Sept 10 panel: the seed sets the graph wiring, so a claim that
# those seeds "fall to the passive plateau because DPO destabilized" is
# confounded with instance difficulty unless ACE-w/o-DPO and a passive baseline
# are run on the SAME graphs.
#
#   zero_shot_lm : ACE with --no_dpo (LM proposer + lookahead, no weight updates)
#                  -- 5 GPU jobs, identical config/budget to the expansion runs
#                  (--large_scale 30 --episodes 300), via curc_large_scale_seed.sh
#   random       : now produced by jobs/curc_submit_metric_audit_reruns.sh (t2_30node)
#
# Decision rule (stated in advance):
#   w/o-DPO clearly below the plateau on >=4/5 seeds AND Random at the plateau
#     -> DPO instability is real on those instances
#   w/o-DPO also at the plateau            -> the instances are harder; the
#                                             instability narrative must be withdrawn
#
# Usage (from /projects/paco0228/ACE):
#   git pull
#   SKIP_COMPLETED=1 GPU_PARTITION=ah200 GPU_QOS=gpu-long GPU_GRES=gpu:h200:1 \
#       WALL_TIME=48:00:00 bash jobs/curc_submit_30node_expansion_controls.sh
# Output: $OUT/zero_shot_lm/seed_{seed}/ and $OUT/random/seed_{seed}/
# =============================================================================
set -euo pipefail
GPU_PARTITION="${GPU_PARTITION:-artxpro6000}"
GPU_QOS="${GPU_QOS:-gpu-normal}"
GPU_GRES="${GPU_GRES:-gpu:rtx_pro_6000:1}"
WALL_TIME="${WALL_TIME:-24:00:00}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
cd /projects/paco0228/ACE
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace
OUT="/scratch/alpine/paco0228/ACE/results/curc_30node_expansion_controls"
mkdir -p "$OUT/logs"
SEEDS="${SEEDS:-2022 2023 2024 2025 2026}"

cell_done() {  # same episode-threshold convention as the other suites
    local nl; nl=$(find "$1" -name node_losses.csv 2>/dev/null | head -1); [[ -n "$nl" ]] || return 1
    python -c "import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); sys.exit(0 if int(d['episode'].max())>=$2 else 1)" "$nl" 2>/dev/null
}
N=0
for SEED in $SEEDS; do
    if [ "$SKIP_COMPLETED" = "1" ] && cell_done "$OUT/zero_shot_lm/seed_$SEED" 299; then echo "  SKIP (done): zero_shot_lm seed=$SEED"; else
    JOB=$(sbatch --parsable --job-name="xctl_zsl_s${SEED}" \
        --partition=$GPU_PARTITION --qos=$GPU_QOS --nodes=1 --ntasks=1 --gres=$GPU_GRES \
        --cpus-per-task=8 --mem=64G --time=$WALL_TIME \
        --output="$OUT/logs/zero_shot_lm_seed${SEED}_%j.out" --error="$OUT/logs/zero_shot_lm_seed${SEED}_%j.err" \
        --export=ALL,SEED=$SEED,OUT="$OUT/zero_shot_lm",NO_DPO_FLAG="--no_dpo" jobs/curc_large_scale_seed.sh)
    echo "  Submitted: ACE-w/o-DPO 30-node seed=$SEED -> Job $JOB"; N=$((N+1)); fi
    # Random on these seeds is produced by the metric-audit re-run
    # (jobs/curc_submit_metric_audit_reruns.sh, suite t2_30node, EPISODES=300).
done
echo; echo "$N job(s) submitted. Aggregate with: python scripts/analysis/aggregate_policy_update_suite.py --root $OUT --min-episode 0"

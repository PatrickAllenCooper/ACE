#!/bin/bash
# =============================================================================
# DPO-alternative comparison on the 5-node benchmark: dpo (paper default) vs
# sft_best (imitate the best lookahead candidate only) vs ranking (pairwise
# Bradley-Terry loss with no reference-policy term).
#
# 3 policy_update modes x 5 seeds = 15 jobs.
#
# Note: per the ICLR resubmission framing, the contribution is LM prior +
# lookahead selection, with the preference-learning rule as one calibration
# component -- a tie between dpo and sft_best/ranking here is an acceptable,
# even informative, result (it says the reference-policy KL anchor or the
# preference-comparison structure isn't where DPO's value comes from), not
# grounds to abandon the method.
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   git pull   # ensure latest --policy_update flag
#   bash jobs/curc_submit_dpo_alternatives.sh
#
# Output: results/curc_dpo_alternatives/{policy_update}/seed_{seed}/
#   Each run writes metrics.csv (with a "policy_update" column) and
#   query_budget.json, same schema as the main ACE runs.
#
# SLURM resources per job:
#   partition : aa100 (A100 GPU)
#   time      : $WALL_TIME, default 24:00:00 (gpu-normal's cap); raise it
#               together with GPU_QOS=gpu-long for from-scratch cells
# =============================================================================

set -euo pipefail

# GPU targeting. Default to RTX Pro 6000 (Aug 2026 Alpine expansion): the
# original aa100/a100-40gb + 32G config OOM'd on every dpo/sft_best cell
# (jobs 30942609-30942618, Aug 7) for the same host-RAM reason bf5's budget-
# fairness suite did -- see jobs/curc_submit_5node_budget_fairness.sh. Those
# same jobs also predate the HF Hub 429 fix (resolve_local_hf_snapshot(),
# committed ~Aug 9-10), which is why the surviving sft_best/ranking cells
# "completed" in 12-17s: a 429 on tokenizer load, not a real run. Override
# if needed:
#   GPU_PARTITION=<partition> GPU_QOS=<qos> GPU_GRES=<gres> bash jobs/<this script>
#
# Wall time raised 8h -> 24h (Sept 8): at the 8h cap, all 5 `dpo` cells and 2
# of 5 `sft_best` cells TIMEOUT'd (jobs 32208655-662). This worker runs 200
# episodes on the 5-node benchmark, the same workload the bf5 budget-fairness
# suite measured at 5h51m-7h50m per seed -- i.e. 8h was always marginal. 24h
# is the gpu-normal QoS cap and is the same fix applied to the 30-node
# budget-fairness worker in August for this exact reason. Runs that already
# have a checkpoint will resume rather than restart.
#
# Host RAM stays at 128G deliberately: 13 of 15 cells ran clean at that level
# and the RTX Pro 6000 nodes are only proven to grant ~90-140G per single-GPU
# job, so raising it further risks a job that never schedules.
GPU_PARTITION="${GPU_PARTITION:-artxpro6000}"
GPU_QOS="${GPU_QOS:-gpu-normal}"
GPU_GRES="${GPU_GRES:-gpu:rtx_pro_6000:1}"

# Wall time is overridable because 24h is not always enough. Measured pace on
# ah200 (Sept 10) is ~8-10 min/episode, so a from-scratch 200-episode cell
# needs 27-33h and will TIMEOUT under gpu-normal's 24h QoS cap; a cell
# resuming from a checkpoint usually fits. gpu-normal will reject a request
# above its cap, so raising this only helps in combination with a longer-QoS
# override, e.g.:
#   GPU_QOS=gpu-long WALL_TIME=48:00:00 GPU_PARTITION=ah200 \
#       GPU_GRES=gpu:h200:1 bash jobs/<this script>
WALL_TIME="${WALL_TIME:-24:00:00}"

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine/paco0228/ACE/results/curc_dpo_alternatives"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " DPO-alternative comparison -- CURC SLURM (15 jobs)"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

# MODES is overridable so a subset can be submitted without touching cells
# that are still RUNNING under an earlier submission (SKIP_COMPLETED only
# skips *finished* cells, so a blanket resubmit would duplicate them):
#   MODES=ranking bash jobs/<this script>
MODES="${MODES:-dpo sft_best ranking}"
SEEDS="42 123 456 789 1011"

# Set SKIP_COMPLETED=1 to skip any (mode, seed) cell whose node_losses.csv
# already reached episode 199 (the 200-episode cap). Presence of the file is
# NOT sufficient: a wall-time TIMEOUT leaves a partial node_losses.csv behind
# (this is exactly what jobs 32208655-662 produced), and treating those as
# done would silently drop cells from the aggregate. Same convention as
# jobs/curc_submit_5node_budget_fairness.sh.
#
# NOTE: this is a completed-output check only -- it does NOT detect a
# duplicate still PENDING/RUNNING in the queue. Always check
# `squeue -u $USER` before resubmitting, or two jobs will write the same
# output directory concurrently.
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

cell_done() {
    local mode=$1 seed=$2
    local nl
    nl=$(find "$OUT/${mode}/seed_${seed}" -name node_losses.csv 2>/dev/null | head -1)
    [[ -n "$nl" ]] || return 1
    python -c "
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
sys.exit(0 if 'episode' in df.columns and int(df['episode'].max()) >= 199 else 1)
" "$nl" 2>/dev/null
}

for MODE in $MODES; do
    for SEED in $SEEDS; do
        if [ "$SKIP_COMPLETED" = "1" ] && cell_done "$MODE" "$SEED"; then
            echo "  SKIP (done): dpoalt_${MODE} seed=$SEED"
            continue
        fi
        JOB=$(sbatch --parsable \
            --job-name="dpoalt_${MODE:0:4}_s${SEED}" \
            --partition=$GPU_PARTITION --qos=$GPU_QOS \
            --nodes=1 --ntasks=1 --gres=$GPU_GRES \
            --cpus-per-task=8 --mem=128G \
            --time=$WALL_TIME \
            --output="$OUT/logs/${MODE}_seed${SEED}_%j.out" \
            --error="$OUT/logs/${MODE}_seed${SEED}_%j.err" \
            --export=ALL,POLICY_UPDATE=$MODE,SEED=$SEED,OUT=$OUT \
            jobs/curc_dpo_alternative_seed.sh)
        echo "  Submitted: policy_update=$MODE seed=$SEED -> Job $JOB"
    done
done

echo ""
echo "15 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally with:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"

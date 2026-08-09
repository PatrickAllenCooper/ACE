#!/bin/bash
# =============================================================================
# E3: 100-node scaling frontier, enabled by the Aug 2026 Alpine expansion.
#
# Extends the paper's scaling figure (5/15/30/50 nodes) to 100 nodes (2x the
# previous ceiling). Two things the new hardware fixes that the 50-node cell
# could not avoid:
#
#   1. Wall-time truncation: on aa100/gpu-normal, ACE-50 was capped at 24h
#      per submission and had to checkpoint-resume across several reruns of
#      curc_submit_scaling.sh to reach its plateau, which the paper carries
#      as an honesty caveat. gpu-long gives a single 7-day window -- ACE-100
#      runs to its plateau in one job, no caveat needed.
#   2. Memory: 100-node compact-prompt ACE (bf16 + gradient checkpointing,
#      both auto-enabled by --large_scale >= 50) previously had to fit an
#      A100's 40/80GB; H200's 141GB gives comfortable headroom.
#
# Methods (compact prompt is the scaling enabler beyond ~50 nodes, per
# HuggingFacePolicy's prompt_strategy):
#   ace          : full ACE (LM prior + lookahead + DPO), --lookahead_on_student
#                  so the reported query budget stays honest at this scale
#   zero_shot_lm : LM prior + lookahead-selection only (--no_dpo), same
#                  student-mode lookahead, for the LM-prior-vs-DPO decomposition
#   random       : passive MLP-learner baseline (CPU, acpu partition)
#   round_robin  : passive MLP-learner baseline (CPU, acpu partition)
#
# 3 seeds per method (matching the existing scaling sweep's seed count at
# N=15/30/50).
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   git pull
#   bash jobs/curc_submit_100node_frontier.sh
#
# Output: results/curc_100node_frontier/nodes100/{method}/seed_{seed}/...
#
# GPU targeting: defaults to ah200/gpu-long (7-day QoS); override via
# GPU_PARTITION/GPU_QOS/GPU_GRES if H200's gpu-long pool (5 total, 2/user) is
# saturated, e.g. to fall back to RTX Pro 6000's gpu-long allocation:
#   GPU_PARTITION=artxpro6000 GPU_QOS=gpu-long GPU_GRES=gpu:rtx_pro_6000:1 \
#       bash jobs/curc_submit_100node_frontier.sh
# =============================================================================

set -euo pipefail

# Default to RTX Pro 6000 (gpu-long): the 100-node ACE/ZSL runs use the
# default 1.5B policy + compact prompt + auto bf16/checkpointing at N>=50,
# which fits in 96GB. Routing here avoids the contested H200 pool.
GPU_PARTITION="${GPU_PARTITION:-artxpro6000}"
GPU_QOS="${GPU_QOS:-gpu-long}"
GPU_GRES="${GPU_GRES:-gpu:rtx_pro_6000:1}"

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine/paco0228/ACE/results/curc_100node_frontier"
mkdir -p "$OUT/logs"
WORKER="jobs/curc_scaling_seed.sh"

SEEDS="${SEEDS:-42 123 456}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
# GPU_ONLY=1 skips CPU baselines (already completed on 2026-08-07).
GPU_ONLY="${GPU_ONLY:-0}"
METHODS_GPU="${METHODS_GPU:-ace zero_shot_lm}"
METHODS_CPU="${METHODS_CPU:-random round_robin}"
if [ "$GPU_ONLY" = "1" ]; then METHODS_CPU=""; fi

cell_done() {
    local method=$1 seed=$2
    local seed_dir="$OUT/nodes100/${method}/seed_${seed}"
    [[ -d "$seed_dir" ]] || return 1
    find "$seed_dir" \( -name node_losses.csv -o -name summary.csv \) 2>/dev/null | grep -q .
}

echo "================================================================"
echo " 100-node scaling frontier -- CURC SLURM"
echo " Output : $OUT   Seeds: $SEEDS   Partition: $GPU_PARTITION/$GPU_GRES"
echo " Started: $(date)"
echo "================================================================"

for SEED in $SEEDS; do
    for METHOD in $METHODS_GPU; do
        if [ "$SKIP_COMPLETED" = "1" ] && cell_done "$METHOD" "$SEED"; then
            echo "  SKIP (done): fr100_${METHOD} seed=$SEED"
            continue
        fi
        local_name="fr100_${METHOD:0:3}_s${SEED}"
        wall="7-00:00:00"
        [ "$METHOD" = "zero_shot_lm" ] && wall="2-00:00:00"
        JOB=$(sbatch --parsable \
            --job-name="$local_name" \
            --partition="$GPU_PARTITION" --qos="$GPU_QOS" \
            --nodes=1 --ntasks=1 --gres="$GPU_GRES" \
            --cpus-per-task=8 --mem=96G \
            --time="$wall" \
            --output="$OUT/logs/${METHOD}_seed${SEED}_%j.out" \
            --error="$OUT/logs/${METHOD}_seed${SEED}_%j.err" \
            --export=ALL,SCALE=100,METHOD=$METHOD,SEED=$SEED,OUT=$OUT,PROMPT_STRATEGY=compact,PROMPT_TOP_M=8,LOOKAHEAD_STUDENT=1,EPISODES=100 \
            "$WORKER")
        echo "  Submitted: $local_name -> Job $JOB ($GPU_PARTITION)"
    done

    for METHOD in $METHODS_CPU; do
        if [ "$SKIP_COMPLETED" = "1" ] && cell_done "$METHOD" "$SEED"; then
            echo "  SKIP (done): fr100_${METHOD} seed=$SEED"
            continue
        fi
        JOB=$(sbatch --parsable \
            --job-name="fr100_${METHOD:0:3}_s${SEED}" \
            --partition=acpu --qos=cpu-normal \
            --nodes=1 --ntasks=1 \
            --cpus-per-task=4 --mem=16G \
            --time=10:00:00 \
            --output="$OUT/logs/${METHOD}_seed${SEED}_%j.out" \
            --error="$OUT/logs/${METHOD}_seed${SEED}_%j.err" \
            --export=ALL,SCALE=100,METHOD=$METHOD,SEED=$SEED,OUT=$OUT,EPISODES=150 \
            "$WORKER")
        echo "  Submitted: fr100_${METHOD} seed=$SEED -> Job $JOB"
    done
done

echo ""
echo "Monitor: squeue -u \$USER | grep fr100"
echo "Logs in: $OUT/logs/"
echo ""
echo "ACE/zero_shot_lm use gpu-long's single 7-day/2-day window -- no"
echo "checkpoint-resume resubmission should be needed, but the worker's"
echo "stable per-seed output dir supports it if a job is preempted."

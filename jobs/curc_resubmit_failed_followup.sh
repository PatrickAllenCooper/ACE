#!/bin/bash
# =============================================================================
# Targeted resubmission for the 17 follow-up jobs that did not produce results
# in the May 11-12 batch:
#
#   anon30   ACE  x 5  (FAILED in 25-85s on int(node[1:]) bug, now fixed)
#   anon30   ZSL  x 5  (same bug, now fixed)
#   nodes50  ACE  x 5  (4 OOM during DPO, 1 TIMEOUT; bf16 + grad-ckpt fix)
#   nodes50  ZSL  x 2  (s42, s1011 hit 10h wall time; extend to 12h)
#
# Keeps the 3 already-COMPLETED runs (zsl_n50 s123, s456, s789).
#
# Code fixes that make this batch viable:
#   1. ace_experiments.py: _LargeGroundTruthSCM.mechanisms uses node_idx
#      lookup instead of int(node[1:]), so anon30 generate() no longer
#      crashes on names like 'n_d447'.
#   2. HuggingFacePolicy: gradient_checkpointing kwarg + dtype kwarg;
#      driver enables both at --large_scale 50 (bf16 weights, ckpt activations)
#      to fit a 40 GB A100 during the 4 forward passes of DPO.
#
# Usage (after `git pull` brings these fixes onto CURC):
#   cd /projects/paco0228/ACE
#   bash jobs/curc_resubmit_failed_followup.sh
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/projects/paco0228/ACE/results/curc_30node_followup"
mkdir -p "$OUT/logs"

SEEDS_ALL="42 123 456 789 1011"
# zsl_n50 already has 3 completed seeds; only resubmit the two timeouts.
SEEDS_ZSL_N50="42 1011"

echo "================================================================"
echo " 30/50-node Follow-up RESUBMISSION (17 jobs)"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

# -----------------------------------------------------------------------------
# anon30 / ACE  (5 jobs, 8h)
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 1a: anon30 ACE (5 jobs) <<<"
for SEED in $SEEDS_ALL; do
    JOB=$(sbatch --parsable \
        --job-name="ace_a30r_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=64G \
        --time=08:00:00 \
        --output="$OUT/logs/ace_anon30_resub_seed${SEED}_%j.out" \
        --error="$OUT/logs/ace_anon30_resub_seed${SEED}_%j.err" \
        --export=ALL,CONDITION=anon30,METHOD=ace,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_followup_seed.sh)
    echo "  Submitted: ACE anon30 seed=$SEED -> Job $JOB"
done

# -----------------------------------------------------------------------------
# anon30 / ZSL  (5 jobs, 8h)
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 1b: anon30 Zero-shot LM (5 jobs) <<<"
for SEED in $SEEDS_ALL; do
    JOB=$(sbatch --parsable \
        --job-name="zsl_a30r_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=64G \
        --time=08:00:00 \
        --output="$OUT/logs/zsl_anon30_resub_seed${SEED}_%j.out" \
        --error="$OUT/logs/zsl_anon30_resub_seed${SEED}_%j.err" \
        --export=ALL,CONDITION=anon30,METHOD=zero_shot_lm,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_followup_seed.sh)
    echo "  Submitted: ZSL anon30 seed=$SEED -> Job $JOB"
done

# -----------------------------------------------------------------------------
# nodes50 / ACE  (5 jobs, 14h: 12h was barely enough for the timeout case;
# bf16+ckpt should make per-episode faster, but give headroom)
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 2a: nodes50 ACE (5 jobs, 14h) <<<"
for SEED in $SEEDS_ALL; do
    JOB=$(sbatch --parsable \
        --job-name="ace_n50r_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=78G \
        --time=14:00:00 \
        --output="$OUT/logs/ace_nodes50_resub_seed${SEED}_%j.out" \
        --error="$OUT/logs/ace_nodes50_resub_seed${SEED}_%j.err" \
        --export=ALL,CONDITION=nodes50,METHOD=ace,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_followup_seed.sh)
    echo "  Submitted: ACE nodes50 seed=$SEED -> Job $JOB"
done

# -----------------------------------------------------------------------------
# nodes50 / ZSL  (2 jobs, 12h: only the two TIMEOUTs)
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 2b: nodes50 Zero-shot LM resub (2 jobs, 12h) <<<"
for SEED in $SEEDS_ZSL_N50; do
    JOB=$(sbatch --parsable \
        --job-name="zsl_n50r_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=78G \
        --time=12:00:00 \
        --output="$OUT/logs/zsl_nodes50_resub_seed${SEED}_%j.out" \
        --error="$OUT/logs/zsl_nodes50_resub_seed${SEED}_%j.err" \
        --export=ALL,CONDITION=nodes50,METHOD=zero_shot_lm,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_followup_seed.sh)
    echo "  Submitted: ZSL nodes50 seed=$SEED -> Job $JOB"
done

echo ""
echo "Total: 17 jobs submitted."
echo "Monitor with:  squeue -u \$USER --name=ace_a30r,zsl_a30r,ace_n50r,zsl_n50r"
echo "Logs in:       $OUT/logs/"

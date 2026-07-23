#!/bin/bash
# =============================================================================
# CLEAN follow-up resubmission. Run AFTER curc_verify_dpo_fix.sh confirms the
# gradient-checkpointing requires_grad fix is working (commit b24f6f1 or later).
#
# Drops the contaminated cells from the May 23 batch:
#   - anon30 ACE  (ALL 5 contaminated by broken DPO; rerun all)
#   - nodes50 ACE (ALL 5 contaminated by broken DPO; rerun all)
#   - anon30 ZSL  (4 valid seeds; rerun s456 only)
# Keeps the 5 nodes50 ZSL seeds (verified clean).
#
# Total: 11 jobs.
#
# Wall times tuned to May 26 evidence:
#   - anon30 ACE: longer (n_xxxx) prompts force gradient checkpointing on at
#     30 nodes too (40 GB A100 OOMs without it). Checkpointing adds ~50%
#     compute -> 14h was insufficient. Bumped to 22h.
#   - nodes50 ACE: working DPO ran ~57 min/episode (vs broken DPO's 5.5
#     min/ep that wasn't actually training). 120 episodes is infeasible at
#     CURC's 24h cap. Reduced to 30 episodes (still gives a clear best/final
#     read) at 20h.
#   - anon30 ZSL s456: 8h was insufficient last time -> 10h cap.
#
# Usage (after `git pull` on CURC):
#   cd /projects/paco0228/ACE
#   bash jobs/curc_verify_dpo_fix.sh           # verify first
#   # ...wait for verify .err to show no requires_grad/0.693 warnings...
#   bash jobs/curc_clean_followup_resubmit.sh
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine1/paco0228/ACE/results/curc_30node_followup"
mkdir -p "$OUT/logs"

SEEDS_ALL="42 123 456 789 1011"

echo "================================================================"
echo " 30/50-node Follow-up CLEAN RESUBMIT (11 jobs)"
echo " Requires commit b24f6f1+ (gradient-checkpointing fix)"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

# -----------------------------------------------------------------------------
# anon30 / ACE  (5 jobs, 14h)
# At 30 nodes the new code disables checkpointing and bf16; this matches the
# canonical 30-node ACE configuration the paper is anchored to.
# -----------------------------------------------------------------------------
echo ""
echo ">>> anon30 ACE (5 jobs, 22h) <<<"
for SEED in $SEEDS_ALL; do
    JOB=$(sbatch --parsable \
        --job-name="ace_a30c_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=64G \
        --time=22:00:00 \
        --output="$OUT/logs/ace_anon30_clean_seed${SEED}_%j.out" \
        --error="$OUT/logs/ace_anon30_clean_seed${SEED}_%j.err" \
        --export=ALL,CONDITION=anon30,METHOD=ace,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_followup_seed.sh)
    echo "  Submitted: ACE anon30 seed=$SEED -> Job $JOB"
done

# -----------------------------------------------------------------------------
# anon30 / ZSL s456 only  (1 job, 10h)
# -----------------------------------------------------------------------------
echo ""
echo ">>> anon30 ZSL s456 (1 job, 10h) <<<"
for SEED in 456; do
    JOB=$(sbatch --parsable \
        --job-name="zsl_a30c_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=64G \
        --time=10:00:00 \
        --output="$OUT/logs/zsl_anon30_clean_seed${SEED}_%j.out" \
        --error="$OUT/logs/zsl_anon30_clean_seed${SEED}_%j.err" \
        --export=ALL,CONDITION=anon30,METHOD=zero_shot_lm,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_followup_seed.sh)
    echo "  Submitted: ZSL anon30 seed=$SEED -> Job $JOB"
done

# -----------------------------------------------------------------------------
# nodes50 / ACE  (5 jobs, 20h)
# At 50 nodes the new code enables bf16 weights AND gradient checkpointing
# WITH enable_input_require_grads (the missing piece in the May 23 batch).
# -----------------------------------------------------------------------------
echo ""
echo ">>> nodes50 ACE (5 jobs, 20h) <<<"
for SEED in $SEEDS_ALL; do
    JOB=$(sbatch --parsable \
        --job-name="ace_n50c_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=78G \
        --time=20:00:00 \
        --output="$OUT/logs/ace_nodes50_clean_seed${SEED}_%j.out" \
        --error="$OUT/logs/ace_nodes50_clean_seed${SEED}_%j.err" \
        --export=ALL,CONDITION=nodes50,METHOD=ace,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_followup_seed.sh)
    echo "  Submitted: ACE nodes50 seed=$SEED -> Job $JOB"
done

echo ""
echo "Total: 11 clean jobs submitted."
echo "Monitor with:  squeue -u \$USER --name=ace_a30c,zsl_a30c,ace_n50c"
echo "Logs in:       $OUT/logs/"
